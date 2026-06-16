"""Resource tracking for the OLMo SFT example: time, memory, and GPU activity.

Three signals, recorded while a computation runs:

- **wall time** — overall and per named phase (``perf_counter``);
- **peak GPU memory** — ``torch.cuda.max_memory_allocated`` /
  ``max_memory_reserved``, per phase and overall (always on, zero deps);
- **GPU activity** — sampled in a background thread. The primary sampler is
  **NVIDIA DCGM** (``dcgmi dmon``), which exposes the *meaningful* engine
  counters on A100-class GPUs: SM-active, tensor-pipe-active, DRAM-active and
  GR-engine-active fractions, plus power / framebuffer / temperature. These say
  how hard the GPU is actually working — unlike the coarse "GPU utilization %"
  (percent of wall time at least one kernel ran), which on a single-stream
  backward-heavy job just pins near 100%.

If ``dcgmi`` is missing or its profiling fields are unavailable (e.g. on a MIG
slice, or without the DCGM host engine), the tracker **automatically falls
back** to sampling ``nvidia-smi`` (coarse util% / memory / power). The summary
records which sampler was used.

Usage::

    tracker = ResourceTracker(device)
    with tracker:
        with tracker.measure("load_model"):
            bundle = load_hf_model(...)
        with tracker.measure("analyze"):
            analyze(...)
    tracker.save("resources.json")
    print(tracker.format_summary())

The chosen sampler needs no Python package (``dcgmi`` / ``nvidia-smi`` are
system CLIs). See this example's README for the DCGM note.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO

import torch

_MB = 1024.0 * 1024.0

# DCGM field IDs we stream, in order. The (id, key, kind) tuples drive both the
# `-e` argument and the row parser, so they cannot drift apart.
#   kind "frac": 0..1 engine-active fraction (shown as % in the text summary)
#   kind "pct":  0..100 coarse utilization
#   kind "mb"/"w"/"c": framebuffer MB / power W / temperature C
_DCGM_FIELDS: list[tuple[int, str, str]] = [
    (1001, "gr_engine_active", "frac"),  # DCGM_FI_PROF_GR_ENGINE_ACTIVE
    (1002, "sm_active", "frac"),  # DCGM_FI_PROF_SM_ACTIVE
    (1004, "tensor_active", "frac"),  # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
    (1005, "dram_active", "frac"),  # DCGM_FI_PROF_DRAM_ACTIVE
    (203, "gpu_util_pct", "pct"),  # DCGM_FI_DEV_GPU_UTIL (coarse, for reference)
    (252, "fb_used_mb", "mb"),  # DCGM_FI_DEV_FB_USED
    (155, "power_w", "w"),  # DCGM_FI_DEV_POWER_USAGE
    (150, "gpu_temp_c", "c"),  # DCGM_FI_DEV_GPU_TEMP
]

# nvidia-smi fallback query fields, mapped onto the same keys where they exist.
_SMI_FIELDS: list[tuple[str, str]] = [
    ("utilization.gpu", "gpu_util_pct"),
    ("memory.used", "fb_used_mb"),
    ("power.draw", "power_w"),
    ("temperature.gpu", "gpu_temp_c"),
]

# Keys whose values are 0..1 fractions — rendered as percentages in text.
_FRACTION_KEYS = {"gr_engine_active", "sm_active", "tensor_active", "dram_active"}


@dataclass
class _Sample:
    t: float  # seconds since tracker entry
    vals: dict[str, float | None]


@dataclass
class _Phase:
    name: str
    t_start: float
    t_end: float | None = None
    torch_peak_alloc_mb: float = 0.0
    torch_peak_reserved_mb: float = 0.0


def _to_float(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None  # "N/A" / "[N/A]"


@dataclass
class ResourceTracker:
    """Track wall time, peak GPU memory, and sampled GPU activity (DCGM)."""

    device: torch.device | str = "cuda"
    interval_s: float = 0.25
    gpu_index: int | str | None = None

    _samples: list[_Sample] = field(default_factory=list, init=False)
    _phases: list[_Phase] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _proc: subprocess.Popen | None = field(default=None, init=False)
    _t0: float | None = field(default=None, init=False)
    sampler: str = field(default="none", init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.is_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self._gpu_id = self._resolve_gpu_id() if self.is_cuda else None

    # ---- GPU selection -------------------------------------------------

    def _resolve_gpu_id(self) -> str:
        """Return the GPU id (for ``-i``) of the device torch will use."""
        if self.gpu_index is not None:
            return str(self.gpu_index)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        torch_idx = self.device.index if self.device.index is not None else 0
        if visible:
            tokens = [t.strip() for t in visible.split(",") if t.strip()]
            if 0 <= torch_idx < len(tokens):
                return tokens[torch_idx]  # physical index or UUID
        return str(torch_idx)

    # ---- DCGM sampler (primary) ----------------------------------------

    def _dcgm_cmd(self, count: int | None) -> list[str]:
        fields = ",".join(str(f[0]) for f in _DCGM_FIELDS)
        cmd = [
            "dcgmi",
            "dmon",
            "-e",
            fields,
            "-d",
            str(max(1, int(self.interval_s * 1000))),
            "-i",
            str(self._gpu_id),
        ]
        if count is not None:
            cmd += ["-c", str(count)]
        return cmd

    @staticmethod
    def _parse_dcgm_row(line: str) -> dict[str, float | None] | None:
        """Parse one ``dcgmi dmon`` data row into the field keys, or None."""
        toks = line.split()
        # Data rows look like: "GPU 0  0.50  0.45  ...". Header lines start "#".
        if len(toks) < 2 + len(_DCGM_FIELDS) or toks[0] != "GPU":
            return None
        if not toks[1].lstrip("-").isdigit():
            return None
        values = toks[2 : 2 + len(_DCGM_FIELDS)]
        return {key: _to_float(v) for (_, key, _kind), v in zip(_DCGM_FIELDS, values, strict=True)}

    def _dcgm_available(self) -> bool:
        """One-shot probe: does ``dcgmi dmon`` yield a parseable row?"""
        try:
            out = subprocess.run(
                self._dcgm_cmd(count=1),
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return False
        if out.returncode != 0:
            return False
        return any(self._parse_dcgm_row(ln) is not None for ln in out.stdout.splitlines())

    def _read_dcgm_stream(self, stream: IO[str]) -> None:
        assert self._t0 is not None
        for line in stream:
            if self._stop.is_set():
                break
            vals = self._parse_dcgm_row(line)
            if vals is not None:
                self._samples.append(_Sample(t=time.perf_counter() - self._t0, vals=vals))

    def _start_dcgm(self) -> bool:
        try:
            self._proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                self._dcgm_cmd(count=None),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, OSError):
            return False
        assert self._proc.stdout is not None
        self._thread = threading.Thread(
            target=self._read_dcgm_stream, args=(self._proc.stdout,), daemon=True
        )
        self._thread.start()
        return True

    # ---- nvidia-smi sampler (fallback) ---------------------------------

    def _poll_smi(self) -> _Sample | None:
        assert self._t0 is not None
        query = ",".join(f for f, _ in _SMI_FIELDS)
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "-i",
            str(self._gpu_id),
        ]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5.0, check=True
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return None
        line = out.splitlines()[0] if out else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_SMI_FIELDS):
            return None
        vals = {key: _to_float(parts[i]) for i, (_, key) in enumerate(_SMI_FIELDS)}
        return _Sample(t=time.perf_counter() - self._t0, vals=vals)

    def _run_smi_sampler(self) -> None:
        while not self._stop.is_set():
            sample = self._poll_smi()
            if sample is not None:
                self._samples.append(sample)
            self._stop.wait(self.interval_s)

    # ---- context management --------------------------------------------

    def __enter__(self) -> ResourceTracker:
        self._t0 = time.perf_counter()
        if not self.is_cuda:
            return self
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        if self._dcgm_available() and self._start_dcgm():
            self.sampler = "dcgm"
        else:
            self.sampler = "nvidia-smi"
            self._thread = threading.Thread(target=self._run_smi_sampler, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Time a named phase and record its peak torch memory."""
        assert self._t0 is not None, "use `with tracker:` before measure()"
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        phase = _Phase(name=name, t_start=time.perf_counter() - self._t0)
        self._phases.append(phase)
        try:
            yield
        finally:
            if self.is_cuda:
                torch.cuda.synchronize(self.device)
                phase.torch_peak_alloc_mb = torch.cuda.max_memory_allocated(self.device) / _MB
                phase.torch_peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / _MB
            phase.t_end = time.perf_counter() - self._t0

    # ---- summary -------------------------------------------------------

    @staticmethod
    def _stats(values: list[float | None]) -> dict[str, float] | None:
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return {"mean": sum(vals) / len(vals), "max": max(vals), "min": min(vals)}

    def _all_keys(self) -> list[str]:
        keys: list[str] = []
        for s in self._samples:
            for k in s.vals:
                if k not in keys:
                    keys.append(k)
        return keys

    def _agg(self, samples: list[_Sample]) -> dict[str, dict[str, float] | None]:
        return {k: self._stats([s.vals.get(k) for s in samples]) for k in self._all_keys()}

    def _slice(self, t_start: float, t_end: float | None) -> list[_Sample]:
        end = t_end if t_end is not None else float("inf")
        return [s for s in self._samples if t_start <= s.t <= end]

    def summary(self) -> dict:
        total_wall = (
            max((p.t_end for p in self._phases if p.t_end is not None), default=0.0)
            if self._phases
            else (time.perf_counter() - self._t0 if self._t0 is not None else 0.0)
        )
        return {
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.device) if self.is_cuda else None,
            "gpu_id": self._gpu_id,
            "sampler": self.sampler,
            "sampler_interval_s": self.interval_s,
            "n_gpu_samples": len(self._samples),
            "wall_s_total": total_wall,
            "torch_peak_alloc_mb": max((p.torch_peak_alloc_mb for p in self._phases), default=0.0),
            "torch_peak_reserved_mb": max(
                (p.torch_peak_reserved_mb for p in self._phases), default=0.0
            ),
            "gpu": self._agg(self._samples),
            "phases": [
                {
                    "name": p.name,
                    "t_start": p.t_start,
                    "t_end": p.t_end,
                    "wall_s": (p.t_end - p.t_start) if p.t_end is not None else None,
                    "torch_peak_alloc_mb": p.torch_peak_alloc_mb,
                    "torch_peak_reserved_mb": p.torch_peak_reserved_mb,
                    "gpu": self._agg(self._slice(p.t_start, p.t_end)),
                }
                for p in self._phases
            ],
        }

    def timeline(self) -> list[dict]:
        """Raw per-sample GPU telemetry (for plotting an activity trace)."""
        return [{"t": s.t, **s.vals} for s in self._samples]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"summary": self.summary(), "timeline": self.timeline()}, f, indent=2)

    @staticmethod
    def _fmt_pct(key: str, stat: dict[str, float] | None) -> str | None:
        if stat is None:
            return None
        scale = 100.0 if key in _FRACTION_KEYS else 1.0
        unit = "%" if key in _FRACTION_KEYS or key.endswith("_pct") else ""
        return f"mean {stat['mean'] * scale:.0f}{unit}, max {stat['max'] * scale:.0f}{unit}"

    def format_summary(self) -> str:
        s = self.summary()
        lines = [
            f"device:        {s['device']}" + (f" ({s['gpu_name']})" if s["gpu_name"] else ""),
            f"GPU sampler:   {s['sampler']} ({s['n_gpu_samples']} samples @ {s['sampler_interval_s']}s)",
            f"wall (total):  {s['wall_s_total']:.1f} s",
            f"torch peak:    {s['torch_peak_alloc_mb']:.0f} MB allocated, "
            f"{s['torch_peak_reserved_mb']:.0f} MB reserved",
        ]
        gpu = s["gpu"]
        labels = [
            ("gr_engine_active", "GR engine"),
            ("sm_active", "SM active"),
            ("tensor_active", "tensor core"),
            ("dram_active", "DRAM (mem BW)"),
            ("gpu_util_pct", "GPU util"),
            ("fb_used_mb", "FB mem (MB)"),
            ("power_w", "power (W)"),
            ("gpu_temp_c", "temp (C)"),
        ]
        for key, label in labels:
            stat = gpu.get(key)
            if stat is None:
                continue
            if key in _FRACTION_KEYS or key.endswith("_pct"):
                txt = self._fmt_pct(key, stat)
            else:
                txt = f"mean {stat['mean']:.0f}, max {stat['max']:.0f}"
            lines.append(f"  {label:<14} {txt}")
        lines.append("phases:")
        for p in s["phases"]:
            busy = p["gpu"].get("sm_active") or p["gpu"].get("gpu_util_pct")
            scale = 100.0 if (p["gpu"].get("sm_active")) else 1.0
            bstr = f", busy mean {busy['mean'] * scale:.0f}%" if busy else ""
            wall = f"{p['wall_s']:.1f}s" if p["wall_s"] is not None else "—"
            lines.append(
                f"  {p['name']:<14} {wall:>8}  peak {p['torch_peak_alloc_mb']:.0f} MB{bstr}"
            )
        return "\n".join(lines)
