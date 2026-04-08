"""Model loading dispatch. v1 only ships HF causal-LM via load_hf_model."""

from vatis.models.hf import load_hf_model

__all__ = ["load_hf_model"]
