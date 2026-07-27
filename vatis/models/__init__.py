"""Model loading dispatch.

Two sources, same output shape (a :class:`~vatis.models.hf.ModelBundle`):

- :func:`load_hf_model` — HuggingFace causal LMs by ``(name, revision)``.
- :func:`load_local_bundle` — native checkpoint bundles written by a training
  run (``00_context.md`` §schema), which additionally expose the saved
  optimizer state aligned to the canonical flattening order.
"""

from vatis.models.hf import ModelBundle, load_hf_model
from vatis.models.local_bundle import (
    BundleMetadata,
    BundleSchemaError,
    CheckpointMeta,
    LocalBundleCheckpoint,
    LocalBundleSource,
    OptimizerStateView,
    load_local_bundle,
    plain_lm_forward,
    resolve_model_factory,
)

__all__ = [
    "BundleMetadata",
    "BundleSchemaError",
    "CheckpointMeta",
    "LocalBundleCheckpoint",
    "LocalBundleSource",
    "ModelBundle",
    "OptimizerStateView",
    "load_hf_model",
    "load_local_bundle",
    "plain_lm_forward",
    "resolve_model_factory",
]
