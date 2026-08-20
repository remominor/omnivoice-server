"""Compatibility checks for private OmniVoice 0.2.1 integrations.

The normal server path only depends on OmniVoice's public API.  Low-VRAM
loading, split CFG, and FlashInfer intentionally touch private model details,
so keep their version and structural checks in one small module.
"""

from __future__ import annotations

import inspect
from importlib.metadata import PackageNotFoundError, version
from typing import Any

SUPPORTED_OMNIVOICE_VERSION = "0.2.1"


def installed_version() -> str:
    try:
        return version("omnivoice")
    except PackageNotFoundError as exc:
        raise RuntimeError("OmniVoice is not installed") from exc


def require_private_compatibility() -> None:
    """Reject unknown OmniVoice builds before applying private patches."""
    version = installed_version()
    if version != SUPPORTED_OMNIVOICE_VERSION:
        raise RuntimeError(
            "private optimizations require omnivoice=="
            f"{SUPPORTED_OMNIVOICE_VERSION}, found {version}"
        )

    from omnivoice.models.omnivoice import OmniVoice

    required = ("_generate_iterative", "_prepare_inference_inputs", "generate")
    missing = [name for name in required if not hasattr(OmniVoice, name)]
    if missing:
        raise RuntimeError(f"OmniVoice private API is missing: {', '.join(missing)}")

    params = inspect.signature(OmniVoice._generate_iterative).parameters
    if tuple(params)[1:3] != ("task", "gen_config"):
        raise RuntimeError("OmniVoice _generate_iterative signature is unsupported")


def require_flashinfer_compatibility(model: Any) -> None:
    """Validate the loaded Qwen3 layout before mutating it for FlashInfer."""
    require_private_compatibility()
    llm = getattr(model, "llm", None)
    if llm is None:
        raise RuntimeError("OmniVoice model has no language model")
    layers = getattr(llm, "layers", None)
    if not layers:
        raise RuntimeError("OmniVoice model has no Qwen3 decoder layers")
    attention = getattr(layers[0], "self_attn", None)
    if attention is None:
        raise RuntimeError("OmniVoice Qwen3 layer has no self-attention module")
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "head_dim")
    missing = [name for name in required if not hasattr(attention, name)]
    if missing:
        raise RuntimeError(f"Qwen3 attention layout is unsupported: {', '.join(missing)}")
    if not getattr(llm.config, "rope_parameters", None):
        raise RuntimeError("Qwen3 rope parameters are unavailable")
