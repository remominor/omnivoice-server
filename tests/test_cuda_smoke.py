"""Opt-in CUDA smoke test for real model/loader combinations.

Run one mode per process because each ModelService owns a complete model:

    OMNIVOICE_RUN_CUDA_SMOKE=1 \
    OMNIVOICE_CUDA_SMOKE_MODE=low-vram \
    uv run pytest -q -s tests/test_cuda_smoke.py

The normal test suite skips this module. Set OMNIVOICE_CUDA_SMOKE_REF_AUDIO
to exercise a clone-reference path. Add OMNIVOICE_CUDA_SMOKE_SAVED_PROMPT=1
to require the adjacent .tokens.pt sidecar and measure the normal warm clone
path instead of encoding the reference.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest

RUN_SMOKE = os.getenv("OMNIVOICE_RUN_CUDA_SMOKE", "").lower() in {"1", "true", "yes"}
MODE = os.getenv("OMNIVOICE_CUDA_SMOKE_MODE", "standard").lower()
VALID_MODES = {
    "standard",
    "low-vram",
    "flashinfer",
    "flashinfer-graph",
    "low-vram-flashinfer",
}


@pytest.mark.skipif(not RUN_SMOKE, reason="set OMNIVOICE_RUN_CUDA_SMOKE=1 to run CUDA smoke")
def test_cuda_loader_option_and_generation() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if MODE not in VALID_MODES:
        pytest.fail(f"unsupported OMNIVOICE_CUDA_SMOKE_MODE={MODE!r}")

    from omnivoice_server.config import Settings
    from omnivoice_server.services.model import ModelService

    use_flashinfer = MODE in {"flashinfer", "flashinfer-graph", "low-vram-flashinfer"}
    use_graphs = MODE == "flashinfer-graph"
    use_low_vram = MODE in {"low-vram", "low-vram-flashinfer"}
    cfg = Settings(
        device="cuda",
        num_step=int(os.getenv("OMNIVOICE_CUDA_SMOKE_STEPS", "8")),
        max_concurrent=1,
        low_vram_mode=use_low_vram,
        offload_voice_encoder=True,
        flashinfer_mode=use_flashinfer,
        flashinfer_cuda_graph=use_graphs,
        flashinfer_cuda_graph_max_shapes=int(
            os.getenv("OMNIVOICE_CUDA_SMOKE_MAX_SHAPES", "2")
        ),
        split_cfg_batch=not use_flashinfer,
    )
    service = ModelService(cfg)
    service._load_sync()

    assert service.is_loaded
    if use_low_vram:
        assert service._low_vram_active, "low-VRAM loader silently fell back"
    else:
        assert not service._low_vram_active
    assert service._flashinfer_active is use_flashinfer

    model = service.model
    if use_graphs:
        assert getattr(model, "_fi_enable_cuda_graph", False)
    else:
        assert not getattr(model, "_fi_enable_cuda_graph", False)

    ref_audio = os.getenv("OMNIVOICE_CUDA_SMOKE_REF_AUDIO")
    if ref_audio:
        ref_path = Path(ref_audio)
        assert ref_path.is_file(), f"reference audio not found: {ref_path}"
        ref_text = os.getenv("OMNIVOICE_CUDA_SMOKE_REF_TEXT")
        use_saved_prompt = os.getenv("OMNIVOICE_CUDA_SMOKE_SAVED_PROMPT", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_saved_prompt:
            sidecar = ref_path.with_suffix(".tokens.pt")
            assert sidecar.is_file(), f"saved prompt sidecar not found: {sidecar}"
            prompt = service.get_or_create_voice_clone_prompt(
                "cuda-smoke-saved-profile", str(ref_path), ref_text
            )
        else:
            prompt = service.create_voice_clone_prompt(str(ref_path), ref_text)
        outputs = model.generate(
            text=os.getenv(
                "OMNIVOICE_CUDA_SMOKE_TEXT",
                "This is a CUDA clone smoke test.",
            ),
            voice_clone_prompt=prompt,
            num_step=cfg.num_step,
        )
    else:
        outputs = model.generate(
            text=os.getenv(
                "OMNIVOICE_CUDA_SMOKE_TEXT",
                "This is a CUDA loader smoke test.",
            ),
            num_step=cfg.num_step,
        )

    assert outputs and all(_sample_count(output) > 0 for output in outputs)
    if use_low_vram:
        tokenizer = model.audio_tokenizer
        encoder_names = ("semantic_model", "acoustic_encoder", "encoder_semantic", "fc", "fc1")
        assert all(getattr(tokenizer, name, None) is None for name in encoder_names)

    torch.cuda.synchronize()
    snapshot = {
        "mode": MODE,
        "low_vram_active": service._low_vram_active,
        "flashinfer_active": service._flashinfer_active,
        "cuda_graph_enabled": bool(getattr(model, "_fi_enable_cuda_graph", False)),
        "graph_cache_entries": service.debug_snapshot().get("flashinfer_graph_cache_entries", 0),
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }
    print(json.dumps(snapshot, sort_keys=True))

    del outputs, service
    gc.collect()
    torch.cuda.empty_cache()


def _sample_count(output) -> int:
    """Support OmniVoice's 0.2.1 NumPy output contract and legacy tensors."""
    if hasattr(output, "numel"):
        return int(output.numel())
    return int(output.size)
