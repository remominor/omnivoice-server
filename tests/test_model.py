"""Tests for model service utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from omnivoice_server.services.model import ModelService


def test_private_omnivoice_021_compatibility_guard_accepts_pinned_runtime():
    from omnivoice_server.omnivoice_compat import (
        SUPPORTED_OMNIVOICE_VERSION,
        installed_version,
        require_private_compatibility,
    )

    assert installed_version() == SUPPORTED_OMNIVOICE_VERSION
    require_private_compatibility()


def test_audio_tokenizer_boundary_wrapper_converts_bfloat16_decode_output(tmp_path):
    class Tokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

        def encode(self, values):
            return values

        def decode(self, _tokens):
            return SimpleNamespace(audio_values=torch.ones(1, 4, dtype=torch.bfloat16))

    model = SimpleNamespace(audio_tokenizer=Tokenizer())
    ModelService._ensure_audio_tokenizer_input_dtype(model)
    assert model.audio_tokenizer.encode(torch.ones(1, dtype=torch.float32)).dtype == torch.bfloat16
    assert model.audio_tokenizer.decode(torch.zeros(1)).audio_values.dtype == torch.float32


def test_modelservice_has_nan_handles_numpy_array_direct():
    arr = np.array([0.0, 1.0, np.nan], dtype=np.float32)
    assert ModelService._has_nan(arr) is True


def test_modelservice_has_nan_handles_numpy_array_in_list():
    arr = np.array([0.0, 1.0, np.nan], dtype=np.float32)
    assert ModelService._has_nan([arr]) is True


def test_modelservice_has_nan_handles_nested_numpy_collections():
    arr = np.array([0.0, 1.0], dtype=np.float32)
    assert ModelService._has_nan([[arr, arr]]) is False


def test_modelservice_has_nan_handles_torch_tensor_with_nan():
    t = torch.tensor([0.0, float("nan"), 1.0])
    assert ModelService._has_nan(t) is True


def test_modelservice_has_nan_handles_torch_tensor_without_nan():
    t = torch.tensor([0.0, 1.0, 2.0])
    assert ModelService._has_nan(t) is False


def test_remember_audio_tokenizer_device_uses_decoder_projection():
    class HeterogeneousTokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.semantic_model = nn.Linear(1, 1, device="meta")
            self.fc2 = nn.Linear(1, 1, device="cpu")

    model = SimpleNamespace(audio_tokenizer=HeterogeneousTokenizer())

    ModelService._remember_audio_tokenizer_device(model)

    assert model._omnivoice_server_audio_tokenizer_device == torch.device("cpu")


def test_audio_tokenizer_decode_moves_codes_to_remembered_device(monkeypatch):
    class FakeCodes:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    codes = FakeCodes()
    model = SimpleNamespace(
        _omnivoice_server_audio_tokenizer_device=torch.device("cuda", 0)
    )
    received = []

    monkeypatch.setattr(torch, "is_tensor", lambda value: value is codes)
    wrapped = ModelService._wrap_audio_tokenizer_decode(received.append, model)

    wrapped(codes)

    assert codes.moved_to == torch.device("cuda", 0)
    assert received == [codes]
