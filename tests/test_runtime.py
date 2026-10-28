from types import SimpleNamespace

import pytest
import torch

from csm_inference.runtime import ResidentCSM


def test_context_budget_rejected_before_model_execution():
    runtime = ResidentCSM.__new__(ResidentCSM)
    runtime.inputs = lambda *a: {"input_ids": torch.ones(1, 600, dtype=torch.long)}
    with pytest.raises(ValueError, match="context"):
        runtime.codes("long input", max_frames=1500)


@pytest.mark.parametrize("frames", [0, 1501])
def test_frame_budget_validated_without_loading_a_model(frames):
    runtime = ResidentCSM.__new__(ResidentCSM)
    with pytest.raises(ValueError, match="max_frames"):
        runtime.codes("Hello", max_frames=frames)


def test_eos_requires_a_complete_codebook_frame():
    runtime = ResidentCSM.__new__(ResidentCSM)
    runtime.model = SimpleNamespace(config=SimpleNamespace(codebook_eos_token_id=0))
    assert not runtime.has_eos(torch.tensor([[[0, 1], [2, 0]]]))
    assert runtime.has_eos(torch.tensor([[[1, 2], [0, 0]]]))
