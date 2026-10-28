from types import SimpleNamespace

import torch
from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d

from csm_inference.decoding.mimi import StatefulMimi


def test_causal_convolutions_match_full_decode_with_transpose_bias():
    torch.manual_seed(3)
    config = MimiConfig(use_causal_conv=True, trim_right_ratio=1.0)
    network = torch.nn.Sequential(
        MimiConvTranspose1d(config, 2, 3, kernel_size=4, stride=2, bias=True),
        MimiConv1d(config, 3, 2, kernel_size=7, bias=True),
        MimiConvTranspose1d(config, 2, 1, kernel_size=8, stride=4, bias=True),
        MimiConv1d(config, 1, 1, kernel_size=3, bias=True),
    ).eval()
    values = torch.randn(1, 2, 19)
    with torch.inference_mode():
        expected = network(values)
        state = StatefulMimi(SimpleNamespace(decoder=network))
        try:
            streamed = torch.cat([network(values[..., i : i + 4]) for i in range(0, 19, 4)], dim=-1)
        finally:
            state.close()
        assert torch.allclose(expected, streamed, atol=1e-6)
        assert torch.equal(network(values), expected)  # Original forwards restored.
