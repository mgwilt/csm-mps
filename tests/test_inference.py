import numpy as np
import pytest
import torch
from transformers import CsmDepthDecoderConfig
from transformers.models.csm.modeling_csm import CsmDepthDecoderForCausalLM

from csm_inference.decoding.depth import direct_depth_generate
from csm_inference.decoding.stream import StatefulAudioStreamer


@pytest.mark.parametrize("sample", [False, True])
def test_direct_depth_matches_reference_on_small_real_model(sample):
    torch.manual_seed(9)
    config = CsmDepthDecoderConfig(
        hidden_size=32,
        intermediate_size=64,
        backbone_hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        num_codebooks=4,
        pad_token_id=63,
        bos_token_id=0,
        eos_token_id=None,
    )
    model = CsmDepthDecoderForCausalLM(config).eval()
    model.generation_config.do_sample = sample
    model.generation_config.temperature = 0.9 if sample else None
    model.generation_config.top_k = 5 if sample else None
    model.generation_config.min_new_tokens = 3
    model.generation_config.max_new_tokens = 3
    ids = torch.tensor([[0, 12]])
    hidden = torch.randn(1, 64)
    with torch.inference_mode():
        torch.manual_seed(42)
        expected = model.generate(input_ids=ids, backbone_last_hidden_state=hidden)
        torch.manual_seed(42)
        actual = direct_depth_generate(model, ids, hidden)
    assert torch.equal(expected, actual)


class FakeRuntime:
    device = "cpu"

    class model:
        codec_model = None

        class config:
            num_codebooks = 2
            codebook_eos_token_id = 0

    def decode(self, codes):
        return np.repeat(codes[0, :, 0].numpy().astype(np.float32), 1920)


@pytest.fixture
def fake_codec(monkeypatch):
    class Codec:
        def __init__(self, model):
            pass

        def decode(self, codes):
            return FakeRuntime().decode(codes)

        def close(self):
            pass

    monkeypatch.setattr("csm_inference.decoding.mimi.StatefulMimi", Codec)


def test_streaming_flush_has_no_missing_or_duplicated_samples(fake_codec):
    runtime = FakeRuntime()
    chunks = []
    stream = StatefulAudioStreamer(runtime, chunks.append, chunk_frames=4)
    stream.put(torch.tensor([[123, 456, 789]]))  # Initial text prompt, not audio.
    for value in range(1, 10):
        stream.put(torch.tensor([[value, value]]))
    stream.put(torch.zeros(1, 2, dtype=torch.long))
    stream.end()
    stream.end()  # Flush is idempotent.
    assert stream.saw_eos
    assert np.array_equal(np.concatenate(chunks), np.repeat(np.arange(1, 10), 1920))
    assert stream.metrics()["chunks"] == 3


def test_streaming_consumer_failure_stops_generation(fake_codec):
    def cancelled(audio):
        raise BrokenPipeError("Client left")

    stream = StatefulAudioStreamer(FakeRuntime(), cancelled, chunk_frames=4)
    stream.put(torch.tensor([[123]]))
    for _ in range(3):
        stream.put(torch.tensor([[1, 1]]))
    with pytest.raises(BrokenPipeError):
        stream.put(torch.tensor([[1, 1]]))
