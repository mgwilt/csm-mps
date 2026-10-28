"""One resident CSM checkpoint shared by file synthesis, serving, and benchmarks."""

import types

import numpy as np
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration

from .config import CONTEXT_LENGTH, DEFAULT_MAX_FRAMES, MAX_FRAMES, MODEL_ID
from .decoding.depth import direct_depth_generate
from .model import resolve_device


def synchronize(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


class ResidentCSM:
    """Single-request runtime; callers must serialize access to generation/state."""

    def __init__(self, device="auto", dtype=None, codec_dtype="float32", engine="direct-depth"):
        self.device = resolve_device(device)
        dtype = dtype or ("float32" if self.device == "cpu" else "bfloat16")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        # Direct sharded loading onto MPS crashes with the tested checkpoint.
        # Materialize on CPU before transferring the complete model.
        self.model = (
            CsmForConditionalGeneration.from_pretrained(
                MODEL_ID, dtype=getattr(torch, dtype), device_map="cpu", attn_implementation="sdpa"
            )
            .eval()
            .to(self.device)
        )
        self.model.codec_model.to(dtype=getattr(torch, codec_dtype))
        self.original_depth_generate = self.model.depth_decoder.generate
        self.select(engine)

    def select(self, engine):
        if engine not in ("baseline", "direct-depth"):
            raise ValueError(f"Unknown inference engine: {engine}")
        self.engine = engine
        self.model.depth_decoder.generate = (
            self.original_depth_generate
            if engine == "baseline"
            else types.MethodType(direct_depth_generate, self.model.depth_decoder)
        )

    def inputs(self, text, speaker=0, reference_audio=None, reference_text=None):
        if not text.strip():
            raise ValueError("Text must not be empty.")
        if speaker not in (0, 1):
            raise ValueError("Speaker must be 0 or 1.")
        if (reference_audio is None) != (reference_text is None):
            raise ValueError("Reference audio and its transcript must be supplied together.")
        if reference_audio is not None:
            conversation = [
                {
                    "role": str(speaker),
                    "content": [
                        {"type": "text", "text": reference_text},
                        {"type": "audio", "path": reference_audio},
                    ],
                },
                {"role": str(speaker), "content": [{"type": "text", "text": text}]},
            ]
            return self.processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.device)
        return self.processor(
            f"[{speaker}]{text}", add_special_tokens=True, return_tensors="pt"
        ).to(self.device)

    @torch.inference_mode()
    def codes(
        self,
        text,
        seed=42,
        max_frames=DEFAULT_MAX_FRAMES,
        streamer=None,
        *,
        speaker=0,
        reference_audio=None,
        reference_text=None,
    ):
        if not 1 <= max_frames <= MAX_FRAMES:
            raise ValueError(f"max_frames must be between 1 and {MAX_FRAMES}.")
        inputs = self.inputs(text, speaker, reference_audio, reference_text)
        if inputs["input_ids"].shape[1] + max_frames > CONTEXT_LENGTH:
            raise ValueError(
                f"Text plus requested audio exceeds the {CONTEXT_LENGTH}-position context."
            )
        torch.manual_seed(seed)
        return self.model.generate(
            **inputs,
            output_audio=False,
            max_new_tokens=max_frames,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            streamer=streamer,
        )

    def has_eos(self, codes):
        return bool((codes[0] == self.model.config.codebook_eos_token_id).all(dim=-1).any())

    @torch.inference_mode()
    def decode(self, codes):
        eos = (codes[0] == self.model.config.codebook_eos_token_id).all(dim=-1).nonzero()
        cutoff = int(eos.min()) if eos.numel() else codes.shape[1]
        if cutoff == 0:
            raise ValueError("CSM generated no audio frames.")
        samples = (
            self.model.codec_model.decode(codes[:, :cutoff].transpose(1, 2))
            .audio_values[0, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("CSM returned empty or non-finite audio.")
        return samples
