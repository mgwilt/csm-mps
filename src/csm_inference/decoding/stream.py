"""Frame delivery and request-scoped stateful Mimi decoding."""

import time

import numpy as np
import torch

from ..config import SAMPLE_RATE


class FrameStreamer:
    """Frame transport, EOS handling, delivery metrics, and flush lifecycle."""

    def __init__(self, runtime, emit, chunk_frames=4, lookahead_frames=2):
        if chunk_frames < 1 or lookahead_frames < 0:
            raise ValueError("Invalid chunk/lookahead size")
        self.runtime, self.emit = runtime, emit
        self.chunk_frames, self.lookahead_frames = chunk_frames, lookahead_frames
        self.frames = []
        self.prompt_pending = True
        self.emitted = 0
        self.saw_eos = False
        self.ended = False
        self.started = time.perf_counter()
        self.first_pcm_seconds = None
        self.chunk_arrivals = []
        self.codec_seconds = 0.0

    def put(self, value):
        if self.prompt_pending:
            self.prompt_pending = False
            return
        if self.ended:
            raise RuntimeError("Received frame after stream ended")
        expected = (1, self.runtime.model.config.num_codebooks)
        if tuple(value.shape) != expected:
            raise ValueError(f"Expected generated frame {expected}, got {value.shape}")
        if bool((value == self.runtime.model.config.codebook_eos_token_id).all()):
            self.saw_eos = True
            return
        self.frames.append(value.to(self.runtime.device))
        if len(self.frames) % self.chunk_frames == 0:
            self._decode(final=False)

    def _decode(self, final):
        raise NotImplementedError

    def end(self):
        if not self.ended:
            self._decode(final=True)
            self.ended = True

    def metrics(self):
        total = time.perf_counter() - self.started
        audio_seconds = self.emitted / SAMPLE_RATE
        return {
            "total_seconds": total,
            "audio_seconds": audio_seconds,
            "rtf": total / audio_seconds if audio_seconds else None,
            "first_pcm_seconds": self.first_pcm_seconds,
            "codec_seconds": self.codec_seconds,
            "chunks": len(self.chunk_arrivals),
            "arrivals": self.chunk_arrivals,
            "eos": self.saw_eos,
        }

    def close(self):
        pass


class StatefulAudioStreamer(FrameStreamer):
    def __init__(self, runtime, emit, chunk_frames=4):
        if chunk_frames < 4:
            raise ValueError(
                "Stateful Mimi needs at least four initial frames for reflection padding"
            )
        super().__init__(runtime, emit, chunk_frames, lookahead_frames=0)
        from .mimi import StatefulMimi

        self.codec = StatefulMimi(runtime.model.codec_model)
        self.decoded_frames = 0

    def _decode(self, final):
        if self.decoded_frames == len(self.frames):
            return
        started = time.perf_counter()
        codes = torch.stack(self.frames[self.decoded_frames :], dim=1)
        chunk = self.codec.decode(codes)
        self.codec_seconds += time.perf_counter() - started
        if not np.isfinite(chunk).all():
            raise RuntimeError("Non-finite codec output")
        now = time.perf_counter() - self.started
        if self.first_pcm_seconds is None:
            self.first_pcm_seconds = now
        self.chunk_arrivals.append({"seconds": now, "samples": len(chunk)})
        self.emit(chunk.copy())
        self.emitted += len(chunk)
        self.decoded_frames = len(self.frames)

    def close(self):
        self.codec.close()
