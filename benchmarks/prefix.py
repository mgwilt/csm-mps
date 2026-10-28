"""Full-prefix replay reference for streaming benchmark comparisons."""

import time

import numpy as np
import torch

from csm_inference.config import SAMPLES_PER_FRAME
from csm_inference.decoding.stream import FrameStreamer


class PrefixAudioStreamer(FrameStreamer):
    def _decode(self, final):
        if not self.frames:
            return
        started = time.perf_counter()
        codes = torch.stack(self.frames, dim=1)
        audio = self.runtime.decode(codes)
        self.codec_seconds += time.perf_counter() - started
        stable_end = (
            len(audio) if final else max(0, len(audio) - self.lookahead_frames * SAMPLES_PER_FRAME)
        )
        if stable_end <= self.emitted:
            return
        chunk = audio[self.emitted : stable_end].copy()
        if not np.isfinite(chunk).all():
            raise RuntimeError("Non-finite codec output")
        now = time.perf_counter() - self.started
        if self.first_pcm_seconds is None:
            self.first_pcm_seconds = now
        self.chunk_arrivals.append({"seconds": now, "samples": len(chunk)})
        self.emit(chunk)
        self.emitted = stable_end
