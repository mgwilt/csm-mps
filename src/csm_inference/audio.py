"""WAV input/output for offline synthesis."""

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE


def load_reference(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Reference audio must have a {SAMPLE_RATE} Hz sample rate.")
    audio = audio.mean(axis=1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Reference audio must contain finite, nonempty samples.")
    return audio


def save_wav(path, samples):
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("CSM returned empty or non-finite audio.")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
