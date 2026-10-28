"""Matched warm CSM baseline/candidate timing with retained codes and audio."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from csm_inference.runtime import ResidentCSM, synchronize

from .cases import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/depth"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--max-frames", type=int, default=250)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime = ResidentCSM(device="mps", codec_dtype="bfloat16", engine="baseline")
    print(f"Loaded in {time.perf_counter() - started:.2f}s", flush=True)
    records = []
    reference_codes = {}
    for candidate in ("baseline", "direct-depth"):
        runtime.select(candidate)
        runtime.codes("A short warmup.", max_frames=25)
        synchronize(runtime.device)
        for index, text in enumerate(CASES):
            for seed in args.seeds:
                synchronize(runtime.device)
                start = time.perf_counter()
                codes = runtime.codes(text, seed, args.max_frames)
                synchronize(runtime.device)
                generated = time.perf_counter()
                audio = runtime.decode(codes)
                synchronize(runtime.device)
                end = time.perf_counter()
                codes_cpu = codes.cpu()
                key = (index, seed)
                if candidate == "baseline":
                    reference_codes[key] = codes_cpu
                reference = reference_codes[key]
                record = {
                    "candidate": candidate,
                    "case": index,
                    "seed": seed,
                    "text": text,
                    "frames": codes.shape[1],
                    "audio_seconds": len(audio) / 24000,
                    "generation_seconds": generated - start,
                    "decode_seconds": end - generated,
                    "total_seconds": end - start,
                    "rtf": (end - start) / (len(audio) / 24000),
                    "codes_exact": torch.equal(reference, codes_cpu),
                    "finite": bool(np.isfinite(audio).all()),
                    "capped": codes.shape[1] >= args.max_frames,
                }
                name = f"{candidate}-{index}-{seed}"
                sf.write(args.output / f"{name}.wav", audio, 24000)
                np.save(args.output / f"{name}.npy", codes_cpu.numpy())
                records.append(record)
                (args.output / "results.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
                if candidate != "baseline" and not record["codes_exact"]:
                    raise RuntimeError(
                        "Candidate changed generated codes; reject pending a separate quality evaluation"
                    )


if __name__ == "__main__":
    main()
