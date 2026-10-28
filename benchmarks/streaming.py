"""Verify prefix-replay streaming against the same complete Mimi decode."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmarks.prefix import PrefixAudioStreamer
from csm_inference.decoding.stream import StatefulAudioStreamer
from csm_inference.runtime import ResidentCSM

from .cases import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/stream"))
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--codec", choices=["prefix", "stateful"], default="stateful")
    parser.add_argument("--codec-dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = ResidentCSM(device="mps", codec_dtype=args.codec_dtype)
    runtime.select("direct-depth")
    if args.codec_dtype == "float32":
        runtime.model.codec_model.float()
    runtime.codes("A short warmup.", max_frames=25)
    warm_type = StatefulAudioStreamer if args.codec == "stateful" else PrefixAudioStreamer
    warm = warm_type(runtime, lambda chunk: None, chunk_frames=args.chunk_frames)
    try:
        runtime.codes("A short warmup.", max_frames=25, streamer=warm)
    finally:
        warm.close()
    results = []
    for index, text in enumerate(CASES):
        chunks = []
        streamer_type = StatefulAudioStreamer if args.codec == "stateful" else PrefixAudioStreamer
        streamer = streamer_type(runtime, chunks.append, chunk_frames=args.chunk_frames)
        try:
            codes = runtime.codes(text, streamer=streamer)
        finally:
            streamer.close()
        metrics = streamer.metrics()
        combined = np.concatenate(chunks)
        complete = runtime.decode(codes)
        metrics.update(
            {
                "case": index,
                "text": text,
                "chunk_frames": args.chunk_frames,
                "codec": args.codec,
                "codec_dtype": args.codec_dtype,
                "same_length": len(combined) == len(complete),
                "max_sample_error": float(np.max(np.abs(combined - complete))),
                "rmse": float(np.sqrt(np.mean((combined - complete) ** 2))),
            }
        )
        sf.write(args.output / f"stream-{index}.wav", combined, 24000)
        sf.write(args.output / f"complete-{index}.wav", complete, 24000)
        np.save(args.output / f"codes-{index}.npy", codes.cpu().numpy())
        results.append(metrics)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(metrics), flush=True)
        if (
            args.codec == "stateful"
            and args.codec_dtype == "float32"
            and metrics["max_sample_error"] > 1e-5
        ):
            raise RuntimeError("Stateful FP32 codec failed the full-decode numerical gate")


if __name__ == "__main__":
    main()
