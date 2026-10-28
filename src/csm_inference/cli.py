"""CSM-1B file synthesis and live playback through the local inference server."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .config import DEFAULT_MAX_FRAMES, DEFAULT_PORT, MAX_FRAMES, MODEL_ID, SAMPLE_RATE
from .model import check_access, download_model, resolve_device


def frame_count(value):
    value = int(value)
    if not 1 <= value <= MAX_FRAMES:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_FRAMES}")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check runtime, device, and model access")
    doctor.add_argument("--offline", action="store_true", help="Skip the model access request")
    commands.add_parser("download", help="Download the CSM-1B Transformers checkpoint")
    stream = commands.add_parser("stream", help="Play incoming audio immediately from csm-server")
    say = commands.add_parser("say", help="Generate a complete WAV, then optionally play it")
    for item in (say, stream):
        item.add_argument("text", help="Text to speak; use - to read stdin")
        item.add_argument("--seed", type=int, default=42)
        item.add_argument(
            "--max-audio-tokens",
            "--max-frames",
            dest="max_frames",
            type=frame_count,
            default=DEFAULT_MAX_FRAMES,
            help="Maximum audio frames (12.5 per second)",
        )
    stream.add_argument("--port", type=int, default=DEFAULT_PORT)
    say.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    say.add_argument("--engine", choices=["direct-depth", "baseline"], default="direct-depth")
    say.add_argument(
        "--fast-depth",
        action="store_const",
        const="direct-depth",
        dest="engine",
        help=argparse.SUPPRESS,
    )
    say.add_argument("--output", "-o", type=Path, default=Path("outputs/speech.wav"))
    say.add_argument("--speaker", type=int, choices=[0, 1], default=0)
    say.add_argument("--reference-audio", type=Path)
    say.add_argument("--reference-text")
    say.add_argument("--play", action="store_true", help="Play the completed WAV after generation")
    return result


def synthesize(text, args):
    from .audio import load_reference, save_wav
    from .runtime import ResidentCSM, synchronize

    reference = load_reference(args.reference_audio) if args.reference_audio else None
    started = time.perf_counter()
    runtime = ResidentCSM(device=args.device, engine=args.engine)
    synchronize(runtime.device)
    loaded = time.perf_counter()
    codes = runtime.codes(
        text,
        args.seed,
        args.max_frames,
        speaker=args.speaker,
        reference_audio=reference,
        reference_text=args.reference_text,
    )
    if not runtime.has_eos(codes):
        raise RuntimeError("Generation reached the frame limit without EOS. Increase --max-frames.")
    samples = runtime.decode(codes)
    synchronize(runtime.device)
    finished = time.perf_counter()
    save_wav(args.output, samples)
    return {
        "model": MODEL_ID,
        "engine": runtime.engine,
        "device": runtime.device,
        "seed": args.seed,
        "sample_rate": SAMPLE_RATE,
        "codec_dtype": "float32",
        "load_seconds": loaded - started,
        "generation_seconds": finished - loaded,
        "audio_seconds": len(samples) / SAMPLE_RATE,
        "rtf": (finished - loaded) / (len(samples) / SAMPLE_RATE),
        "output": str(args.output.resolve()),
        "text": text,
    }


def run(args):
    if args.command == "download":
        print(download_model())
        return 0
    if args.command == "doctor":
        import torch
        import transformers
        from huggingface_hub import get_token

        report = {
            "model": MODEL_ID,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": resolve_device(),
            "hugging_face_login": bool(get_token()),
        }
        failed = False
        if not args.offline:
            try:
                check_access()
                report["model_access"] = "OK"
            except ValueError as error:
                report["model_access"] = str(error)
                failed = True
        print(json.dumps(report, indent=2))
        return int(failed)
    text = sys.stdin.read().strip() if args.text == "-" else args.text.strip()
    if not text:
        raise ValueError("Text must not be empty.")
    if args.command == "stream":
        from .client import stream_text

        metrics = stream_text(text, port=args.port, seed=args.seed, max_frames=args.max_frames)
    else:
        if bool(args.reference_audio) != bool(args.reference_text):
            raise ValueError("Use --reference-audio and --reference-text together.")
        metrics = synthesize(text, args)
        args.output.with_suffix(".txt").write_text(text + "\n")
        args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(text, flush=True)
        if args.play:
            subprocess.run(["afplay", str(args.output)], check=True)
    print(json.dumps(metrics, indent=2), file=sys.stderr)
    return 0


def main():
    args = parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
