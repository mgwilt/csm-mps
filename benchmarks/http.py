"""Measure actual loopback HTTP PCM arrival, completion, and busy rejection."""

import argparse
import http.client
import json
import time
import wave
from pathlib import Path

from .cases import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/http"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, text in enumerate(CASES):
        connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=120)
        start = time.perf_counter()
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps({"text": text}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()}")
        request_id = response.getheader("X-Request-ID")
        first = response.read(2)
        first_pcm = time.perf_counter() - start
        busy = http.client.HTTPConnection("127.0.0.1", args.port, timeout=10)
        busy.request("POST", "/v1/audio/speech", json.dumps({"text": "Overlap"}))
        busy_response = busy.getresponse()
        busy_status = busy_response.status
        busy_response.read()
        busy.close()
        if busy_status != 409:
            raise RuntimeError(f"Overlapping synthesis returned {busy_status}, expected 409")
        pcm = first + response.read()
        total = time.perf_counter() - start
        connection.close()
        with wave.open(str(args.output / f"http-{index}.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            output.writeframes(pcm)
        metadata = http.client.HTTPConnection("127.0.0.1", args.port, timeout=10)
        metadata.request("GET", "/metrics")
        metrics = json.loads(metadata.getresponse().read())["last_request"]
        metadata.close()
        if metrics["request_id"] != request_id or metrics["status"] != "complete":
            raise RuntimeError("Request did not complete cleanly")
        duration = len(pcm) / 48000
        record = {
            "case": index,
            "text": text,
            "http_status": response.status,
            "overlap_status": busy_status,
            "first_pcm_seconds": first_pcm,
            "total_seconds": total,
            "audio_seconds": duration,
            "rtf": total / duration,
            "server": metrics,
        }
        results.append(record)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({k: v for k, v in record.items() if k != "server"}), flush=True)


if __name__ == "__main__":
    main()
