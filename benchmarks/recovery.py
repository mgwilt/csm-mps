"""Exercise real HTTP failure recovery, cancellation, then a longer utterance."""

import http.client
import json
import time
import wave
from pathlib import Path


def connection():
    return http.client.HTTPConnection("127.0.0.1", 7861, timeout=120)


def idle():
    for _ in range(100):
        client = connection()
        client.request("GET", "/health")
        busy = json.loads(client.getresponse().read())["busy"]
        client.close()
        if not busy:
            return
        time.sleep(0.05)
    raise RuntimeError("Server did not release request ownership")


def main():
    output = Path("outputs/benchmarks/recovery")
    output.mkdir(parents=True, exist_ok=True)
    checks = {}
    client = connection()
    client.request("POST", "/v1/audio/speech", json.dumps({"text": ""}))
    response = client.getresponse()
    checks["empty_text_status"] = response.status
    response.read()
    client.close()

    client = connection()
    client.request(
        "POST",
        "/v1/audio/speech",
        json.dumps({"text": "This sentence cannot fit into one frame.", "max_frames": 1}),
    )
    response = client.getresponse()
    try:
        response.read()
        checks["cap_signalled_as_incomplete"] = False
    except http.client.IncompleteRead:
        checks["cap_signalled_as_incomplete"] = True
    client.close()
    idle()

    client = connection()
    client.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(
            {"text": "This request will be cancelled before the sentence has finished speaking."}
        ),
    )
    response = client.getresponse()
    response.read(2)
    response.close()
    client.close()
    idle()
    observer = connection()
    observer.request("GET", "/metrics")
    checks["cancelled_request"] = json.loads(observer.getresponse().read())["last_request"]
    observer.close()

    text = (
        "Okay, here's what happened. I went out for a walk because I'd been staring at "
        "the same problem all morning, and I thought a little fresh air might help. "
        "About ten minutes later, it suddenly clicked. The problem wasn't the complicated "
        "part I'd been worrying about. It was one tiny assumption I'd made right at the "
        "beginning. So I stopped, wrote a note on my phone, and headed home. Of course, "
        "by then I'd completely forgotten to buy the milk. But I did finally fix the problem."
    )
    client = connection()
    started = time.perf_counter()
    client.request("POST", "/v1/audio/speech", json.dumps({"text": text, "max_frames": 750}))
    response = client.getresponse()
    first = response.read(2)
    first_pcm = time.perf_counter() - started
    pcm = first + response.read()
    total = time.perf_counter() - started
    client.close()
    idle()
    observer = connection()
    observer.request("GET", "/metrics")
    metrics = json.loads(observer.getresponse().read())["last_request"]
    observer.close()
    arrivals = metrics["arrivals"]
    already_available = 0
    minimum_extra_buffer = 0
    for chunk in arrivals:
        minimum_extra_buffer = max(
            minimum_extra_buffer,
            chunk["seconds"] - arrivals[0]["seconds"] - already_available / 24000,
        )
        already_available += chunk["samples"]
    checks["long_request"] = {
        "text": text,
        "first_pcm_seconds": first_pcm,
        "total_seconds": total,
        "audio_seconds": len(pcm) / 48000,
        "rtf": total / (len(pcm) / 48000),
        "minimum_extra_playback_buffer_seconds": minimum_extra_buffer,
        "server": metrics,
    }
    with wave.open(str(output / "long.wav"), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    (output / "results.json").write_text(json.dumps(checks, indent=2))
    print(
        json.dumps(
            {
                **checks,
                "long_request": {k: v for k, v in checks["long_request"].items() if k != "server"},
            },
            indent=2,
        )
    )
    assert checks["empty_text_status"] == 400
    assert checks["cap_signalled_as_incomplete"]
    assert checks["cancelled_request"]["status"] == "cancelled"
    assert metrics["status"] == "complete" and metrics["eos"]


if __name__ == "__main__":
    main()
