"""Play the server's PCM incrementally using the installed ffplay binary."""

import http.client
import json
import shutil
import subprocess
import time

from .config import DEFAULT_MAX_FRAMES, DEFAULT_PORT, PCM_BYTES_PER_SECOND, SAMPLE_RATE


def pump_response(response, sink, started):
    first = None
    size = 0
    while True:
        # read() with a large size can wait for several audio chunks. read1()
        # forwards the bytes currently available from one HTTP chunk instead.
        chunk = response.read1(8192)
        if not chunk:
            break
        sink.write(chunk)
        sink.flush()
        if first is None:
            first = time.perf_counter() - started
        size += len(chunk)
    if not size or size % 2:
        raise RuntimeError("Server returned empty or incomplete PCM samples")
    return {
        "first_pcm_to_player_seconds": first,
        "network_complete_seconds": time.perf_counter() - started,
        "audio_seconds": size / PCM_BYTES_PER_SECOND,
    }


def stream_text(text, *, port=DEFAULT_PORT, seed=42, max_frames=DEFAULT_MAX_FRAMES):
    player_path = shutil.which("ffplay")
    if not player_path:
        raise RuntimeError("Streaming playback requires ffplay (the FFmpeg installation).")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    player = None
    completed = False
    try:
        try:
            connection.connect()
        except OSError as error:
            raise RuntimeError(
                f"CSM server is unavailable on port {port}. Start it in another "
                f"terminal: uv run csm-server --port {port}"
            ) from error
        player = subprocess.Popen(
            [
                player_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-nodisp",
                "-autoexit",
                "-probesize",
                "32",
                "-analyzeduration",
                "0",
                "-f",
                "s16le",
                "-ar",
                str(SAMPLE_RATE),
                "-ch_layout",
                "mono",
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )
        started = time.perf_counter()
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps({"text": text, "seed": seed, "max_frames": max_frames}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"Server returned HTTP {response.status}: {response.read(4096).decode(errors='replace')}"
            )
        if (
            response.getheader("X-Audio-Format") != "pcm_s16le"
            or response.getheader("X-Sample-Rate") != str(SAMPLE_RATE)
            or response.getheader("X-Channels") != "1"
        ):
            raise RuntimeError("Unsupported server audio format")
        metrics = pump_response(response, player.stdin, started)
        player.stdin.close()
        if player.wait() != 0:
            raise RuntimeError("ffplay failed to play the audio; check the audio output device")
        completed = True
        metrics["playback_process_complete_seconds"] = time.perf_counter() - started
        return metrics
    except http.client.HTTPException as error:
        raise RuntimeError(
            "Audio stream ended incompletely; generation may have reached its frame limit"
        ) from error
    finally:
        connection.close()  # Also cancels generation when playback is interrupted.
        if player is not None and not completed:
            if player.poll() is None:
                player.terminate()
            try:
                player.wait(timeout=3)
            except subprocess.TimeoutExpired:
                player.kill()
                player.wait()
            if player.stdin is not None and not player.stdin.closed:
                try:
                    player.stdin.close()
                except BrokenPipeError:
                    pass
