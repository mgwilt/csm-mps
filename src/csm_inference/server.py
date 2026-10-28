"""Loopback-only, single-synthesis CSM PCM streaming server."""

import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import CONTEXT_LENGTH, DEFAULT_MAX_FRAMES, DEFAULT_PORT, MAX_FRAMES, SAMPLE_RATE


def create_server(runtime, port=DEFAULT_PORT, chunk_frames=4):
    import numpy as np

    from .decoding.stream import StatefulAudioStreamer

    if chunk_frames < 4:
        raise ValueError("Stateful streaming requires at least four frames per chunk")
    active = threading.Lock()
    state = {"last_request": None}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def json_response(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.json_response(
                    200,
                    {
                        "status": "ok",
                        "busy": active.locked(),
                        "sample_rate": SAMPLE_RATE,
                        "format": "pcm_s16le",
                        "channels": 1,
                        "candidate": "direct-depth",
                        "codec": "stateful-fp32",
                        "chunk_frames": chunk_frames,
                    },
                )
            elif self.path == "/metrics":
                self.json_response(200, state)
            else:
                self.json_response(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/v1/audio/speech":
                self.json_response(404, {"error": "Not found"})
                return
            self.connection.settimeout(30)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("JSON body must be between 1 and 32768 bytes")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or set(data) - {
                    "text",
                    "seed",
                    "max_frames",
                }:
                    raise ValueError("Supported fields: text, seed, max_frames")
                text = data.get("text", "")
                if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                    raise ValueError("text must contain 1–4000 characters")
                seed, max_frames = data.get("seed", 42), data.get("max_frames", DEFAULT_MAX_FRAMES)
                if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                    raise ValueError("seed must be a uint32 integer")
                if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
                    raise ValueError(f"max_frames must be between 1 and {MAX_FRAMES}")
            except (ValueError, TypeError, OSError) as error:
                self.close_connection = True
                self.json_response(400, {"error": str(error)})
                return
            if not active.acquire(blocking=False):
                self.close_connection = True
                self.json_response(409, {"error": "Synthesis is busy"})
                return

            headers_sent = False
            streamer = None
            request_id = uuid.uuid4().hex
            try:
                if runtime.inputs(text)["input_ids"].shape[1] + max_frames > CONTEXT_LENGTH:
                    raise ValueError(
                        f"Text plus requested audio exceeds the {CONTEXT_LENGTH}-position context"
                    )

                def emit(audio):
                    nonlocal headers_sent
                    if not headers_sent:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Transfer-Encoding", "chunked")
                        self.send_header("X-Audio-Format", "pcm_s16le")
                        self.send_header("X-Sample-Rate", str(SAMPLE_RATE))
                        self.send_header("X-Channels", "1")
                        self.send_header("X-Request-ID", request_id)
                        self.end_headers()
                        headers_sent = True
                    payload = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()  # Backpressure/cancellation stays in the owning worker.

                streamer = StatefulAudioStreamer(runtime, emit, chunk_frames=chunk_frames)
                runtime.codes(text, seed, max_frames, streamer)
                if not streamer.saw_eos:
                    raise RuntimeError("Frame limit reached without EOS; stream is incomplete")
                if not headers_sent:
                    raise RuntimeError("Model returned no audio")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                state["last_request"] = {
                    **streamer.metrics(),
                    "status": "complete",
                    "request_id": request_id,
                }
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                state["last_request"] = {
                    "status": "cancelled",
                    "request_id": request_id,
                }
                self.close_connection = True
            except Exception as error:  # noqa: BLE001 — HTTP worker boundary; release model ownership.
                state["last_request"] = {
                    "status": "failed",
                    "error": str(error),
                    "request_id": request_id,
                }
                if not headers_sent:
                    self.json_response(
                        400 if isinstance(error, ValueError) else 500,
                        state["last_request"],
                    )
                else:
                    # Do not send a successful terminal HTTP chunk after a partial failure.
                    self.close_connection = True
            finally:
                try:
                    if streamer is not None:
                        streamer.close()
                finally:
                    active.release()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(prog="csm-server", description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda", "auto"], default="mps")
    parser.add_argument("--chunk-frames", type=int, default=4)
    args = parser.parse_args()
    if args.chunk_frames < 4:
        parser.error("chunk-frames must be at least four")
    from .decoding.stream import StatefulAudioStreamer
    from .runtime import ResidentCSM

    runtime = ResidentCSM(device=args.device, codec_dtype="float32")
    warm = StatefulAudioStreamer(runtime, lambda chunk: None, chunk_frames=args.chunk_frames)
    try:
        runtime.codes("A short warmup.", max_frames=25, streamer=warm)
    finally:
        warm.close()
    server = create_server(runtime, args.port, args.chunk_frames)
    print(f"CSM streaming at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
