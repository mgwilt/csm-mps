# Usage

## Runtime and model access

Use Python 3.12 and `uv sync --locked`. The pinned Torch/Transformers versions matter because the optimized decoder adapts their model internals. The qualified GPU path is native arm64 macOS with MPS. CPU/CUDA file-synthesis routes exist, but their speed and streaming behavior are not qualified by the recorded Mac measurements.

Accept the CSM-1B access conditions on Hugging Face and authenticate using `uv run hf auth login`. `uv run csm download` caches only the Transformers checkpoint and processor files. The standard Hugging Face cache and login are reused; no model weights or credentials belong in this repository. Cached inference runs locally without a hosted generation API. `csm doctor --offline` skips the online access check.

## Streaming playback

Start `uv run csm-server` in one terminal. Wait for its listening message, then run:

```sh
uv run csm stream "The first sentence plays while the next is still being generated."
uv run csm stream - --max-frames 1500 < examples/passage.txt
uv run csm stream "Use another server port." --port 7862
```

Playback requires `ffplay` on PATH. Ctrl-C stops the player and closes the HTTP connection, cancelling the active synthesis. Metrics distinguish first PCM delivered to the player from network completion and player-process completion. They do not measure acoustic output latency.

`--max-frames` and the compatibility spelling `--max-audio-tokens` set the same hard generation limit, from 1 to 1500. At 12.5 frames per second, the default 250 allows about 20 seconds. Long text and audio context can exhaust the model's 2048-position context sooner. A limit can truncate speech; incomplete HTTP streams are errors, not successful audio. Seeds help comparisons but do not guarantee equal results across hardware or dependency changes.

## File synthesis and reference voices

```sh
uv run csm say "Write a WAV." -o outputs/speech.wav
uv run csm say "Play the completed WAV." --play
uv run csm say "Compare the reference depth loop." --engine baseline
uv run csm say "Use the CPU." --device cpu
uv run csm say "Continue in this voice." \
  --reference-audio reference.wav \
  --reference-text "The exact words spoken in the reference recording."
```

File synthesis writes a 24 kHz mono PCM16 WAV plus `.txt` and `.json` sidecars. `--play` starts after the complete WAV is generated; use `stream` for incremental playback. References must be 24 kHz WAV audio with a matching transcript; stereo is mixed to mono. `--speaker` selects ID 0 or 1. The base model does not supply fixed named voices, and unconditioned voices can vary.

Without an explicit device, file synthesis selects CUDA, then MPS, then CPU. The server defaults to MPS; override with `csm-server --device cpu`. A restricted process sandbox can hide Metal devices. Run from a normal native terminal if MPS is unexpectedly unavailable. Sharded weights load on CPU before transfer to avoid the observed direct-to-MPS loader crash.

The runtime uses the Transformers implementation. It does not apply the standalone upstream generator's SilentCipher watermark. Identify shared outputs as synthetic. This is the released base CSM checkpoint, not Sesame's hosted conversational system.

## HTTP API

The loopback server keeps one model resident. `GET /health` reports readiness, busy state and audio format; `GET /metrics` reports the last synthesis's status and chunk-arrival timings. The server provides no authentication or TLS and binds only to `127.0.0.1`.

```sh
curl --fail http://127.0.0.1:7861/health
curl --fail --no-buffer http://127.0.0.1:7861/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"text":"Welcome. These audio chunks arrive during generation.","seed":42,"max_frames":250}' \
  --output speech.pcm
```

`POST /v1/audio/speech` accepts:

| Field | Default | Bounds |
| --- | --- | --- |
| `text` | Required | 1–4000 characters; nonempty after trimming |
| `seed` | 42 | Unsigned 32-bit integer |
| `max_frames` | 250 | 1–1500, subject to the context limit |

The response is HTTP-chunked raw **signed 16-bit little-endian mono PCM at 24 kHz**, not a WAV. Ordinary chunks contain four codec frames (320 ms); the final chunk can be shorter. Headers expose `X-Audio-Format`, `X-Sample-Rate`, `X-Channels` and `X-Request-ID`.

Invalid input returns 400; overlapping synthesis returns 409. Disconnects unwind generation and restore request-local codec state before releasing ownership. A cap or failure after headers closes the body without a successful terminal HTTP chunk. Clients must detect incomplete transfers and discard incomplete audio. Voice references are currently available through file synthesis, not the HTTP endpoint.
