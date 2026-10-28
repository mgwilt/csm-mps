# CSM MPS

Local inference for [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b), with optimized Apple Silicon decoding and live audio playback.

## Setup

Use a native Apple Silicon Mac, macOS 14+, Python 3.12, and [uv](https://docs.astral.sh/uv/). Live playback also needs `ffplay` from FFmpeg (`brew install ffmpeg`).

Accept access on the [model page](https://huggingface.co/sesame/csm-1b), then:

```sh
git clone https://github.com/mgwilt/csm-mps.git
cd csm-mps
uv sync --locked
uv run hf auth login
uv run csm download
uv run csm doctor
```

## Run

Start the resident server:

```sh
uv run csm-server
```

In another terminal, play audio as it is generated:

```sh
uv run csm stream "Hello. This starts playing before generation finishes."
uv run csm stream - --max-frames 1500 < examples/passage.txt
```

Or generate a complete WAV and optionally play it afterward:

```sh
uv run csm say "Hello from CSM." -o outputs/hello.wav --play
```

The server listens on `127.0.0.1:7861` and accepts one synthesis at a time. `--max-frames` defaults to 250 (about 20 seconds); text and audio share a 2,048-position context. `sesame` and `sesame-server` remain aliases. See [usage and HTTP API](docs/usage.md) for voice references, device selection, and raw PCM requests.

## What changes

The checkpoint is unchanged. Our depth decoder runs the same 31 codebook steps with a frame-local KV cache and direct output-head selection, avoiding repeated general-purpose `generate()` calls. Stateful Mimi decoding retains convolution history, upsampling overlap, and transformer KV state, so only new frames are decoded. Generation uses BF16; the streaming codec uses FP32. `--engine baseline` selects the original depth loop for WAV comparisons.

## Measured performance

Apple M3 Ultra, 512 GB; warmed model, small pilot cohorts. RTF below 1 is faster than playback.

| Measurement | Result |
| --- | ---: |
| Buffered median RTF, reference → optimized | 1.124 → 0.870 |
| HTTP streaming RTF | 0.90–0.97 |
| First PCM over HTTP | 310–368 ms |
| Longer streamed passage | 25.52 s at RTF 0.903 |

[Methods, correctness checks, and limitations](docs/performance.md) · [Reproduce benchmarks](docs/benchmarking.md) · [Contributing](CONTRIBUTING.md)

Independent project; not affiliated with Sesame. Code: [Apache-2.0](LICENSE). Model weights are downloaded separately under their upstream terms.
