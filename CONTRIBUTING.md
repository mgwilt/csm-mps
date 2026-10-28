# Contributing

Use Python 3.12 and the locked uv environment:

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
HF_HUB_OFFLINE=1 uv run pytest
uv build
```

Unit tests use small randomly initialized models and fake audio/HTTP data. They do not download model weights or require a GPU. A passing unit suite does not qualify GPU performance.

Keep model loading and generation in `src/csm_inference/runtime.py`, decoder changes in `decoding/`, HTTP ownership in `server.py`, and playback in `client.py`. Benchmark tools and measured receipts belong in `benchmarks/`.

For inference changes, follow [the benchmark guide](docs/benchmarking.md). Retain input text, seeds, checkpoint/dependency identity, capped outputs and failed candidates. Report first PCM separately from acoustic playback latency. Do not replace historical observations with newly estimated values.

Exclude tokens, environment files, model weights, and generated audio from changes. The model remains a separately downloaded dependency. Code contributions use [Apache-2.0](LICENSE).
