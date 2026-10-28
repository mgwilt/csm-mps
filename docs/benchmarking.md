# Benchmarking

The tools in `benchmarks/` run from a repository checkout. They are not installed in the runtime wheel. Model benchmarks need CSM-1B access, the locked environment and a visible MPS device; unit tests do not.

```sh
uv sync --locked
uv run python -m benchmarks.buffered
uv run python -m benchmarks.streaming
```

`buffered` compares the original depth loop with the specialized loop using three prompts and two seeds. It retains code arrays, WAVs, timings and capped-output flags under `outputs/benchmarks/`. Generated codes must match exactly for the candidate to pass. Both arms use BF16 codec weights to reproduce the recorded comparison.

`streaming` compares the concatenated streaming waveform against a full decode of the same codes. The default candidate uses FP32 stateful Mimi; its maximum absolute sample error must stay below `1e-5`. `--codec prefix` and `--codec-dtype bfloat16` retain exploratory reference paths. Codec precision and warmup policy must be included in comparisons.

With `uv run csm-server` running, run these probes sequentially:

```sh
uv run python -m benchmarks.http
uv run python -m benchmarks.recovery
```

The HTTP probe measures actual client-observed first PCM and completion, retaining WAVs and checking overlapping requests. The recovery probe tests invalid input, a generation cap, cancellation and a longer utterance after failures. It currently targets the default loopback port 7861.

Historical observations are retained in [`results/2026-10-27`](../benchmarks/results/2026-10-27/manifest.json). Prompt text, seeds and durations are historical inputs and must not be edited to make new runs appear matched. Read [the performance report](performance.md) before quoting numbers: cohorts are small, one buffered case was capped, initial codec warmup conditions differed, and no listening study or acoustic latency measurement was performed.

For a new optimization, isolate one change, keep precision and sampling explicit, record checkpoint/dependency identity, and retain unsuccessful cases. Compare correctness before interpreting speed. Benchmark reports should include load time separately from warmed request time, output duration, RTF, time to first PCM, cap/EOS status, and streaming/full-waveform error.
