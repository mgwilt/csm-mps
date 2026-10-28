# Sesame MPS inference evaluation — October 27, 2026

The pilot supports a persistent Sesame inference server with specialized depth generation and stateful Mimi streaming. On this Mac, the measured server produces audio faster than playback and delivers its first PCM in approximately 0.31–0.37 seconds. These are small local cohorts, not production latency guarantees.

## Reference and environment

- Apple M3 Ultra, 512 GB unified memory, macOS 27.0.1; Python 3.12.14, Torch 2.14.1, Transformers 5.21.0, locked uv dependencies.
- Model: `sesame/csm-1b`, checkpoint revision `c92a71e1c419772e25be7dc14d952c2521a740ab`.
- Generation remains BF16, temperature 0.9 and top-k 50 for backbone and depth. No codebooks, model layers or sampling steps were removed. The server uses FP32 for Mimi.
- No text-generation model was resident during these tests. Measurements exclude download/model-load time and acoustic playback latency.


## Buffered inference: matched six-case pilot

Three prompts × two seeds, one excluded 25-frame warmup per arm, same resident model, same sampling. GPU work was synchronized around timing. Each arm retained codes and audio. Total RTF includes text preparation, code generation, full codec decoding and host transfer.

| Arm | Median RTF | Worst observed RTF | Exact code match |
| --- | ---: | ---: | --- |
| Transformers reference | 1.124 | 1.323 | Reference |
| Specialized depth loop | 0.870 | 0.873 | 6/6 cases |

Median RTF fell about 23%; the candidate remained below 1 in this cohort. Both arms generated 52 seconds of audio across the six cases. One seed/prompt reached the 250-frame cap in both arms and is retained as a capped case, not a complete utterance. Samples were finite. The cases ran sequentially by arm, not as a randomized statistical trial.

The specialized loop uses the same depth model and a fresh frame-local KV cache, directly selects each codebook head by integer index, and applies the effective Transformers temperature/top-k sampling settings. It avoids repeatedly entering general generation machinery. Small real-model tests cover greedy and sampled generation; actual checkpoint runs matched all six generated code arrays, including the capped case.

## Streaming progression

| Candidate | Three observed RTFs | First PCM | Finding |
| --- | --- | --- | --- |
| BF16 full-prefix replay | 1.224, 0.946, 0.942 | 0.294–0.872 s | Works, but growing replay cost; first codec use was cold |
| BF16 stateful Mimi | 0.993, 1.003, 0.970 | 0.300–0.396 s | State retained; numerical differences require care |
| FP32 stateful Mimi, after stream warmup | 0.916, 0.922, 0.912 | 0.300–0.301 s | Selected candidate |

These exploratory rows have different codec warmup/precision conditions and are not a controlled speedup ablation. Raw receipts preserve those differences. The first two are rejected intermediate candidates, not the server's selected configuration.

For the selected three-case run, concatenated streaming PCM had exactly the full-decode sample count. Before PCM16 conversion, maximum absolute waveform error versus full **FP32** Mimi decoding was 0.000000574–0.00000203; RMSE was 0.000000034–0.000000106. BF16 stateful decoding had larger errors (up to 0.0127), which motivated the FP32 codec. This does not claim that FP32 audio is bit-identical to the previous BF16 codec, or replace perceptual evaluation.

The stateful codec retains:

- Causal convolution input history.
- Transposed-convolution overlap, with bias counted once.
- Decoder transformer KV state, isolated per request.

Only newly generated code frames are decoded. Temporary module adapters are restored on completion, errors and disconnects. A CPU convolution test independently compares full versus chunked decoding with biased transposed convolutions. Model-dependent code remains limited to the locked Transformers implementation and one active request.

## Real loopback HTTP delivery

The warmed server was tested through actual HTTP sockets, including an overlapping POST during every short request.

| Request | Audio duration | Client-observed RTF | First PCM | Completion |
| --- | ---: | ---: | ---: | --- |
| Short 1 | 5.60 s | 0.965 | 0.347 s | Complete |
| Short 2 | 3.92 s | 0.962 | 0.329 s | Complete |
| Short 3 | 4.40 s | 0.928 | 0.310 s | Complete |
| Longer passage | 25.52 s | 0.903 | 0.368 s | Complete, EOS |

The longer passage produced 320 ms chunks while generation continued. Its recorded arrival schedule required no additional startup buffer beyond waiting for the first chunk to avoid a simulated playback underrun. This is a producer/arrival check, not an acoustic playback test.

Observed protocol checks:

- Overlapping synthesis rejected with HTTP 409 in all three short probes.
- Empty text rejected with HTTP 400.
- A one-frame generation limit produced an incomplete HTTP body, not a successful stream.
- Client disconnect cancelled the active request; ownership was released.
- The longer request completed after both failure cases, verifying recovery and state restoration.

## Reproduction and retained evidence

Commands are in [the benchmark guide](benchmarking.md). Raw JSON is tracked in `benchmarks/results/2026-10-27/`, including a manifest with checkpoint and dependency identity. Generated audio/code arrays remain under ignored `outputs/benchmarks/`; they are local inspection artifacts. The server prototype is `src/csm_inference/server.py`; it binds only to loopback and does not expose an authenticated public service.

## Next useful hill climbs

1. Repeat on held-out prompts, reference voices and longer utterances, with ASR plus listening checks, interleaved arms, and enough observations for useful tail estimates.
2. Profile the remaining backbone and depth kernel-launch costs. Evaluate static caches/compiled fixed decode steps independently; do not assume CUDA graph optimizations apply to Metal.
3. Measure concurrent text generation and CSM together, interruption responsiveness, backpressure with slow consumers, process memory and sustained service load before treating this as a conversational production server.

A persistent server removes per-command model load costs and provides streaming lifecycle control. Persistence alone does not account for the measured steady-state generation improvement; that comes from the specialized decoder.

The separate `--fast-depth` CLI smoke test completed successfully, but its fresh-process generation RTF was 1.88 for 2.48 seconds of audio. Cold kernel/setup costs matter for short commands; the below-one results above describe a resident, warmed server rather than every CLI invocation.
