# ARM Benchmark Harness

`benchsuite/arm_bench.py` is the Apple Silicon benchmark harness for ripgrep.
It emits raw CSV, detailed JSON, and a readable text summary for the built-in
ARM scenarios.

## Decision Metadata

Each measured CSV row now includes the benchmark decisions that explain why a
sample behaved the way it did. The JSON output exposes the same data at both
the config level (`decision_metadata`) and the per-sample level (`samples[*]`).

Key fields:

- `threads_selected`
- `threads_reason`
- `apple_pcore_count_detected`
- `mmap_mode`
- `auto_mmap_enabled`
- `auto_mmap_reason`
- `effective_mmap_enabled`
- `path_mode`
- `multiline_enabled`
- `multiline_with_matcher`
- `search_strategy`
- `search_strategy_detail`

The harness mirrors the current `hiargs` and `searcher` logic and also runs a
short-lived `--trace` probe so the metadata reflects the behavior of the binary
under test, including `--compare-bin` builds.

## xctrace Profiling

Optional xctrace capture can be enabled directly from the harness:

```bash
./benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --scenarios mmap_multiline,directory_io \
  --profile time-profiler \
  --profile-scenarios mmap_multiline \
  --profile-samples 1 \
  --json benchsuite/results/arm-profile.json \
  --raw benchsuite/results/arm-profile.csv
```

Available templates:

- `time-profiler`
- `system-trace`
- `poi`

Sample selection:

- `--profile-samples N`
  Replays the first `N` clean samples per config under xctrace.
- `--profile-on-best-delta`
  Replays the config pair with the largest clean median delta in each profiled
  scenario.

Notes:

- Profile replays are diagnostic artifacts and are not included in the timing
  statistics.
- Trace bundles are written under
  `benchsuite/results/profiles/<timestamp>/...`.
- Each CSV row and JSON sample record can point at its matching
  `profile_trace_path` / `profile_summary_path`.
- The profiler also emits a compact JSON summary and, when available, a
  template-specific XML export chosen from the xctrace TOC.

## Common Commands

Run all scenarios:

```bash
./benchsuite/arm_bench.py --dir /tmp/benchsuite
```

Interleaved A/B comparison:

```bash
./benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --compare-bin /tmp/rg-upstream \
  --cache both
```

Profile only the largest delta in thread scaling:

```bash
./benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --scenarios thread_scaling \
  --profile system-trace \
  --profile-on-best-delta
```
