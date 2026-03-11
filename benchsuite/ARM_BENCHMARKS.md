# ARM Benchmark Runner

This document covers the Apple Silicon benchmark harness in
`benchsuite/arm_bench.py`.

## What it does

`arm_bench.py` runs a focused set of ripgrep benchmarks for ARM and Apple
Silicon machines. It adds:

- interleaved multi-config sampling
- thermal checks and cooldowns
- warm and cold cache modes
- optional interleaved A/B binary comparison
- JSON, CSV, and human-readable summaries
- Apple Silicon scheduler telemetry for the
  `thread_scaling_contention` scenario

## Prerequisites

- macOS on Apple Silicon is the intended platform
- Python 3
- Rust toolchain if you want the script to build ripgrep for you
- benchmark corpora downloaded into a suite directory
- `sudo` access if you want:
  - `powermetrics` telemetry
  - cold-cache runs
  - `thread_scaling_contention`

Important: `arm_bench.py` uses non-interactive `sudo -n`. That means your sudo
credentials must already be cached, or your account must have passwordless sudo
for the relevant commands.

## Benchmark Corpus Setup

Create or choose a suite directory, then download the corpora you need.

```bash
mkdir -p /tmp/benchsuite
./benchsuite/benchsuite --dir /tmp/benchsuite --download linux subtitles-en
```

For a larger one-time setup:

```bash
./benchsuite/benchsuite --dir /tmp/benchsuite --download all
```

## Basic Usage

From the repo root:

```bash
python3 benchsuite/arm_bench.py --dir /tmp/benchsuite
```

That will:

- detect or build a ripgrep binary
- run the default scenario set
- print a summary to stdout

List available scenarios:

```bash
python3 benchsuite/arm_bench.py --list
```

Run a subset:

```bash
python3 benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --scenarios thread_scaling,mmap_vs_read
```

## Contention Scenario

The new scheduler-placement scenario is `thread_scaling_contention`.

It:

- runs the Linux kernel literal-search workload under competing CPU-bound user
  processes
- collects `powermetrics` AMP/IPC telemetry during each measured ripgrep sample
- reports whether P-core dispatch was observed while the system was contended

Run it like this:

```bash
sudo -v
python3 benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --sudo \
  --scenarios thread_scaling_contention \
  --summary /tmp/arm-bench-summary.txt \
  --raw /tmp/arm-bench-raw.csv \
  --json /tmp/arm-bench-results.json
```

Notes:

- `--sudo` is required for this scenario
- if `sudo -n` is not available, the script will fail fast
- the summary now includes a `Scheduler placement:` section for contention runs
- CSV and JSON outputs include per-sample contention and P-core telemetry fields

## Useful Command Lines

Warm-cache run with explicit outputs:

```bash
python3 benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --summary /tmp/arm-bench-summary.txt \
  --raw /tmp/arm-bench-raw.csv \
  --json /tmp/arm-bench-results.json
```

Warm + cold cache:

```bash
sudo -v
python3 benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --sudo \
  --cache both
```

Interleaved A/B comparison against another binary:

```bash
python3 benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --bin ./target/release-lto/rg \
  --compare-bin /tmp/rg-upstream \
  --primary-label candidate \
  --compare-label baseline
```

Override sample counts while preserving thread-scaling minimums:

```bash
python3 benchsuite/arm_bench.py \
  --dir /tmp/benchsuite \
  --samples 12
```

## Output Files

- `--summary`: human-readable report
- `--raw`: CSV with one row per sample
- `--json`: detailed machine-readable output

For `thread_scaling_contention`, the outputs include fields such as:

- whether contention was active
- contender count/type
- telemetry status
- whether P-core dispatch was observed
- P-core share / cycle counters when available

## Environment Guidance

For cleaner data:

- close heavy background applications
- avoid running backups or indexing during the benchmark
- start from a cool machine
- prefer repeated runs over drawing conclusions from a single run

By default, dirty benchmark environments stop the run. You can override that
with:

```bash
python3 benchsuite/arm_bench.py --dir /tmp/benchsuite --allow-dirty-env
```

Use that only when you intentionally want a best-effort run.
