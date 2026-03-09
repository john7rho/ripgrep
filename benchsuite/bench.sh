#!/usr/bin/env bash
set -euo pipefail

# BlitzGrep A/B benchmark script
# Reproduces all benchmarks from the blog post:
#   https://byzantine.so/blog/blitzgrep-apple-silicon-performance
#
# Requires: hyperfine, Rust toolchain, ~2GB disk for corpus
# Usage:
#   ./benchsuite/bench.sh              # full run
#   ./benchsuite/bench.sh --dry-run    # validate deps only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
SUITE_DIR="${BENCH_SUITE_DIR:-/tmp/benchsuite}"
LINUX_DIR="$SUITE_DIR/linux"
SUBS_FILE="$SUITE_DIR/subtitles/en.sample.txt"

WARMUPS=5
RUNS=15
DRY_RUN=false

# Parse args
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      echo "Usage: $0 [--dry-run]"
      echo "  --dry-run  Validate dependencies without running benchmarks"
      exit 0
      ;;
  esac
done

# ── Dependency checks ──────────────────────────────────────────────────────

check_dep() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: $1 not found. Please install it." >&2
    exit 1
  fi
}

check_dep hyperfine
check_dep cargo
check_dep git

echo "=== BlitzGrep A/B Benchmark ==="
echo "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
echo "Memory:   $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
echo "OS:       $(uname -srm)"
echo "Rust:     $(rustc --version)"
echo "hyperfine: $(hyperfine --version)"
echo ""

if $DRY_RUN; then
  echo "[dry-run] Dependencies OK."
  # Check corpus
  if [ -d "$LINUX_DIR" ]; then
    echo "[dry-run] Linux kernel corpus found at $LINUX_DIR"
  else
    echo "[dry-run] WARNING: Linux kernel corpus not found at $LINUX_DIR"
    echo "          Download with: git clone --depth 1 https://github.com/torvalds/linux $LINUX_DIR"
  fi
  if [ -f "$SUBS_FILE" ]; then
    echo "[dry-run] Subtitles corpus found at $SUBS_FILE"
  else
    echo "[dry-run] WARNING: Subtitles corpus not found at $SUBS_FILE"
  fi
  exit 0
fi

# ── Build binaries ─────────────────────────────────────────────────────────

UPSTREAM_BIN="/tmp/rg-upstream"
FORK_BIN="/tmp/rg-fork"

echo "--- Building fork binary ---"
cd "$REPO_DIR"
FORK_REV=$(git rev-parse --short HEAD)
cargo build --profile release-lto 2>&1 | tail -1
cp target/release-lto/rg "$FORK_BIN"
echo "Fork binary: $FORK_BIN (rev $FORK_REV)"

echo "--- Building upstream binary ---"
UPSTREAM_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/BurntSushi/ripgrep "$UPSTREAM_DIR" 2>/dev/null
cd "$UPSTREAM_DIR"
UPSTREAM_REV=$(git rev-parse --short HEAD)
cargo build --release 2>&1 | tail -1
cp target/release/rg "$UPSTREAM_BIN"
rm -rf "$UPSTREAM_DIR"
echo "Upstream binary: $UPSTREAM_BIN (rev $UPSTREAM_REV)"
cd "$REPO_DIR"

# ── Corpus check ───────────────────────────────────────────────────────────

if [ ! -d "$LINUX_DIR" ]; then
  echo "--- Downloading Linux kernel corpus ---"
  mkdir -p "$SUITE_DIR"
  git clone --depth 1 https://github.com/torvalds/linux "$LINUX_DIR"
fi

if [ ! -f "$SUBS_FILE" ]; then
  echo "WARNING: Subtitles corpus not found at $SUBS_FILE"
  echo "Single-file benchmarks will be skipped."
  SKIP_SINGLE=true
else
  SKIP_SINGLE=false
fi

# ── Run benchmarks ─────────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"

echo ""
echo "=== Running 7 benchmarks (${WARMUPS} warmups, ${RUNS} runs each) ==="
echo ""

echo "--- 1/7: Dir sparse (PM_RESUME) ---"
hyperfine --warmup "$WARMUPS" --runs "$RUNS" -i \
  --export-json "$RESULTS_DIR/dir-sparse.json" \
  -n upstream "$UPSTREAM_BIN -c PM_RESUME $LINUX_DIR" \
  -n fork "$FORK_BIN -c PM_RESUME $LINUX_DIR"

echo "--- 2/7: Dir dense (return) ---"
hyperfine --warmup "$WARMUPS" --runs "$RUNS" -i \
  --export-json "$RESULTS_DIR/dir-dense.json" \
  -n upstream "$UPSTREAM_BIN -c return $LINUX_DIR" \
  -n fork "$FORK_BIN -c return $LINUX_DIR"

echo "--- 3/7: Dir no-match ---"
hyperfine --warmup "$WARMUPS" --runs "$RUNS" -i \
  --export-json "$RESULTS_DIR/dir-nomatch.json" \
  -n upstream "$UPSTREAM_BIN -c ZZZZNOTFOUNDPATTERNZZZZ $LINUX_DIR" \
  -n fork "$FORK_BIN -c ZZZZNOTFOUNDPATTERNZZZZ $LINUX_DIR"

if ! $SKIP_SINGLE; then
  echo "--- 4/7: Multiline cross-line ---"
  hyperfine --warmup "$WARMUPS" --runs "$RUNS" \
    --export-json "$RESULTS_DIR/multiline-crossline.json" \
    -n upstream "$UPSTREAM_BIN -n -U 'Sherlock\n.*Holmes' $SUBS_FILE > /dev/null" \
    -n fork "$FORK_BIN -n -U 'Sherlock\n.*Holmes' $SUBS_FILE > /dev/null"

  echo "--- 5/7: Single-file literal ---"
  hyperfine --warmup "$WARMUPS" --runs "$RUNS" \
    --export-json "$RESULTS_DIR/singlefile-literal.json" \
    -n upstream "$UPSTREAM_BIN -n -j1 'Sherlock Holmes' $SUBS_FILE > /dev/null" \
    -n fork "$FORK_BIN -n -j1 'Sherlock Holmes' $SUBS_FILE > /dev/null"

  echo "--- 6/7: Multiline literal ---"
  hyperfine --warmup "$WARMUPS" --runs "$RUNS" \
    --export-json "$RESULTS_DIR/multiline-literal.json" \
    -n upstream "$UPSTREAM_BIN -n -U 'Sherlock Holmes' $SUBS_FILE > /dev/null" \
    -n fork "$FORK_BIN -n -U 'Sherlock Holmes' $SUBS_FILE > /dev/null"

  echo "--- 7/7: No-literal regex ---"
  hyperfine --warmup "$WARMUPS" --runs "$RUNS" \
    --export-json "$RESULTS_DIR/noliteral-regex.json" \
    -n upstream "$UPSTREAM_BIN -n -j1 '\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5}' $SUBS_FILE > /dev/null" \
    -n fork "$FORK_BIN -n -j1 '\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5}' $SUBS_FILE > /dev/null"
fi

# ── RSS measurements ──────────────────────────────────────────────────────

echo ""
echo "--- Peak RSS measurements (5 runs each) ---"
RSS_DIR="$RESULTS_DIR/rss"
mkdir -p "$RSS_DIR"

for i in 1 2 3 4 5; do
  /usr/bin/time -l "$UPSTREAM_BIN" -c PM_RESUME "$LINUX_DIR" >/dev/null 2>"$RSS_DIR/upstream-$i.txt"
  /usr/bin/time -l "$FORK_BIN" -c PM_RESUME "$LINUX_DIR" >/dev/null 2>"$RSS_DIR/fork-$i.txt"
done

echo "Upstream peak RSS:"
for i in 1 2 3 4 5; do grep "maximum resident" "$RSS_DIR/upstream-$i.txt"; done
echo "Fork peak RSS:"
for i in 1 2 3 4 5; do grep "maximum resident" "$RSS_DIR/fork-$i.txt"; done

# ── Throughput ─────────────────────────────────────────────────────────────

echo ""
echo "--- Throughput ---"
if [ -d "$LINUX_DIR" ]; then
  CORPUS_BYTES=$(find "$LINUX_DIR" -type f -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END {print s}' || echo 0)
  echo "Linux kernel corpus: $CORPUS_BYTES bytes"
fi

# ── Summary ────────────────────────────────────────────────────────────────

echo ""
echo "=== Summary ==="
echo "Results saved to: $RESULTS_DIR/"
echo "Upstream rev: $UPSTREAM_REV"
echo "Fork rev:     $FORK_REV"
echo ""

# Print table from JSON results
python3 -c "
import json, os, glob

results_dir = '$RESULTS_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*.json')))

print('{:<35} {:>12} {:>12} {:>8}'.format('Benchmark', 'Upstream', 'Fork', 'Speedup'))
print('-' * 70)

for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    with open(f) as fh:
        data = json.load(fh)
    u = data['results'][0]
    fk = data['results'][1]
    u_ms = u['mean'] * 1000
    f_ms = fk['mean'] * 1000
    speedup = u['mean'] / fk['mean']
    print('{:<35} {:>8.1f} ms   {:>8.1f} ms   {:>5.2f}x'.format(name, u_ms, f_ms, speedup))
" 2>/dev/null || echo "(python3 not available for summary table)"

echo ""
echo "Done."
