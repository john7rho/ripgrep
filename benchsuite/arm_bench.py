#!/usr/bin/env python3

"""
arm_bench.py -- ARM/Apple Silicon benchmark runner for ripgrep.

This script provides thermal-aware, statistically rigorous benchmarks
targeting Apple Silicon (M1-M5) and other ARM chips. It implements
interleaved A/B execution, Mann-Whitney U significance testing, and
core topology awareness.

Canonical standards are defined in CLAUDE.md section "ARM Benchmark Standards".

Usage:
    ./arm_bench.py --dir /tmp/benchsuite
    ./arm_bench.py --dir /tmp/benchsuite --baseline runs/arm-m5-baseline/raw.csv
    ./arm_bench.py --dir /tmp/benchsuite --compare-bin /tmp/rg-upstream --cache both
    ./arm_bench.py --dir /tmp/benchsuite --scenarios thread_scaling,mmap_vs_read
    ./arm_bench.py --help
"""

import argparse
import csv
import datetime
import json
import math
import os
import os.path as path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import resource
except ImportError:
    # resource module is not available on non-POSIX platforms (e.g., Windows).
    # Gracefully degrade: run_timed will use fallback timing.
    resource = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Corpus constants (match benchsuite/benchsuite exactly)
# ---------------------------------------------------------------------------

SUBTITLES_DIR = 'subtitles'
SUBTITLES_EN_NAME = 'en.txt'
SUBTITLES_EN_NAME_SAMPLE = 'en.sample.txt'
SUBTITLES_EN_NAME_GZ = '%s.gz' % SUBTITLES_EN_NAME
SUBTITLES_EN_URL = (
    'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/en.txt.gz'
)
SUBTITLES_RU_NAME = 'ru.txt'
SUBTITLES_RU_NAME_GZ = '%s.gz' % SUBTITLES_RU_NAME
SUBTITLES_RU_URL = (
    'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/ru.txt.gz'
)

LINUX_DIR = 'linux'
LINUX_CLONE = 'https://github.com/BurntSushi/linux'

# ---------------------------------------------------------------------------
# Benchmark parameters (from CLAUDE.md ARM Benchmark Standards)
# ---------------------------------------------------------------------------

DEFAULT_WARMUPS = 3
DEFAULT_SAMPLES = 10
THREAD_SCALING_SAMPLES = 15
CV_THRESHOLD = 0.10  # 10% -- default; overridden per-scenario
CV_THRESHOLD_SINGLE_FILE = 0.05  # 5% for single-file benchmarks
CV_THRESHOLD_MULTI_FILE = 0.12  # 12% for multi-file/thread-scaling benchmarks
REGRESSION_THRESHOLD = 0.05  # 5% slower
REGRESSION_P_VALUE = 0.05
COOLDOWN_MIN_SECONDS = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def eprint(*args: Any, **kwargs: Any) -> None:
    """Print to stderr."""
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def run_cmd(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a command, printing it to stderr first."""
    eprint('# %s' % ' '.join(cmd))
    kwargs['check'] = True
    return subprocess.run(cmd, **kwargs)


def sysctl_value(key: str) -> Optional[str]:
    """Read a sysctl value, returning None on failure."""
    try:
        result = subprocess.run(
            ['sysctl', '-n', key],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def is_arm() -> bool:
    """Return True if running on ARM architecture."""
    return platform.machine() in ('arm64', 'aarch64')


def is_macos() -> bool:
    """Return True if running on macOS."""
    return sys.platform == 'darwin'


def is_apple_silicon() -> bool:
    """Return True if running on Apple Silicon Mac."""
    return is_arm() and is_macos()



def check_background_interference() -> List[str]:
    """Check for background processes that may interfere with benchmarks.

    Checks for Spotlight indexing and Time Machine activity.
    Returns a list of warning strings (empty if no interference detected).
    """
    interference_warnings: List[str] = []

    # Check Spotlight indexing
    try:
        result = subprocess.run(
            ['mdutil', '-s', '/'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and 'Indexing enabled' in result.stdout:
            interference_warnings.append(
                'Spotlight indexing is enabled on /. This may cause I/O '
                'interference during directory-walk benchmarks. Consider '
                'disabling with: sudo mdutil -i off /'
            )
    except Exception:
        pass

    # Check Time Machine
    try:
        result = subprocess.run(
            ['tmutil', 'status'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and 'Running = 1' in result.stdout:
            interference_warnings.append(
                'Time Machine backup is currently running. This may cause '
                'I/O and CPU interference. Consider waiting for it to finish.'
            )
    except Exception:
        pass

    # Check swap usage
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'vm.swapusage'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            # Output like: "total = 2048.00M  used = 123.45M  free = 1924.55M  ..."
            # Parse used value
            parts = result.stdout.split()
            for i, token in enumerate(parts):
                if token == 'used' and i + 2 < len(parts):
                    used_str = parts[i + 2].rstrip('M').rstrip('G').rstrip('K')
                    try:
                        used_val = float(used_str)
                        if used_val > 0:
                            interference_warnings.append(
                                'Swap is in use (%s). Memory pressure may '
                                'cause I/O interference and unpredictable '
                                'latency. Consider closing applications to '
                                'free memory.' % result.stdout.strip()
                            )
                    except ValueError:
                        pass
                    break
    except Exception:
        pass

    return interference_warnings


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------


def detect_system_info() -> Dict[str, Any]:
    """Detect and return system information for benchmark metadata.

    Captures chip model, core topology, memory, OS version, Rust toolchain,
    and ripgrep build information.
    """
    info: Dict[str, Any] = {
        'platform': sys.platform,
        'arch': platform.machine(),
        'is_arm': is_arm(),
        'is_apple_silicon': is_apple_silicon(),
        'python_version': platform.python_version(),
        'timestamp': datetime.datetime.now().isoformat(),
    }

    if is_macos():
        # Chip model
        brand = sysctl_value('machdep.cpu.brand_string')
        if brand:
            info['chip'] = brand
        else:
            hw_model = sysctl_value('hw.model')
            if hw_model:
                info['chip'] = hw_model

        # macOS version
        info['os_version'] = platform.mac_ver()[0]

        # Memory
        memsize = sysctl_value('hw.memsize')
        if memsize:
            info['memory_gb'] = int(memsize) / (1024 ** 3)

        # Core topology
        nperflevels = sysctl_value('hw.nperflevels')
        if nperflevels:
            info['perf_levels'] = int(nperflevels)
            cores = {}
            for level in range(int(nperflevels)):
                logical = sysctl_value(
                    'hw.perflevel%d.logicalcpu' % level
                )
                physical = sysctl_value(
                    'hw.perflevel%d.physicalcpu' % level
                )
                name = sysctl_value('hw.perflevel%d.name' % level)
                cores['level%d' % level] = {
                    'name': name or ('level%d' % level),
                    'logical': int(logical) if logical else None,
                    'physical': int(physical) if physical else None,
                }
            info['core_topology'] = cores
        else:
            # Fallback: total CPU count
            logical = sysctl_value('hw.logicalcpu')
            physical = sysctl_value('hw.physicalcpu')
            info['core_topology'] = {
                'logical': int(logical) if logical else None,
                'physical': int(physical) if physical else None,
            }

    # Total logical CPUs
    total_logical = sysctl_value('hw.logicalcpu')
    if total_logical:
        info['total_logical_cpus'] = int(total_logical)
    else:
        info['total_logical_cpus'] = os.cpu_count()

    # Rust toolchain
    try:
        result = subprocess.run(
            ['rustc', '--version'], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['rust_version'] = result.stdout.strip()
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Thermal monitoring
# ---------------------------------------------------------------------------


class ThermalMonitor:
    """Monitor thermal state on macOS via sysctl or powermetrics.

    When sudo is available, uses powermetrics for detailed thermal data.
    Falls back to sysctl thermal readings otherwise.
    """

    def __init__(self, use_sudo: bool = False) -> None:
        self.use_sudo = use_sudo
        self.has_powermetrics = use_sudo and shutil.which('powermetrics') is not None
        self._baseline_thermal: Optional[str] = None
        self._baseline_cpu_freq: Optional[Dict[str, int]] = None

    def get_thermal_state(self) -> str:
        """Get current thermal pressure state.

        Tries multiple sysctl keys in order of specificity.
        Returns one of: 'nominal', 'fair', 'serious', 'critical', 'unknown'.
        """
        # Try NSProcessInfo-style thermal state
        state = sysctl_value('kern.thermalstate')
        if state is not None:
            # kern.thermalstate: 0=nominal, 1=fair, 2=serious, 3=critical
            mapping = {
                '0': 'nominal', '1': 'fair',
                '2': 'serious', '3': 'critical',
            }
            return mapping.get(state, 'unknown(%s)' % state)

        # Try xcpm thermal level (Intel/Rosetta path, may exist on some configs)
        level = sysctl_value('machdep.xcpm.cpu_thermal_level')
        if level is not None:
            try:
                lv = int(level)
                if lv == 0:
                    return 'nominal'
                elif lv <= 50:
                    return 'fair'
                elif lv <= 80:
                    return 'serious'
                else:
                    return 'critical'
            except ValueError:
                pass

        # Try powermetrics if sudo available (one-shot sample)
        if self.has_powermetrics:
            try:
                result = subprocess.run(
                    ['sudo', '-n', 'powermetrics',
                     '--samplers', 'smc', '-n', '1', '-i', '100'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if 'CPU Thermal level' in line:
                            # e.g., "CPU Thermal level: 0"
                            try:
                                val = int(line.split(':')[-1].strip())
                                if val == 0:
                                    return 'nominal'
                                return 'elevated(%d)' % val
                            except ValueError:
                                pass
            except Exception:
                pass

        return 'unknown'

    def get_cpu_frequency(self) -> Optional[Dict[str, int]]:
        """Get current CPU cluster frequencies using powermetrics.

        Requires sudo and powermetrics. Parses the cpu_power sampler
        output to extract per-cluster frequencies in MHz.

        Returns a dict like {"P-cluster": 3504, "E-cluster": 2424},
        or None if frequencies cannot be determined.
        """
        if not self.has_powermetrics:
            return None
        try:
            result = subprocess.run(
                ['sudo', '-n', 'powermetrics',
                 '--samplers', 'cpu_power', '-n', '1', '-i', '100'],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return None
            freqs: Dict[str, int] = {}
            for line in result.stdout.splitlines():
                # Look for lines like:
                #   "P-Cluster HW active frequency: 3504 MHz"
                #   "E-Cluster HW active frequency: 2424 MHz"
                # or: "Cluster 0 (P) HW active frequency: 3504 MHz"
                lower = line.lower()
                if 'hw active frequency' in lower or 'active frequency' in lower:
                    # Try to extract cluster name and frequency
                    parts = line.split(':')
                    if len(parts) >= 2:
                        cluster_part = parts[0].strip()
                        freq_part = parts[1].strip()
                        # Extract MHz value
                        freq_tokens = freq_part.split()
                        if freq_tokens:
                            try:
                                freq_mhz = int(freq_tokens[0])
                            except ValueError:
                                continue
                            # Normalize cluster name
                            if 'p-cluster' in cluster_part.lower() or '(p)' in cluster_part.lower():
                                freqs['P-cluster'] = freq_mhz
                            elif 'e-cluster' in cluster_part.lower() or '(e)' in cluster_part.lower():
                                freqs['E-cluster'] = freq_mhz
                            else:
                                freqs[cluster_part] = freq_mhz
            return freqs if freqs else None
        except Exception:
            return None

    def check_dvfs_throttling(self) -> Tuple[bool, Optional[Dict[str, int]]]:
        """Check if DVFS throttling is occurring by comparing to baseline.

        Compares current CPU cluster frequencies to the baseline captured
        at script start. Flags if P-cluster frequency drops below 90% of
        baseline.

        Returns (is_dvfs_throttled, current_freqs).
        """
        current = self.get_cpu_frequency()
        if current is None or self._baseline_cpu_freq is None:
            return False, current
        # Check P-cluster specifically
        baseline_p = self._baseline_cpu_freq.get('P-cluster')
        current_p = current.get('P-cluster')
        if baseline_p is not None and current_p is not None and baseline_p > 0:
            if current_p < baseline_p * 0.90:
                return True, current
        return False, current

    def capture_baseline(self) -> str:
        """Capture baseline thermal state and CPU frequency before benchmarks begin."""
        self._baseline_thermal = self.get_thermal_state()
        self._baseline_cpu_freq = self.get_cpu_frequency()
        if self._baseline_cpu_freq:
            eprint('  [thermal] Baseline CPU frequencies: %s' % (
                ', '.join('%s=%dMHz' % (k, v)
                          for k, v in sorted(self._baseline_cpu_freq.items()))
            ))
        return self._baseline_thermal

    def check_throttling(self) -> Tuple[bool, str]:
        """Check if thermal throttling is occurring.

        Returns (is_throttled, state_description).
        Checks both kern.thermalstate and DVFS frequency drop.
        """
        state = self.get_thermal_state()
        # Consider anything worse than 'nominal' as potential throttling
        is_throttled = state not in ('nominal', 'unknown')
        # Also check DVFS
        dvfs_throttled, _ = self.check_dvfs_throttling()
        if dvfs_throttled and not is_throttled:
            state = state + '+dvfs_throttled'
            is_throttled = True
        return is_throttled, state

    def cooldown(self, seconds: int = COOLDOWN_MIN_SECONDS) -> None:
        """Wait for system to cool down between benchmark groups.

        Waits at least `seconds`, polling thermal state and CPU frequency.
        When sudo is available, polls actual CPU frequency during cooldown
        and waits until frequency recovers to >= 95% of baseline (up to
        3x the cooldown period).
        """
        eprint('  [thermal] Cooldown: waiting %ds...' % seconds)
        time.sleep(seconds)

        # If thermal state is elevated or frequency is low, wait longer (up to 3x)
        max_extra = seconds * 2
        waited = 0
        while waited < max_extra:
            is_throttled, state = self.check_throttling()
            # Also check if CPU frequency has recovered
            freq_recovered = True
            if self.has_powermetrics and self._baseline_cpu_freq:
                current_freq = self.get_cpu_frequency()
                if current_freq:
                    baseline_p = self._baseline_cpu_freq.get('P-cluster')
                    current_p = current_freq.get('P-cluster')
                    if baseline_p and current_p and current_p < baseline_p * 0.95:
                        freq_recovered = False
                        eprint(
                            '  [thermal] CPU frequency not recovered: '
                            'P-cluster=%dMHz (baseline=%dMHz, target=%.0fMHz)'
                            % (current_p, baseline_p, baseline_p * 0.95)
                        )
            if not is_throttled and freq_recovered:
                break
            eprint('  [thermal] Elevated (%s), waiting 5s more...' % state)
            time.sleep(5)
            waited += 5


# ---------------------------------------------------------------------------
# Page cache control
# ---------------------------------------------------------------------------


def purge_cache(use_sudo: bool = False) -> bool:
    """Attempt to purge the filesystem page cache (requires sudo on macOS).

    Returns True if purge succeeded.
    """
    if not is_macos():
        eprint('  [cache] purge only available on macOS, skipping')
        return False
    if not use_sudo:
        eprint('  [cache] purge requires sudo, skipping (warm cache mode)')
        return False
    try:
        result = subprocess.run(
            ['sudo', '-n', 'purge'],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            eprint('  [cache] purge failed (exit code %d)' % result.returncode)
            return False
        return True
    except Exception:
        eprint('  [cache] purge failed')
        return False


# ---------------------------------------------------------------------------
# Corpus management (reuses paths from benchsuite/benchsuite)
# ---------------------------------------------------------------------------


def has_linux(suite_dir: str) -> bool:
    """Check if Linux kernel corpus is available."""
    checkout_dir = path.join(suite_dir, LINUX_DIR)
    # Check for Makefile rather than vmlinux -- kernel build is not
    # required for search benchmarks, only the source tree.
    return path.exists(path.join(checkout_dir, 'Makefile'))


def has_subtitles_en(suite_dir: str) -> bool:
    """Check if English subtitles corpus is available."""
    subtitle_dir = path.join(suite_dir, SUBTITLES_DIR)
    return path.exists(path.join(subtitle_dir, SUBTITLES_EN_NAME_SAMPLE))


def has_subtitles_ru(suite_dir: str) -> bool:
    """Check if Russian subtitles corpus is available."""
    subtitle_dir = path.join(suite_dir, SUBTITLES_DIR)
    return path.exists(path.join(subtitle_dir, SUBTITLES_RU_NAME))


def require_corpus(suite_dir: str, name: str) -> None:
    """Raise an error if a required corpus is not available."""
    checks = {
        'linux': has_linux,
        'subtitles-en': has_subtitles_en,
        'subtitles-ru': has_subtitles_ru,
    }
    if name not in checks:
        raise ValueError('Unknown corpus: %s' % name)
    if not checks[name](suite_dir):
        raise RuntimeError(
            'Corpus "%s" not found in %s. '
            'Run: benchsuite/benchsuite --dir %s --download %s'
            % (name, suite_dir, suite_dir, name)
        )


def linux_dir(suite_dir: str) -> str:
    """Return path to Linux kernel source directory."""
    return path.join(suite_dir, LINUX_DIR)


def subtitles_en_path(suite_dir: str) -> str:
    """Return path to English subtitles sample file."""
    return path.join(suite_dir, SUBTITLES_DIR, SUBTITLES_EN_NAME_SAMPLE)


def subtitles_en_full_path(suite_dir: str) -> str:
    """Return path to full English subtitles file."""
    return path.join(suite_dir, SUBTITLES_DIR, SUBTITLES_EN_NAME)


# ---------------------------------------------------------------------------
# Build management
# ---------------------------------------------------------------------------


def find_rg_binary(project_root: str) -> Optional[str]:
    """Find the ripgrep binary built with release-lto profile."""
    # release-lto profile outputs to target/release-lto/
    lto_path = path.join(project_root, 'target', 'release-lto', 'rg')
    if path.exists(lto_path):
        return lto_path
    # Fallback: release profile
    release_path = path.join(project_root, 'target', 'release', 'rg')
    if path.exists(release_path):
        return release_path
    return None


def build_rg(project_root: str) -> Tuple[str, str, List[str]]:
    """Build ripgrep with release-lto profile.

    Returns (binary_path, profile_used, warnings).
    """
    warnings: List[str] = []

    # Try release-lto first
    eprint('Building ripgrep with release-lto profile...')
    try:
        run_cmd(
            ['cargo', 'build', '--profile', 'release-lto'],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        binary = path.join(project_root, 'target', 'release-lto', 'rg')
        if path.exists(binary):
            return binary, 'release-lto', warnings
    except subprocess.CalledProcessError:
        eprint('release-lto build failed, falling back to --release')
        warnings.append(
            'release-lto profile unavailable; fell back to --release. '
            'Results may differ from LTO-optimized builds.'
        )

    # Fallback to release
    run_cmd(
        ['cargo', 'build', '--release'],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    binary = path.join(project_root, 'target', 'release', 'rg')
    return binary, 'release', warnings


def get_rg_version(rg_bin: str) -> str:
    """Get the version string from the ripgrep binary."""
    try:
        result = subprocess.run(
            [rg_bin, '--version'], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def detect_binary_profile(rg_bin: str) -> str:
    """Best-effort detection of the Cargo profile used for a binary path."""
    if 'release-lto' in rg_bin:
        return 'release-lto'
    if 'release' in rg_bin:
        return 'release'
    return 'custom'


# ---------------------------------------------------------------------------
# Timing and execution
# ---------------------------------------------------------------------------


class TimingResult:
    """Result of a single timed command execution."""

    def __init__(
        self,
        wall_time: float,
        user_time: float,
        sys_time: float,
        line_count: Optional[int],
        returncode: int,
        cpu_freq_mhz: Optional[Dict[str, int]] = None,
    ) -> None:
        self.wall_time = wall_time
        self.user_time = user_time
        self.sys_time = sys_time
        self.line_count = line_count
        self.returncode = returncode
        self.cpu_freq_mhz = cpu_freq_mhz


def run_timed(
    cmd: List[str],
    cwd: Optional[str] = None,
    count_lines: bool = False,
) -> TimingResult:
    """Run a command and measure wall, user, and sys time.

    Uses resource.getrusage to capture user+sys time of child processes.
    Note: RUSAGE_CHILDREN delta approach is correct only because execution
    is single-threaded — no other children are waited for between the
    before/after snapshots.

    When count_lines is False (default), stdout is sent to /dev/null to
    avoid pipe I/O overhead inflating wall time measurements.
    """
    kwargs: Dict[str, Any] = {
        'stderr': subprocess.DEVNULL,
    }
    if cwd:
        kwargs['cwd'] = cwd
    if count_lines:
        kwargs['stdout'] = subprocess.PIPE
    else:
        kwargs['stdout'] = subprocess.DEVNULL

    # Capture child resource usage before
    if resource is not None:
        ru_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    else:
        ru_before = None

    start = time.monotonic()
    completed = subprocess.run(cmd, **kwargs)
    end = time.monotonic()

    wall_time = end - start
    if resource is not None and ru_before is not None:
        ru_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        user_time = ru_after.ru_utime - ru_before.ru_utime
        sys_time = ru_after.ru_stime - ru_before.ru_stime
    else:
        user_time = 0.0
        sys_time = 0.0

    line_count = None
    if count_lines and completed.stdout is not None:
        line_count = completed.stdout.count(b'\n')

    return TimingResult(
        wall_time=wall_time,
        user_time=user_time,
        sys_time=sys_time,
        line_count=line_count,
        returncode=completed.returncode,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(samples: List[float]) -> Dict[str, float]:
    """Compute median, p5, p95, mean, stdev, CV, MAD, and cv_mad for a list of samples.

    Returns a dict with keys: median, p5, p95, mean, stdev, cv, mad, cv_mad, n.
    MAD (median absolute deviation) and cv_mad (MAD/median) are robust
    alternatives to stdev and CV.
    """
    if not samples:
        return {
            'median': 0.0, 'p5': 0.0, 'p95': 0.0,
            'mean': 0.0, 'stdev': 0.0, 'cv': 0.0,
            'mad': 0.0, 'cv_mad': 0.0, 'n': 0,
        }
    n = len(samples)
    sorted_s = sorted(samples)
    median = statistics.median(sorted_s)
    mean = statistics.mean(sorted_s)
    stdev = statistics.stdev(sorted_s) if n > 1 else 0.0
    cv = stdev / mean if mean > 0 else 0.0

    # MAD: median absolute deviation
    abs_devs = sorted(abs(x - median) for x in sorted_s)
    mad = statistics.median(abs_devs)
    cv_mad = mad / median if median > 0 else 0.0

    # Percentiles using nearest-rank method
    p5_idx = max(0, int(math.ceil(0.05 * n)) - 1)
    p95_idx = min(n - 1, int(math.ceil(0.95 * n)) - 1)
    p5 = sorted_s[p5_idx]
    p95 = sorted_s[p95_idx]

    return {
        'median': median,
        'p5': p5,
        'p95': p95,
        'mean': mean,
        'stdev': stdev,
        'cv': cv,
        'mad': mad,
        'cv_mad': cv_mad,
        'n': n,
    }


def _quartile(sorted_data: List[float], q: float) -> float:
    """Compute quartile using linear interpolation.

    Args:
        sorted_data: Sorted list of values.
        q: Quantile to compute (e.g. 0.25 for Q1, 0.75 for Q3).

    Returns:
        Interpolated value at the given quantile.
    """
    n = len(sorted_data)
    pos = q * (n - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_data[low]
    frac = pos - low
    return sorted_data[low] * (1 - frac) + sorted_data[high] * frac


def detect_outliers(samples: List[float]) -> List[int]:
    """Detect outlier indices using the 3*IQR rule.

    Returns a list of indices into `samples` that are suspected outliers.
    Requires at least 4 samples to compute IQR meaningfully.
    Uses linear interpolation for quartile computation.
    """
    if len(samples) < 4:
        return []
    sorted_s = sorted(samples)
    q1 = _quartile(sorted_s, 0.25)
    q3 = _quartile(sorted_s, 0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    return [i for i, v in enumerate(samples) if v < lower or v > upper]


def _exact_u_p_value(u_stat: float, n1: int, n2: int) -> float:
    """Compute exact two-tailed p-value for Mann-Whitney U via enumeration.

    Uses dynamic programming to count the number of rank-sum arrangements
    that yield a U statistic <= u_stat. Valid and efficient for small n
    (n1 * n2 <= 400, i.e. up to n1=n2=20).
    """
    # Count arrangements where U <= u_stat (one tail)
    # U = R1 - n1*(n1+1)/2, so we need rank sums R1 such that
    # U1 <= u_stat or U1 >= n1*n2 - u_stat (two-tailed).
    #
    # We use the recurrence: count(u, n1, n2) = number of ways to get
    # U <= u with sample sizes n1 and n2.
    # Total arrangements = C(n1+n2, n1).

    # Build table: ways[u] = number of ways to achieve exactly U = u
    # Using the recursive formula for the U distribution.
    # DP: f[i][j] = number of ways to choose i items from {1..i+j}
    # such that U = rank_sum - i*(i+1)/2 equals some value.
    # Simpler: use the known recurrence for exact U distribution.
    # ways(u, n1, n2) = ways(u, n1-1, n2) + ways(u - n2, n1, n2-1)
    # with ways(0, 0, n2) = 1, ways(u, 0, n2) = 1 if u==0 else 0

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def count_le(u: int, a: int, b: int) -> int:
        """Count arrangements with U <= u for sample sizes a, b."""
        if a == 0:
            return 1 if u >= 0 else 0
        if b == 0:
            return 1 if u >= 0 else 0
        if u < 0:
            return 0
        # The a-th item from group 1 can have rank a+b (contributing b to U)
        # or not (contributing less).
        return count_le(u - b, a - 1, b) + count_le(u, a, b - 1)

    total = math.comb(n1 + n2, n1)
    u_int = int(round(u_stat))

    # Two-tailed: P(U <= u_stat) for both tails
    left_count = count_le(u_int, n1, n2)
    p_value = 2.0 * left_count / total
    return min(p_value, 1.0)


# Maximum product n1*n2 for which we use the exact test
_EXACT_U_MAX_PRODUCT = 400  # covers up to n1=n2=20


def mann_whitney_u(
    xs: List[float], ys: List[float]
) -> Tuple[float, float]:
    """Compute Mann-Whitney U statistic and p-value.

    Uses exact permutation distribution for small samples (n1*n2 <= 400,
    i.e. up to n1=n2=20). Falls back to normal approximation with
    continuity correction for larger samples.
    Returns (U_statistic, p_value).
    """
    n1 = len(xs)
    n2 = len(ys)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combine and rank
    combined = [(v, 'x') for v in xs] + [(v, 'y') for v in ys]
    combined.sort(key=lambda t: t[0])

    # Assign ranks with tie handling (average rank for ties)
    ranks: List[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum ranks for group x
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 'x')

    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    # Use exact test for small samples, normal approximation for large
    if n1 * n2 <= _EXACT_U_MAX_PRODUCT:
        p_value = _exact_u_p_value(u_stat, n1, n2)
        return u_stat, p_value

    # Normal approximation with continuity correction for large samples
    mu = n1 * n2 / 2.0
    N = n1 + n2
    # Count tie groups from the combined sorted values
    combined_values = sorted([v for v, _ in combined])
    tie_counts = Counter(combined_values)
    tie_correction = sum(t ** 3 - t for t in tie_counts.values())
    sigma = math.sqrt(
        (n1 * n2 / 12.0) * ((N + 1) - tie_correction / (N * (N - 1)))
    ) if N > 1 else 0.0
    if sigma == 0:
        return u_stat, 1.0

    # Apply continuity correction
    z = (abs(u_stat - mu) - 0.5) / sigma
    # Approximate two-tailed p-value using error function
    p_value = 2.0 * (1.0 - _normal_cdf(z))
    return u_stat, p_value


def _normal_cdf(z: float) -> float:
    """Approximate the standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compare_sample_sets(
    reference_times: List[float],
    candidate_times: List[float],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Compare two sample sets using medians and Mann-Whitney U.

    The candidate is interpreted relative to the reference:
    pct_change = (candidate_median - reference_median) / reference_median.
    Positive percentages therefore mean the candidate is slower.
    """
    warnings: List[str] = []
    if len(reference_times) < 5 or len(candidate_times) < 5:
        warnings.append(
            'need >= 5 samples (reference=%d, candidate=%d)'
            % (len(reference_times), len(candidate_times))
        )
        return None, warnings

    if len(reference_times) < 8 or len(candidate_times) < 8:
        warnings.append(
            'normal approximation may be inaccurate with n < 8 '
            '(reference=%d, candidate=%d)'
            % (len(reference_times), len(candidate_times))
        )

    reference_median = statistics.median(reference_times)
    candidate_median = statistics.median(candidate_times)
    if reference_median == 0:
        warnings.append('reference median is 0; skipping percent-change comparison')
        return None, warnings

    pct_change = (candidate_median - reference_median) / reference_median
    _, p_value = mann_whitney_u(reference_times, candidate_times)
    is_significant = p_value < REGRESSION_P_VALUE
    is_regression = pct_change > REGRESSION_THRESHOLD and is_significant
    is_improvement = pct_change < -REGRESSION_THRESHOLD and is_significant

    n1 = len(reference_times)
    n2 = len(candidate_times)
    reference_mean = statistics.mean(reference_times)
    candidate_mean = statistics.mean(candidate_times)
    reference_sd = statistics.stdev(reference_times) if n1 > 1 else 0.0
    candidate_sd = statistics.stdev(candidate_times) if n2 > 1 else 0.0
    pooled_sd = math.sqrt(
        ((n1 - 1) * reference_sd ** 2 + (n2 - 1) * candidate_sd ** 2)
        / max(n1 + n2 - 2, 1)
    ) if (reference_sd > 0 or candidate_sd > 0) else 0.0
    if pooled_sd > 0:
        effect_size = abs(candidate_mean - reference_mean) / pooled_sd
        harmonic_n = 2.0 * n1 * n2 / (n1 + n2)
        z_alpha = 1.96  # two-tailed alpha = 0.05
        noncentrality = effect_size * math.sqrt(harmonic_n / 2.0)
        power = _normal_cdf(noncentrality - z_alpha)
    else:
        effect_size = 0.0
        power = 0.0

    return {
        'reference_median': reference_median,
        'candidate_median': candidate_median,
        'pct_change': pct_change,
        'p_value': p_value,
        'is_significant': is_significant,
        'is_regression': is_regression,
        'is_improvement': is_improvement,
        'effect_size': effect_size,
        'power': power,
        'n_reference': n1,
        'n_candidate': n2,
    }, warnings


# ---------------------------------------------------------------------------
# Benchmark scenario definitions
# ---------------------------------------------------------------------------


class BenchmarkConfig:
    """Configuration for a single benchmark variant (one command to measure)."""

    def __init__(
        self,
        name: str,
        cmd: List[str],
        cwd: Optional[str] = None,
        samples: int = DEFAULT_SAMPLES,
        warmups: int = DEFAULT_WARMUPS,
        cv_threshold: Optional[float] = None,
        logical_name: Optional[str] = None,
        variant_label: Optional[str] = None,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.samples = samples
        self.warmups = warmups
        self.cv_threshold = cv_threshold  # None means use group/global default
        self.logical_name = name if logical_name is None else logical_name
        self.variant_label = variant_label


class BenchmarkGroup:
    """A group of benchmark variants to compare (A/B or multi-way)."""

    def __init__(
        self,
        name: str,
        description: str,
        configs: List[BenchmarkConfig],
        interleaved: bool = True,
        cv_threshold: float = CV_THRESHOLD,
    ) -> None:
        self.name = name
        self.description = description
        self.configs = configs
        self.interleaved = interleaved
        self.cv_threshold = cv_threshold


class BenchmarkResult:
    """Results for one benchmark config across all samples."""

    def __init__(self, config: BenchmarkConfig, cv_threshold: float = CV_THRESHOLD) -> None:
        self.config = config
        self.cv_threshold = cv_threshold
        self.wall_times: List[float] = []
        self.user_times: List[float] = []
        self.sys_times: List[float] = []
        self.line_counts: List[Optional[int]] = []
        self.thermal_states: List[str] = []
        self.cpu_frequencies: List[Optional[Dict[str, int]]] = []
        self._cached_stats: Optional[Dict[str, float]] = None
        self._cached_stats_len: int = 0

    @property
    def stats(self) -> Dict[str, float]:
        """Compute statistics on wall_times (cached; invalidated when new samples are added)."""
        if self._cached_stats is None or self._cached_stats_len != len(self.wall_times):
            self._cached_stats = compute_stats(self.wall_times)
            self._cached_stats_len = len(self.wall_times)
        return self._cached_stats

    @property
    def is_unreliable(self) -> bool:
        """Return True if CV exceeds the applicable threshold.

        Uses cv (stdev/mean) per CLAUDE.md: 'Flag results with CV > 10%
        as unreliable.' cv_mad is still available in stats for informational
        purposes.
        """
        return self.stats['cv'] > self.cv_threshold

    @property
    def throttled_sample_indices(self) -> List[int]:
        """Return indices where thermal_states indicate throttling.

        Throttling is anything not 'nominal' and not 'unknown'.
        """
        return [
            i for i, state in enumerate(self.thermal_states)
            if state not in ('nominal', 'unknown')
        ]

    @property
    def thermal_tainted_count(self) -> int:
        """Return the number of samples collected during thermal throttling."""
        return len(self.throttled_sample_indices)

    @property
    def outlier_indices(self) -> List[int]:
        """Return indices of suspected outlier samples (3*IQR rule)."""
        return detect_outliers(self.wall_times)

    @property
    def outlier_count(self) -> int:
        """Return the number of suspected outliers."""
        return len(self.outlier_indices)


class GroupResult:
    """Results for an entire benchmark group."""

    def __init__(self, group: BenchmarkGroup) -> None:
        self.group = group
        self.results: Dict[str, BenchmarkResult] = {}
        for cfg in group.configs:
            # Per-config threshold overrides group threshold
            threshold = cfg.cv_threshold if cfg.cv_threshold is not None else group.cv_threshold
            self.results[cfg.name] = BenchmarkResult(cfg, cv_threshold=threshold)

    @property
    def has_unreliable(self) -> bool:
        """Return True if any config result is unreliable."""
        return any(r.is_unreliable for r in self.results.values())


class SuiteResult:
    """Results for one cache-mode suite."""

    def __init__(
        self,
        cache_mode: str,
        group_results: List[GroupResult],
        comparisons: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.cache_mode = cache_mode
        self.group_results = group_results
        self.comparisons = [] if comparisons is None else comparisons


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs benchmark groups with thermal management and interleaved execution."""

    def __init__(
        self,
        rg_bin: str,
        suite_dir: str,
        thermal: ThermalMonitor,
        cache_mode: str = 'warm',
        use_sudo: bool = False,
        convergence_threshold: float = 0.05,
    ) -> None:
        self.rg_bin = rg_bin
        self.suite_dir = suite_dir
        self.thermal = thermal
        self.cache_mode = cache_mode
        self.use_sudo = use_sudo
        self.convergence_threshold = convergence_threshold

    @staticmethod
    def _thermal_severity(state: str) -> int:
        """Return numeric severity for a thermal state string."""
        thermal_order = ['nominal', 'unknown', 'fair', 'serious', 'critical']
        for i, s in enumerate(thermal_order):
            if state.startswith(s):
                return i
        # Anything with 'elevated' or unrecognized is between fair and serious
        return 2

    def _check_convergence_3sample(
        self, times: List[float], threshold: float,
    ) -> bool:
        """Check 3-sample convergence: all pairwise differences within threshold."""
        if len(times) < 3:
            return False
        last3 = times[-3:]
        for i in range(3):
            for j in range(i + 1, 3):
                slower = max(last3[i], last3[j])
                faster = min(last3[i], last3[j])
                if faster > 0 and (slower - faster) / faster > threshold:
                    return False
        return True

    def _run_warmups(
        self, group: BenchmarkGroup, result: GroupResult,
        warmup_count: int, max_extra_warmups: int = 4,
        convergence_threshold: float = 0.05,
    ) -> None:
        """Run warmup iterations with steady-state convergence detection.

        Runs at least `warmup_count` warmup rounds. After that, checks
        whether the last three warmup times per config have converged
        (all pairwise differences within `convergence_threshold` relative
        to the faster one). If any config hasn't converged, runs up to
        `max_extra_warmups` additional rounds.

        Convergence checking is skipped in cold cache mode (convergence
        is meaningless when the cache is purged each time).
        """
        eprint('  Warmups: %d per config (+ up to %d for convergence)...'
               % (warmup_count, max_extra_warmups))

        # Track last 3 warmup times per config for convergence
        last_times: Dict[str, List[float]] = {
            cfg.name: [] for cfg in group.configs
        }

        for w in range(warmup_count):
            for cfg in group.configs:
                if self.cache_mode == 'cold':
                    purge_cache(self.use_sudo)
                timing = run_timed(cfg.cmd, cwd=cfg.cwd, count_lines=False)
                # Track last 3 times
                last_times[cfg.name].append(timing.wall_time)
                if len(last_times[cfg.name]) > 3:
                    last_times[cfg.name] = last_times[cfg.name][-3:]

        # Skip convergence checking in cold cache mode
        if self.cache_mode == 'cold':
            eprint('  [warmup] Skipping convergence check (cold cache mode)')
            return

        # Convergence check: run extra warmups if last 3 times haven't converged
        extra = 0
        while extra < max_extra_warmups:
            all_converged = True
            for cfg in group.configs:
                if not self._check_convergence_3sample(
                    last_times[cfg.name], convergence_threshold
                ):
                    all_converged = False
                    break
            if all_converged:
                break
            eprint('  [warmup] Not converged, running extra warmup %d...' % (extra + 1))
            for cfg in group.configs:
                timing = run_timed(cfg.cmd, cwd=cfg.cwd, count_lines=False)
                last_times[cfg.name].append(timing.wall_time)
                if len(last_times[cfg.name]) > 3:
                    last_times[cfg.name] = last_times[cfg.name][-3:]
            extra += 1

        if extra > 0:
            if extra >= max_extra_warmups:
                # Exhausted extra warmups without convergence
                not_converged = []
                for cfg in group.configs:
                    if not self._check_convergence_3sample(
                        last_times[cfg.name], convergence_threshold
                    ):
                        not_converged.append(cfg.name)
                if not_converged:
                    eprint(
                        '  [warmup] WARNING: convergence NOT achieved after '
                        '%d extra warmup(s) for: %s'
                        % (extra, ', '.join(not_converged))
                    )
                else:
                    eprint('  [warmup] Converged after %d extra warmup(s)' % extra)
            else:
                eprint('  [warmup] Converged after %d extra warmup(s)' % extra)

    @staticmethod
    def _balanced_alternation(
        configs: List[BenchmarkConfig], round_idx: int,
    ) -> List[BenchmarkConfig]:
        """Return configs in balanced alternation order for a given round.

        Uses a randomized balanced block design: even rounds use forward
        order, odd rounds use reverse order. Within each order, adjacent
        pairs are randomly swapped to add jitter without the bias risk
        of a full shuffle.
        """
        ordered = list(configs)
        if round_idx % 2 == 1:
            ordered = list(reversed(ordered))
        # Random non-overlapping adjacent-pair swaps for jitter
        for i in range(0, len(ordered) - 1, 2):
            if random.random() < 0.3:
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]
        return ordered

    def run_group(self, group: BenchmarkGroup) -> GroupResult:
        """Run all configs in a benchmark group with interleaved execution.

        Performs warmups (with convergence detection), then collects samples
        using balanced alternation (not pure random shuffle) with stdout
        sent to /dev/null to avoid pipe overhead.
        """
        result = GroupResult(group)

        eprint('\n=== %s ===' % group.name)
        eprint('  %s' % group.description)

        # Determine sample count (use the max across configs)
        sample_count = max(cfg.samples for cfg in group.configs)
        warmup_count = max(cfg.warmups for cfg in group.configs)

        # Correctness pass: capture expected line counts.  This is a pure
        # correctness check — it runs with count_lines=True (stdout piped)
        # before any performance measurements begin.
        eprint('  Capturing expected line counts...')
        for cfg in group.configs:
            verification = run_timed(cfg.cmd, cwd=cfg.cwd, count_lines=True)
            if verification.line_count is not None:
                result.results[cfg.name].line_counts.append(verification.line_count)

        # Warmups with convergence detection
        self._run_warmups(group, result, warmup_count,
                          convergence_threshold=self.convergence_threshold)

        # Samples (balanced alternation, stdout to /dev/null)
        eprint('  Samples: %d per config (balanced alternation)...' % sample_count)
        for s in range(sample_count):
            # Balanced alternation instead of pure random shuffle
            if group.interleaved:
                configs_order = self._balanced_alternation(group.configs, s)
            else:
                configs_order = list(group.configs)

            for cfg in configs_order:
                if self.cache_mode == 'cold':
                    purge_cache(self.use_sudo)

                # Check thermal state before run
                _, thermal_before = self.thermal.check_throttling()

                # Get CPU frequency before run (if available)
                cpu_freq = self.thermal.get_cpu_frequency()

                # count_lines=False: send stdout to /dev/null to avoid
                # pipe I/O overhead inflating wall time measurements
                timing = run_timed(cfg.cmd, cwd=cfg.cwd, count_lines=False)

                # Check thermal state after run; record the worse of the two
                _, thermal_after = self.thermal.check_throttling()
                if self._thermal_severity(thermal_after) >= self._thermal_severity(thermal_before):
                    thermal_state = thermal_after
                else:
                    thermal_state = thermal_before

                br = result.results[cfg.name]
                br.wall_times.append(timing.wall_time)
                br.user_times.append(timing.user_time)
                br.sys_times.append(timing.sys_time)
                br.thermal_states.append(thermal_state)
                br.cpu_frequencies.append(cpu_freq)

            eprint('  [%d/%d] done' % (s + 1, sample_count))

        # Post-measurement correctness verification
        for cfg in group.configs:
            br = result.results[cfg.name]
            if br.line_counts:
                expected_lc = br.line_counts[0]
                if expected_lc is not None:
                    verify = run_timed(cfg.cmd, cwd=cfg.cwd, count_lines=True)
                    if verify.line_count is not None and verify.line_count != expected_lc:
                        eprint(
                            '  WARNING: Post-measurement correctness check failed for %s: '
                            'expected %d lines, got %d'
                            % (cfg.name, expected_lc, verify.line_count)
                        )

        # Report thermal tainting
        for cfg in group.configs:
            br = result.results[cfg.name]
            tainted = br.thermal_tainted_count
            if tainted > 0:
                eprint(
                    '  WARNING: %s had %d/%d samples collected during '
                    'thermal throttling (indices: %s)'
                    % (cfg.name, tainted, len(br.wall_times),
                       ', '.join(str(i) for i in br.throttled_sample_indices))
                )

        # Report quick stats and outliers
        n = len(group.configs[0].cmd)  # just for label width reference
        for cfg in group.configs:
            br = result.results[cfg.name]
            st = br.stats
            sample_n = int(st.get('n', len(br.wall_times)))
            flags = ''
            if br.is_unreliable:
                flags += ' [UNRELIABLE: cv=%.1f%%]' % (st['cv'] * 100)
            if br.outlier_count > 0:
                flags += ' [%d OUTLIER(S)]' % br.outlier_count
            # Use min/max labels when n < 20
            if sample_n < 20:
                eprint(
                    '  %-30s median=%.4fs  min=%.4fs  max=%.4fs  CV=%.1f%%  cv_mad=%.1f%%%s'
                    % (cfg.name, st['median'], st['p5'], st['p95'],
                       st['cv'] * 100, st['cv_mad'] * 100, flags)
                )
            else:
                eprint(
                    '  %-30s median=%.4fs  p5=%.4fs  p95=%.4fs  CV=%.1f%%  cv_mad=%.1f%%%s'
                    % (cfg.name, st['median'], st['p5'], st['p95'],
                       st['cv'] * 100, st['cv_mad'] * 100, flags)
                )

        return result


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def _adaptive_thread_counts(system_info: Optional[Dict[str, Any]] = None) -> List[int]:
    """Generate thread counts based on the system's P-core count.

    Reads P-core count from system_info['core_topology'], generates thread
    counts: powers of 2 up to P-core count, plus P-core count itself, plus
    total logical CPUs. Falls back to [1, 2, 4, 6, 8] if no topology info.
    """
    if system_info is None:
        return [1, 2, 4, 6, 8]

    topo = system_info.get('core_topology')
    if not topo:
        return [1, 2, 4, 6, 8]

    # Find P-core logical count (level0 is typically Performance on Apple Silicon)
    p_cores = None
    for level_key in sorted(topo.keys()):
        entry = topo.get(level_key)
        if isinstance(entry, dict):
            name = entry.get('name', '').lower()
            if 'performance' in name or name.startswith('p'):
                p_cores = entry.get('logical')
                break

    if p_cores is None or p_cores < 1:
        return [1, 2, 4, 6, 8]

    total_logical = system_info.get('total_logical_cpus') or p_cores

    # Powers of 2 up to P-core count
    counts = set()
    counts.add(1)
    power = 2
    while power <= p_cores:
        counts.add(power)
        power *= 2
    counts.add(p_cores)
    counts.add(total_logical)

    return sorted(counts)


def scenario_thread_scaling(
    rg_bin: str, suite_dir: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> BenchmarkGroup:
    """Thread scaling benchmark: compare 1, 2, 4, 6, 8 threads on Linux corpus.

    Uses a literal search pattern across the Linux kernel source.
    10+ samples per CLAUDE.md standards for thread-scaling tests.
    """
    require_corpus(suite_dir, 'linux')
    cwd = linux_dir(suite_dir)
    pat = 'PM_RESUME'
    thread_counts = _adaptive_thread_counts(system_info)

    configs = []
    for j in thread_counts:
        configs.append(BenchmarkConfig(
            name='rg -j%d' % j,
            cmd=[rg_bin, '-n', '-j', str(j), pat],
            cwd=cwd,
            samples=THREAD_SCALING_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ))

    return BenchmarkGroup(
        name='thread_scaling',
        description='Literal search "PM_RESUME" across Linux kernel with varying thread counts',
        configs=configs,
        interleaved=True,
        cv_threshold=CV_THRESHOLD_MULTI_FILE,
    )


def scenario_thread_scaling_output(
    rg_bin: str, suite_dir: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> BenchmarkGroup:
    """Thread scaling benchmark for output-heavy search modes.

    These modes spend more time in sinks and synchronization than simple
    count-style searches, so they are useful when evaluating workload-aware
    thread policies on Apple Silicon.
    """
    require_corpus(suite_dir, 'linux')
    cwd = linux_dir(suite_dir)
    thread_counts = _adaptive_thread_counts(system_info)

    configs = []
    for j in thread_counts:
        configs.append(BenchmarkConfig(
            name='rg --json -j%d' % j,
            cmd=[rg_bin, '--json', '-j', str(j), 'return'],
            cwd=cwd,
            samples=THREAD_SCALING_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ))
        configs.append(BenchmarkConfig(
            name='rg -n -o -j%d' % j,
            cmd=[rg_bin, '-n', '-o', '-j', str(j), 'return'],
            cwd=cwd,
            samples=THREAD_SCALING_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ))

    return BenchmarkGroup(
        name='thread_scaling_output',
        description='Thread scaling for output-heavy modes (--json and -o) on Linux kernel',
        configs=configs,
        interleaved=True,
        cv_threshold=CV_THRESHOLD_MULTI_FILE,
    )


def scenario_mmap_vs_read(
    rg_bin: str, suite_dir: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> BenchmarkGroup:
    """mmap vs read benchmark on subtitles corpus (large file).

    Compares --mmap vs --no-mmap on a large single file.
    Validates the mmap re-enablement on Apple Silicon.
    """
    require_corpus(suite_dir, 'subtitles-en')
    en = subtitles_en_path(suite_dir)
    pat = 'Sherlock Holmes'

    configs = [
        BenchmarkConfig(
            name='rg --mmap',
            cmd=[rg_bin, '-n', '--mmap', pat, en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg --no-mmap',
            cmd=[rg_bin, '-n', '--no-mmap', pat, en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
    ]

    return BenchmarkGroup(
        name='mmap_vs_read',
        description='Compare --mmap vs --no-mmap on subtitles corpus (en.sample.txt)',
        configs=configs,
        interleaved=True,
        cv_threshold=CV_THRESHOLD_SINGLE_FILE,
    )


def scenario_mmap_multiline(
    rg_bin: str, suite_dir: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> BenchmarkGroup:
    """mmap vs no-mmap in multiline mode on subtitles corpus.

    Commit 07ff63c re-enabled mmap specifically for multiline performance.
    This scenario verifies whether --mmap helps in -U (multiline) mode
    by testing both a cross-line regex and a simple literal.
    """
    require_corpus(suite_dir, 'subtitles-en')
    en = subtitles_en_path(suite_dir)

    configs = [
        # Cross-line pattern: benefits from multiline mmap
        BenchmarkConfig(
            name='rg -U --mmap (cross-line)',
            cmd=[rg_bin, '-n', '-U', '--mmap', r'Sherlock\n.*Holmes', en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg -U --no-mmap (cross-line)',
            cmd=[rg_bin, '-n', '-U', '--no-mmap', r'Sherlock\n.*Holmes', en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        # Simple literal in multiline mode for comparison
        BenchmarkConfig(
            name='rg -U --mmap (literal)',
            cmd=[rg_bin, '-n', '-U', '--mmap', 'Sherlock Holmes', en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg -U --no-mmap (literal)',
            cmd=[rg_bin, '-n', '-U', '--no-mmap', 'Sherlock Holmes', en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
    ]

    return BenchmarkGroup(
        name='mmap_multiline',
        description='Compare --mmap vs --no-mmap in multiline (-U) mode on subtitles corpus',
        configs=configs,
        interleaved=True,
        cv_threshold=CV_THRESHOLD_SINGLE_FILE,
    )


def scenario_multiline_vs_singleline(
    rg_bin: str, suite_dir: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> BenchmarkGroup:
    """Multiline vs single-line benchmark on Linux corpus.

    Compares -U (multiline) vs default single-line mode.
    Tests with both a sparse pattern and a denser pattern.
    """
    require_corpus(suite_dir, 'linux')
    cwd = linux_dir(suite_dir)
    # Sparse pattern
    pat = 'PM_RESUME'

    configs = [
        BenchmarkConfig(
            name='rg (single-line, sparse)',
            cmd=[rg_bin, '-n', pat],
            cwd=cwd,
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg -U (multiline, sparse)',
            cmd=[rg_bin, '-n', '-U', pat],
            cwd=cwd,
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg (single-line, dense)',
            cmd=[rg_bin, '-n', 'return'],
            cwd=cwd,
            samples=DEFAULT_SAMPLES,
            warmups=5,  # Dense patterns need extra warmups for steady state
        ),
        BenchmarkConfig(
            name='rg -U (multiline, dense)',
            cmd=[rg_bin, '-n', '-U', 'return'],
            cwd=cwd,
            samples=DEFAULT_SAMPLES,
            warmups=5,  # Dense patterns need extra warmups for steady state
        ),
    ]

    return BenchmarkGroup(
        name='multiline_vs_singleline',
        description='Compare single-line vs multiline (-U) on Linux kernel (sparse + dense patterns)',
        configs=configs,
        interleaved=True,
        cv_threshold=CV_THRESHOLD_MULTI_FILE,
    )


def scenario_large_file_single_threaded(
    rg_bin: str, suite_dir: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> BenchmarkGroup:
    """Large file single-threaded benchmark on subtitles corpus.

    Isolates SIMD/regex engine performance from I/O scheduling by
    searching a single large file with one thread.
    """
    require_corpus(suite_dir, 'subtitles-en')
    en = subtitles_en_path(suite_dir)

    configs = [
        BenchmarkConfig(
            name='rg literal',
            cmd=[rg_bin, '-n', '-j1', 'Sherlock Holmes', en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg regex',
            cmd=[rg_bin, '-n', '-j1',
                 r'\w+\s+Holmes\s+\w+', en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg no-literal regex',
            cmd=[rg_bin, '-n', '-j1',
                 r'\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5}', en],
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
    ]

    return BenchmarkGroup(
        name='large_file_single_threaded',
        description='Single-threaded search on subtitles corpus (literal, regex, no-literal)',
        configs=configs,
        interleaved=True,
        cv_threshold=CV_THRESHOLD_SINGLE_FILE,
    )


def scenario_directory_io(
    rg_bin: str, suite_dir: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> BenchmarkGroup:
    """Directory walk I/O benchmark on Linux kernel source.

    Searches for a rare pattern to stress directory traversal and
    I/O scheduling on APFS. Also includes a common pattern to compare.
    """
    require_corpus(suite_dir, 'linux')
    cwd = linux_dir(suite_dir)

    configs = [
        BenchmarkConfig(
            name='rg rare pattern',
            cmd=[rg_bin, '-c', 'PM_RESUME'],
            cwd=cwd,
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg common pattern',
            cmd=[rg_bin, '-c', 'return'],
            cwd=cwd,
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
        BenchmarkConfig(
            name='rg no-match pattern',
            cmd=[rg_bin, '-c', 'ZZZZNOTFOUNDPATTERNZZZZ'],
            cwd=cwd,
            samples=DEFAULT_SAMPLES,
            warmups=DEFAULT_WARMUPS,
        ),
    ]

    return BenchmarkGroup(
        name='directory_io',
        description='Directory traversal on Linux kernel source (rare, common, no-match patterns)',
        configs=configs,
        interleaved=True,
        cv_threshold=CV_THRESHOLD_MULTI_FILE,
    )


ALL_SCENARIOS = {
    'thread_scaling': scenario_thread_scaling,
    'thread_scaling_output': scenario_thread_scaling_output,
    'mmap_vs_read': scenario_mmap_vs_read,
    'mmap_multiline': scenario_mmap_multiline,
    'multiline_vs_singleline': scenario_multiline_vs_singleline,
    'large_file_single_threaded': scenario_large_file_single_threaded,
    'directory_io': scenario_directory_io,
}


def parse_cache_modes(cache_arg: str) -> List[str]:
    """Expand a CLI cache selector into concrete suite modes."""
    if cache_arg == 'both':
        return ['warm', 'cold']
    return [cache_arg]


def apply_samples_override(
    group: BenchmarkGroup,
    scenario_name: str,
    sample_override: Optional[int],
) -> None:
    """Apply a global sample-count override with scenario-specific floors."""
    if sample_override is None:
        return
    for cfg in group.configs:
        if scenario_name in ('thread_scaling', 'thread_scaling_output'):
            cfg.samples = max(sample_override, THREAD_SCALING_SAMPLES)
        else:
            cfg.samples = sample_override


def build_scenario_group(
    scenario_name: str,
    rg_bin: str,
    suite_dir: str,
    system_info: Dict[str, Any],
    sample_override: Optional[int],
) -> BenchmarkGroup:
    """Build one benchmark group with sample overrides applied."""
    group = ALL_SCENARIOS[scenario_name](rg_bin, suite_dir, system_info)
    apply_samples_override(group, scenario_name, sample_override)
    return group


def labeled_config(cfg: BenchmarkConfig, label: str) -> BenchmarkConfig:
    """Clone a config with an A/B variant label for interleaved comparison."""
    return BenchmarkConfig(
        name='[%s] %s' % (label, cfg.logical_name),
        cmd=list(cfg.cmd),
        cwd=cfg.cwd,
        samples=cfg.samples,
        warmups=cfg.warmups,
        cv_threshold=cfg.cv_threshold,
        logical_name=cfg.logical_name,
        variant_label=label,
    )


def build_ab_group(
    scenario_name: str,
    primary_bin: str,
    compare_bin: str,
    suite_dir: str,
    system_info: Dict[str, Any],
    sample_override: Optional[int],
    primary_label: str,
    compare_label: str,
) -> BenchmarkGroup:
    """Build one interleaved A/B benchmark group for two binaries."""
    primary = build_scenario_group(
        scenario_name, primary_bin, suite_dir, system_info, sample_override
    )
    compare = build_scenario_group(
        scenario_name, compare_bin, suite_dir, system_info, sample_override
    )
    if primary.name != compare.name:
        raise RuntimeError(
            'Scenario %s produced mismatched group names: %s vs %s'
            % (scenario_name, primary.name, compare.name)
        )

    compare_by_name = {
        cfg.logical_name: cfg for cfg in compare.configs
    }
    configs: List[BenchmarkConfig] = []
    missing: List[str] = []
    for primary_cfg in primary.configs:
        compare_cfg = compare_by_name.pop(primary_cfg.logical_name, None)
        if compare_cfg is None:
            missing.append(primary_cfg.logical_name)
            continue
        configs.append(labeled_config(primary_cfg, primary_label))
        configs.append(labeled_config(compare_cfg, compare_label))
    if missing or compare_by_name:
        missing_list = missing + sorted(compare_by_name.keys())
        raise RuntimeError(
            'Scenario %s produced mismatched config sets: %s'
            % (scenario_name, ', '.join(missing_list))
        )

    return BenchmarkGroup(
        name=primary.name,
        description=(
            '%s (interleaved %s vs %s)'
            % (primary.description, primary_label, compare_label)
        ),
        configs=configs,
        interleaved=True,
        cv_threshold=primary.cv_threshold,
    )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def load_baseline(baseline_path: str) -> Dict[str, List[float]]:
    """Load baseline CSV and return a mapping of suite/config -> wall times."""
    results: Dict[str, List[float]] = {}
    with open(baseline_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            benchmark = row.get('benchmark', '')
            name = row.get('name', '')
            cache_mode = row.get('cache_mode', '')
            if cache_mode:
                key = '%s/%s/%s' % (cache_mode, benchmark, name)
            else:
                key = '%s/%s' % (benchmark, name)
            duration = float(row.get('duration', 0))
            if key not in results:
                results[key] = []
            results[key].append(duration)
    return results


def check_regressions(
    suite_results: Sequence[SuiteResult],
    baseline: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    """Compare benchmark results against a prior CSV baseline."""
    eprint(
        'WARNING: Baseline comparison is NOT interleaved A/B. '
        'Results may be confounded by environmental differences '
        '(thermal state, background load, OS updates) between runs.'
    )
    regressions: List[Dict[str, Any]] = []

    for suite in suite_results:
        for gr in suite.group_results:
            for cfg_name, br in gr.results.items():
                full_key = '%s/%s/%s' % (suite.cache_mode, gr.group.name, cfg_name)
                legacy_key = '%s/%s' % (gr.group.name, cfg_name)
                baseline_times = baseline.get(full_key)
                if baseline_times is None:
                    baseline_times = baseline.get(legacy_key)
                if baseline_times is None:
                    continue

                entry, compare_warnings = compare_sample_sets(
                    baseline_times, br.wall_times
                )
                if entry is None:
                    for warning in compare_warnings:
                        eprint(
                            'WARNING: Skipping regression check for %s: %s'
                            % (full_key, warning)
                        )
                    continue
                for warning in compare_warnings:
                    eprint('WARNING: %s: %s' % (full_key, warning))

                regressions.append({
                    'benchmark': full_key,
                    'cache_mode': suite.cache_mode,
                    'baseline_median': entry['reference_median'],
                    'new_median': entry['candidate_median'],
                    'pct_change': entry['pct_change'],
                    'p_value': entry['p_value'],
                    'is_significant': entry['is_significant'],
                    'is_regression': entry['is_regression'],
                    'effect_size': entry['effect_size'],
                    'power': entry['power'],
                    'n_baseline': entry['n_reference'],
                    'n_new': entry['n_candidate'],
                })

    return regressions


def compare_ab_suites(
    suite_results: Sequence[SuiteResult],
    primary_label: str,
    compare_label: str,
) -> List[str]:
    """Populate suite comparisons for interleaved A/B binary runs."""
    warnings: List[str] = []
    for suite in suite_results:
        comparisons: List[Dict[str, Any]] = []
        for gr in suite.group_results:
            pairings: Dict[str, Dict[str, BenchmarkResult]] = {}
            logical_order: List[str] = []
            for cfg in gr.group.configs:
                logical_name = cfg.logical_name
                if logical_name not in pairings:
                    pairings[logical_name] = {}
                    logical_order.append(logical_name)
                pairings[logical_name][cfg.variant_label or ''] = gr.results[cfg.name]
            for logical_name in logical_order:
                pair = pairings[logical_name]
                if primary_label not in pair or compare_label not in pair:
                    warnings.append(
                        'A/B pair missing for %s/%s in %s cache'
                        % (gr.group.name, logical_name, suite.cache_mode)
                    )
                    continue
                reference = pair[compare_label]
                candidate = pair[primary_label]
                entry, compare_warnings = compare_sample_sets(
                    reference.wall_times,
                    candidate.wall_times,
                )
                full_key = '%s/%s/%s' % (
                    suite.cache_mode, gr.group.name, logical_name,
                )
                if entry is None:
                    for warning in compare_warnings:
                        warnings.append(
                            'A/B compare skipped for %s: %s' % (full_key, warning)
                        )
                    continue
                for warning in compare_warnings:
                    warnings.append('A/B compare note for %s: %s' % (full_key, warning))
                comparisons.append({
                    'benchmark': full_key,
                    'group': gr.group.name,
                    'config': logical_name,
                    'cache_mode': suite.cache_mode,
                    'reference_label': compare_label,
                    'candidate_label': primary_label,
                    'reference_name': reference.config.name,
                    'candidate_name': candidate.config.name,
                    'reference_median': entry['reference_median'],
                    'candidate_median': entry['candidate_median'],
                    'pct_change': entry['pct_change'],
                    'p_value': entry['p_value'],
                    'is_significant': entry['is_significant'],
                    'is_regression': entry['is_regression'],
                    'is_improvement': entry['is_improvement'],
                    'effect_size': entry['effect_size'],
                    'power': entry['power'],
                    'n_reference': entry['n_reference'],
                    'n_candidate': entry['n_candidate'],
                })
        suite.comparisons = comparisons
    return warnings


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def write_csv(
    filepath: str,
    suite_results: Sequence[SuiteResult],
    system_info: Dict[str, Any],
) -> None:
    """Write benchmark results to CSV in a format compatible with raw.csv.

    Existing columns: benchmark, warmup_iter, iter, name, command, duration,
    lines, env.
    ARM-specific columns appended: chip, cores_used, thermal_state, cache_mode,
    cv, median, p5, p95, user_time, sys_time, binary_label, logical_name.
    """
    fields = [
        'benchmark', 'warmup_iter', 'iter',
        'name', 'command', 'duration', 'lines', 'env',
        # ARM-specific columns appended after existing columns
        'chip', 'cores_used', 'thermal_state', 'cache_mode',
        'cv', 'median', 'p5', 'p95', 'user_time', 'sys_time',
        'cpu_freq_mhz', 'cv_mad', 'binary_label', 'logical_name',
    ]

    chip = system_info.get('chip', 'unknown')
    cores_used = ''
    if 'core_topology' in system_info:
        topo = system_info['core_topology']
        parts = []
        for level_key in sorted(topo.keys()):
            if isinstance(topo[level_key], dict) and 'name' in topo[level_key]:
                parts.append('%s:%s' % (
                    topo[level_key]['name'],
                    topo[level_key].get('logical', '?'),
                ))
        if parts:
            cores_used = ' '.join(parts)
        elif 'logical' in topo:
            cores_used = str(topo['logical'])

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()

        for suite in suite_results:
            for gr in suite.group_results:
                for cfg_name, br in gr.results.items():
                    cfg = br.config
                    stats = br.stats
                    for i, wt in enumerate(br.wall_times):
                        thermal = br.thermal_states[i] if i < len(br.thermal_states) else ''
                        cpu_freq_str = ''
                        if i < len(br.cpu_frequencies) and br.cpu_frequencies[i]:
                            freq = br.cpu_frequencies[i]
                            cpu_freq_str = ';'.join(
                                '%s=%d' % (k, v) for k, v in sorted(freq.items())
                            )
                        writer.writerow({
                            'benchmark': gr.group.name,
                            'warmup_iter': cfg.warmups,
                            'iter': i,
                            'name': cfg.name,
                            'command': ' '.join(cfg.cmd),
                            'duration': wt,
                            'lines': br.line_counts[0] if br.line_counts else '',
                            'env': '',
                            'chip': chip,
                            'cores_used': cores_used,
                            'thermal_state': thermal,
                            'cache_mode': suite.cache_mode,
                            'cv': '%.4f' % stats['cv'],
                            'median': '%.6f' % stats['median'],
                            'p5': '%.6f' % stats['p5'],
                            'p95': '%.6f' % stats['p95'],
                            'user_time': '%.6f' % br.user_times[i] if i < len(br.user_times) else '',
                            'sys_time': '%.6f' % br.sys_times[i] if i < len(br.sys_times) else '',
                            'cpu_freq_mhz': cpu_freq_str,
                            'cv_mad': '%.4f' % stats['cv_mad'],
                            'binary_label': cfg.variant_label or '',
                            'logical_name': cfg.logical_name,
                        })


def write_json(
    filepath: str,
    suite_results: Sequence[SuiteResult],
    system_info: Dict[str, Any],
    regressions: List[Dict[str, Any]],
    warnings: List[str],
    run_metadata: Dict[str, Any],
) -> None:
    """Write detailed JSON output with all metadata and results."""
    output: Dict[str, Any] = {
        'metadata': {
            'system': system_info,
            **run_metadata,
            'timestamp': datetime.datetime.now().isoformat(),
            'arm_bench_version': '1.1.0',
        },
        'suites': {},
        'regressions': regressions,
        'warnings': warnings,
    }

    all_comparisons: List[Dict[str, Any]] = []
    for suite in suite_results:
        suite_benchmarks: Dict[str, Any] = {}
        for gr in suite.group_results:
            group_data: Dict[str, Any] = {
                'description': gr.group.description,
                'configs': {},
            }
            for cfg_name, br in gr.results.items():
                stats = br.stats
                sample_n = int(stats.get('n', len(br.wall_times)))
                stats_dict: Dict[str, Any] = {
                    'median': stats['median'],
                    'p5': stats['p5'],
                    'p95': stats['p95'],
                    'mean': stats['mean'],
                    'stdev': stats['stdev'],
                    'cv': stats['cv'],
                    'mad': stats['mad'],
                    'cv_mad': stats['cv_mad'],
                    'n': sample_n,
                }
                if sample_n < 20:
                    stats_dict['note'] = (
                        'p5 and p95 equal min and max respectively '
                        'due to sample size n=%d < 20' % sample_n
                    )
                config_data: Dict[str, Any] = {
                    'command': ' '.join(br.config.cmd),
                    'logical_name': br.config.logical_name,
                    'variant_label': br.config.variant_label,
                    'wall_times': br.wall_times,
                    'user_times': br.user_times,
                    'sys_times': br.sys_times,
                    'line_counts': br.line_counts,
                    'thermal_states': br.thermal_states,
                    'cpu_frequencies': br.cpu_frequencies,
                    'stats': stats_dict,
                    'unreliable': br.is_unreliable,
                    'cv_threshold': br.cv_threshold,
                    'outlier_indices': br.outlier_indices,
                    'outlier_count': br.outlier_count,
                }
                group_data['configs'][cfg_name] = config_data
            suite_benchmarks[gr.group.name] = group_data
        output['suites'][suite.cache_mode] = {
            'benchmarks': suite_benchmarks,
            'comparisons': suite.comparisons,
        }
        all_comparisons.extend(suite.comparisons)

    if len(suite_results) == 1:
        only_suite = next(iter(suite_results))
        output['benchmarks'] = output['suites'][only_suite.cache_mode]['benchmarks']
        output['comparisons'] = only_suite.comparisons
    elif all_comparisons:
        output['comparisons'] = all_comparisons

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def format_summary(
    suite_results: Sequence[SuiteResult],
    system_info: Dict[str, Any],
    regressions: List[Dict[str, Any]],
    warnings: List[str],
    run_metadata: Dict[str, Any],
) -> str:
    """Format a human-readable summary of benchmark results."""
    lines: List[str] = []

    # Header
    lines.append('ARM Benchmark Results')
    lines.append('=' * 60)
    lines.append('')

    # System info
    lines.append('System Information')
    lines.append('-' * 40)
    lines.append('  Chip:          %s' % system_info.get('chip', 'unknown'))
    lines.append('  Architecture:  %s' % system_info.get('arch', 'unknown'))
    lines.append('  macOS:         %s' % system_info.get('os_version', 'unknown'))
    lines.append('  Memory:        %.1f GB' % system_info.get('memory_gb', 0))
    lines.append('  Rust:          %s' % system_info.get('rust_version', 'unknown'))
    lines.append('  Build profile: %s' % run_metadata.get('primary_profile', 'unknown'))
    lines.append(
        '  Cache mode%s:  %s'
        % (
            's' if len(run_metadata.get('cache_modes', [])) != 1 else '',
            ', '.join(run_metadata.get('cache_modes', [])),
        )
    )
    lines.append('  ripgrep:       %s' % run_metadata.get('primary_version', 'unknown'))
    lines.append('  Binary:        %s' % run_metadata.get('primary_binary', 'unknown'))
    if run_metadata.get('compare_binary'):
        lines.append(
            '  Compare (%s): %s'
            % (
                run_metadata.get('compare_label', 'compare'),
                run_metadata.get('compare_binary', 'unknown'),
            )
        )
        lines.append(
            '  Compare ver.:  %s'
            % run_metadata.get('compare_version', 'unknown')
        )
        lines.append(
            '  Compare prof.: %s'
            % run_metadata.get('compare_profile', 'unknown')
        )
    lines.append('  Timestamp:     %s' % system_info.get('timestamp', ''))
    if 'core_topology' in system_info:
        topo = system_info['core_topology']
        for level_key in sorted(topo.keys()):
            if isinstance(topo[level_key], dict) and 'name' in topo[level_key]:
                entry = topo[level_key]
                lines.append(
                    '  Core %s:    %s logical, %s physical'
                    % (entry['name'], entry.get('logical', '?'),
                       entry.get('physical', '?'))
                )
    if system_info.get('is_apple_silicon'):
        lines.append(
            '  NOTE: macOS does not expose CPU core pinning APIs. '
            'Benchmarks rely on the OS scheduler for P-core placement.'
        )
    lines.append('')

    # Benchmark results
    for suite in suite_results:
        if len(run_metadata.get('cache_modes', [])) > 1:
            suite_header = '%s cache suite' % suite.cache_mode.capitalize()
            lines.append(suite_header)
            lines.append('-' * len(suite_header))
            lines.append('')
        for gr in suite.group_results:
            header = '%s' % gr.group.name
            lines.append(header)
            lines.append('-' * len(header))
            lines.append('  %s' % gr.group.description)
            lines.append('')

            medians = {
                name: br.stats['median']
                for name, br in gr.results.items()
                if br.wall_times
            }
            fastest_name = min(medians, key=medians.get) if medians else None
            max_name_len = max(
                (len(name) for name in gr.results.keys()), default=10
            )

            for cfg_name, br in gr.results.items():
                if not br.wall_times:
                    continue
                st = br.stats
                sample_n = int(st.get('n', len(br.wall_times)))
                lc = br.line_counts[0] if br.line_counts else None
                lc_str = ' (lines: %s)' % lc if lc is not None else ''
                fast_marker = '*' if cfg_name == fastest_name else ' '
                flags = ''
                if br.is_unreliable:
                    flags += ' [UNRELIABLE cv=%.1f%%]' % (st['cv'] * 100)
                if br.outlier_count > 0:
                    outlier_vals = [br.wall_times[idx] for idx in br.outlier_indices]
                    flags += ' [%d OUTLIER(S): %s]' % (
                        br.outlier_count,
                        ', '.join('%.4fs' % v for v in outlier_vals),
                    )
                if br.thermal_tainted_count > 0:
                    flags += ' [%d/%d THERMALLY TAINTED]' % (
                        br.thermal_tainted_count, len(br.wall_times),
                    )

                if sample_n < 20:
                    range_label = ('min', 'max')
                else:
                    range_label = ('p5', 'p95')
                lines.append(
                    '  %s %-*s  median=%.4fs  %s=%.4fs  %s=%.4fs  CV=%.1f%%  cv_mad=%.1f%%%s%s'
                    % (fast_marker, max_name_len + 2, cfg_name,
                       st['median'], range_label[0], st['p5'],
                       range_label[1], st['p95'],
                       st['cv'] * 100, st['cv_mad'] * 100, lc_str, flags)
                )
            lines.append('')

        if suite.comparisons:
            compare_header = 'A/B Comparison (%s cache)' % suite.cache_mode
            lines.append(compare_header)
            lines.append('-' * len(compare_header))
            lines.append('')
            for comp in suite.comparisons:
                if comp['is_regression']:
                    status = 'REGRESSION'
                elif comp['is_improvement']:
                    status = 'IMPROVEMENT'
                elif comp['is_significant']:
                    status = 'SIGNIFICANT'
                else:
                    status = 'NO SIG DIFF'
                power_str = ', power=%.0f%%' % (comp['power'] * 100)
                lines.append(
                    '  [%s] %s/%s: %.4fs -> %.4fs (%+.1f%%, p=%.4f%s)'
                    % (
                        status,
                        comp['group'],
                        comp['config'],
                        comp['reference_median'],
                        comp['candidate_median'],
                        comp['pct_change'] * 100,
                        comp['p_value'],
                        power_str,
                    )
                )
            lines.append('')

    # Regressions
    if regressions:
        lines.append('REGRESSION ANALYSIS')
        lines.append('-' * 40)
        lines.append('  NOTE: Baseline comparison is NOT interleaved A/B.')
        lines.append('  Results may be confounded by environmental differences.')
        lines.append('')
        any_regression = False
        for r in regressions:
            status = 'REGRESSION' if r['is_regression'] else 'OK'
            power_str = ''
            if 'power' in r:
                power_str = ', power=%.0f%%' % (r['power'] * 100)
            lines.append(
                '  [%s] %s: %.4fs -> %.4fs (%+.1f%%, p=%.4f%s)'
                % (status, r['benchmark'],
                   r['baseline_median'], r['new_median'],
                   r['pct_change'] * 100, r['p_value'], power_str)
            )
            if r['is_regression']:
                any_regression = True

        lines.append('')
        if any_regression:
            lines.append('VERDICT: REGRESSION DETECTED')
        else:
            lines.append('VERDICT: No significant regressions')
        lines.append('')

    # Warnings
    if warnings:
        lines.append('WARNINGS')
        lines.append('-' * 40)
        for w in warnings:
            lines.append('  - %s' % w)
        lines.append('')

    return '\n'.join(lines)


def collect_suite_warnings(
    suite_results: Sequence[SuiteResult],
) -> Tuple[List[str], List[str]]:
    """Collect reliability, outlier, and thermal warnings from suite results."""
    unreliable: List[str] = []
    warnings: List[str] = []
    for suite in suite_results:
        for gr in suite.group_results:
            for cfg_name, br in gr.results.items():
                benchmark_key = '%s/%s/%s' % (
                    suite.cache_mode, gr.group.name, cfg_name,
                )
                if br.is_unreliable:
                    unreliable.append(
                        '%s (cv=%.1f%%, threshold=%.0f%%)'
                        % (
                            benchmark_key,
                            br.stats['cv'] * 100,
                            br.cv_threshold * 100,
                        )
                    )
                    warnings.append(
                        'Unreliable: %s has cv=%.1f%% (threshold: %.0f%%)'
                        % (
                            benchmark_key,
                            br.stats['cv'] * 100,
                            br.cv_threshold * 100,
                        )
                    )
                if br.outlier_count > 0:
                    outlier_vals = [br.wall_times[idx] for idx in br.outlier_indices]
                    warnings.append(
                        'Outliers: %s has %d suspected outlier(s): %s'
                        % (
                            benchmark_key,
                            br.outlier_count,
                            ', '.join('%.4fs' % v for v in outlier_vals),
                        )
                    )
                if br.thermal_tainted_count > 0:
                    warnings.append(
                        'Thermal: %s had %d/%d samples collected during '
                        'thermal throttling'
                        % (
                            benchmark_key,
                            br.thermal_tainted_count,
                            len(br.wall_times),
                        )
                    )
    return unreliable, warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Main entry point for the ARM benchmark runner.

    Returns exit code: 0 success, 1 error, 2 unreliable results.
    """
    p = argparse.ArgumentParser(
        description='ARM/Apple Silicon benchmark runner for ripgrep. '
                    'Provides thermal-aware, statistically rigorous '
                    'benchmarks with interleaved A/B execution.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --dir /tmp/benchsuite
  %(prog)s --dir /tmp/benchsuite --scenarios thread_scaling,mmap_vs_read
  %(prog)s --dir /tmp/benchsuite --baseline runs/arm-m5-baseline/raw.csv
  %(prog)s --dir /tmp/benchsuite --compare-bin /tmp/rg-upstream --cache both
  %(prog)s --dir /tmp/benchsuite --list
        ''',
    )
    p.add_argument(
        '--dir', metavar='PATH', default=os.getcwd(),
        help='Directory containing benchmark corpora (default: cwd).',
    )
    p.add_argument(
        '--bin', metavar='PATH', default=None,
        help='Path to ripgrep binary. If not provided, builds from source '
             'using release-lto profile.',
    )
    p.add_argument(
        '--compare-bin', metavar='PATH', default=None,
        help='Path to a second ripgrep binary for interleaved A/B comparison.',
    )
    p.add_argument(
        '--primary-label', metavar='NAME', default='candidate',
        help='Label for the primary binary in A/B comparison output '
             '(default: candidate).',
    )
    p.add_argument(
        '--compare-label', metavar='NAME', default='baseline',
        help='Label for --compare-bin in A/B comparison output '
             '(default: baseline).',
    )
    p.add_argument(
        '--baseline', metavar='PATH', default=None,
        help='Path to baseline raw.csv for regression detection. '
             'NOTE: Not interleaved A/B -- may be confounded by environmental differences.',
    )
    p.add_argument(
        '--scenarios', metavar='LIST', default=None,
        help='Comma-separated list of scenarios to run. '
             'Available: %s. Default: all.' % ', '.join(ALL_SCENARIOS.keys()),
    )
    p.add_argument(
        '--raw', metavar='PATH', default=None,
        help='Path to write raw CSV output.',
    )
    p.add_argument(
        '--json', metavar='PATH', default=None,
        help='Path to write detailed JSON output.',
    )
    p.add_argument(
        '--summary', metavar='PATH', default=None,
        help='Path to write human-readable summary (also printed to stdout).',
    )
    p.add_argument(
        '--cache', choices=['warm', 'cold', 'both'], default='warm',
        help='Cache mode: warm (default), cold, or both '
             '(cold uses purge and requires sudo on macOS).',
    )
    p.add_argument(
        '--sudo', action='store_true', default=False,
        help='Enable sudo for powermetrics and purge.',
    )
    p.add_argument(
        '--cooldown', type=int, default=COOLDOWN_MIN_SECONDS,
        help='Cooldown seconds between benchmark groups (default: %d).'
             % COOLDOWN_MIN_SECONDS,
    )
    p.add_argument(
        '--list', action='store_true',
        help='List available benchmark scenarios and exit.',
    )
    p.add_argument(
        '-f', '--force', action='store_true',
        help='Overwrite existing output files.',
    )
    p.add_argument(
        '--samples', type=int, default=None,
        help='Override default sample count for all scenarios. '
             'Thread-scaling tests use max(--samples, %d).' % THREAD_SCALING_SAMPLES,
    )
    p.add_argument(
        '--convergence-threshold', type=float, default=0.05,
        help='Convergence threshold for warmup steady-state detection '
             '(default: 0.05). All pairwise differences in the last 3 '
             'warmup times must be within this fraction of the faster time.',
    )
    p.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducible benchmark ordering. '
             'If not provided, a seed is generated and recorded in output.',
    )

    args = p.parse_args()

    # Validate --samples minimum
    if args.samples is not None and args.samples < 5:
        eprint('WARNING: --samples %d is below the CLAUDE.md minimum of 5. '
               'Results may not be statistically meaningful.' % args.samples)

    cache_modes = parse_cache_modes(args.cache)
    if args.compare_bin and args.baseline:
        eprint(
            'ERROR: --compare-bin and --baseline are mutually exclusive. '
            'Use interleaved A/B comparison or baseline CSV comparison, not both.'
        )
        return 1
    if args.primary_label == args.compare_label:
        eprint('ERROR: --primary-label and --compare-label must differ.')
        return 1
    if 'cold' in cache_modes and is_macos() and not args.sudo:
        eprint(
            'ERROR: cold-cache benchmarks require --sudo on macOS so the '
            'page cache can be purged deterministically.'
        )
        return 1

    # List mode
    if args.list:
        for name in ALL_SCENARIOS:
            print(name)
        return 0

    # ARM detection warning
    if not is_apple_silicon():
        eprint(
            'WARNING: Not running on Apple Silicon. '
            'This benchmark suite is designed for ARM/M-series Macs. '
            'Results on other architectures may not be meaningful for '
            'ARM-specific validation. Continuing anyway...'
        )

    # Detect system info
    system_info = detect_system_info()

    # Seed random number generator for reproducibility
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed)
    system_info['random_seed'] = seed
    eprint('Random seed: %d' % seed)

    eprint('System: %s (%s)' % (
        system_info.get('chip', 'unknown'),
        system_info.get('arch', 'unknown'),
    ))

    # Check for background interference
    bg_warnings = check_background_interference()
    for bw in bg_warnings:
        eprint('WARNING: %s' % bw)

    # Determine project root (assume arm_bench.py is in benchsuite/)
    script_dir = path.dirname(path.abspath(__file__))
    project_root = path.dirname(script_dir)

    # Build or locate binary
    warnings: List[str] = []
    primary_profile = 'unknown'

    if args.bin:
        rg_bin = path.abspath(args.bin)
        if not path.exists(rg_bin):
            eprint('ERROR: Binary not found: %s' % rg_bin)
            return 1
        primary_profile = detect_binary_profile(rg_bin)
        if primary_profile == 'release':
            warnings.append(
                'Binary appears to be from --release (not release-lto). '
                'LTO builds are required for canonical benchmarks.'
            )
        elif primary_profile == 'custom':
            warnings.append('Unable to determine build profile of binary.')
    else:
        # Try to find an existing binary before building
        existing = find_rg_binary(project_root)
        if existing:
            rg_bin = existing
            primary_profile = detect_binary_profile(rg_bin)
            if primary_profile != 'release-lto':
                warnings.append(
                    'Found existing binary at release path (not release-lto). '
                    'LTO builds are required for canonical benchmarks.'
                )
            eprint('Found existing binary: %s' % rg_bin)
        else:
            rg_bin, primary_profile, build_warnings = build_rg(project_root)
            warnings.extend(build_warnings)
            if not path.exists(rg_bin):
                eprint('ERROR: Build succeeded but binary not found at: %s' % rg_bin)
                return 1

    compare_bin: Optional[str] = None
    compare_profile: Optional[str] = None
    compare_version: Optional[str] = None
    if args.compare_bin:
        compare_bin = path.abspath(args.compare_bin)
        if not path.exists(compare_bin):
            eprint('ERROR: Compare binary not found: %s' % compare_bin)
            return 1
        compare_profile = detect_binary_profile(compare_bin)
        if compare_profile == 'release':
            warnings.append(
                'Compare binary appears to be from --release (not release-lto).'
            )
        compare_version = get_rg_version(compare_bin)

    # Record rg version
    rg_version = get_rg_version(rg_bin)
    system_info['rg_version'] = rg_version
    system_info['rg_binary'] = rg_bin
    system_info['build_profile'] = primary_profile
    eprint('ripgrep: %s (%s, profile=%s)' % (rg_version, rg_bin, primary_profile))
    if compare_bin is not None:
        eprint(
            'compare: %s (%s, profile=%s)'
            % (compare_version or 'unknown', compare_bin, compare_profile or 'unknown')
        )

    # Ensure suite dir exists
    suite_dir = path.abspath(args.dir)
    if not path.isdir(suite_dir):
        eprint('ERROR: Suite directory does not exist: %s' % suite_dir)
        eprint('Run: benchsuite/benchsuite --dir %s --download all' % suite_dir)
        return 1

    # Determine scenarios to run
    if args.scenarios:
        scenario_names = [s.strip() for s in args.scenarios.split(',')]
        for s in scenario_names:
            if s not in ALL_SCENARIOS:
                eprint('ERROR: Unknown scenario: %s' % s)
                eprint('Available: %s' % ', '.join(ALL_SCENARIOS.keys()))
                return 1
    else:
        scenario_names = list(ALL_SCENARIOS.keys())

    # Check output file conflicts
    for outpath in [args.raw, args.json, args.summary]:
        if outpath and path.exists(outpath) and not args.force:
            eprint(
                'ERROR: File %s already exists (use --force to overwrite)'
                % outpath
            )
            return 1

    # Initialize thermal monitor
    thermal = ThermalMonitor(use_sudo=args.sudo)
    baseline_thermal = thermal.capture_baseline()
    eprint('Thermal state: %s' % baseline_thermal)

    if not args.sudo:
        eprint(
            'NOTE: Running without --sudo. Thermal monitoring limited to '
            'sysctl readings. Use --sudo for powermetrics-based monitoring.'
        )

    # Build scenario groups
    groups: List[BenchmarkGroup] = []
    for name in scenario_names:
        try:
            if compare_bin is not None:
                group = build_ab_group(
                    name,
                    rg_bin,
                    compare_bin,
                    suite_dir,
                    system_info,
                    args.samples,
                    args.primary_label,
                    args.compare_label,
                )
            else:
                group = build_scenario_group(
                    name, rg_bin, suite_dir, system_info, args.samples
                )
            groups.append(group)
        except RuntimeError as e:
            eprint('SKIP: %s (%s)' % (name, e))
            continue

    if not groups:
        eprint('ERROR: No scenarios to run. Check corpus availability.')
        return 1

    # Add background warnings to warning list
    warnings.extend(bg_warnings)

    # Run benchmarks, once per selected cache mode
    suite_results: List[SuiteResult] = []
    for suite_idx, cache_mode in enumerate(cache_modes):
        if len(cache_modes) > 1:
            eprint('\n### Running %s-cache suite ###' % cache_mode)
        runner = BenchmarkRunner(
            rg_bin=rg_bin,
            suite_dir=suite_dir,
            thermal=thermal,
            cache_mode=cache_mode,
            use_sudo=args.sudo,
            convergence_threshold=args.convergence_threshold,
        )

        group_results: List[GroupResult] = []
        for i, group in enumerate(groups):
            result = runner.run_group(group)
            group_results.append(result)

            # Cooldown between groups (skip after last within suite)
            if i < len(groups) - 1:
                thermal.cooldown(args.cooldown)

        suite_results.append(SuiteResult(cache_mode, group_results))

        # Cooldown between cache suites
        if suite_idx < len(cache_modes) - 1:
            thermal.cooldown(args.cooldown)

    if compare_bin is not None:
        warnings.extend(
            compare_ab_suites(
                suite_results, args.primary_label, args.compare_label
            )
        )

    unreliable_benchmarks, suite_warnings = collect_suite_warnings(suite_results)
    warnings.extend(suite_warnings)

    # Regression detection
    regressions: List[Dict[str, Any]] = []
    if args.baseline:
        if not path.exists(args.baseline):
            eprint('WARNING: Baseline file not found: %s' % args.baseline)
            warnings.append('Baseline file not found: %s' % args.baseline)
        else:
            baseline = load_baseline(args.baseline)
            regressions = check_regressions(suite_results, baseline)

    run_metadata: Dict[str, Any] = {
        'primary_binary': rg_bin,
        'primary_version': rg_version,
        'primary_profile': primary_profile,
        'cache_modes': cache_modes,
    }
    if compare_bin is not None:
        run_metadata.update({
            'compare_binary': compare_bin,
            'compare_version': compare_version,
            'compare_profile': compare_profile,
            'primary_label': args.primary_label,
            'compare_label': args.compare_label,
        })

    # Generate summary
    summary_text = format_summary(
        suite_results, system_info, regressions, warnings, run_metadata,
    )

    # Print summary to stdout
    print(summary_text)

    # Write output files
    if args.raw:
        write_csv(args.raw, suite_results, system_info)
        eprint('CSV written to: %s' % args.raw)

    if args.json:
        write_json(
            args.json, suite_results, system_info,
            regressions, warnings, run_metadata,
        )
        eprint('JSON written to: %s' % args.json)

    if args.summary:
        with open(args.summary, 'w') as f:
            f.write(summary_text)
        eprint('Summary written to: %s' % args.summary)

    # Exit code
    if unreliable_benchmarks:
        eprint(
            'EXIT CODE 2: Unreliable results detected for: %s'
            % '; '.join(unreliable_benchmarks)
        )
        return 2

    # Check for regressions
    has_regression = any(r['is_regression'] for r in regressions)
    has_compare_regression = any(
        comp['is_regression']
        for suite in suite_results
        for comp in suite.comparisons
    )
    if has_regression or has_compare_regression:
        if has_regression:
            eprint('EXIT CODE 1: Regression(s) detected')
        else:
            eprint('EXIT CODE 1: Interleaved A/B regression(s) detected')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
