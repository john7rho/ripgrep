import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


sys.dont_write_bytecode = True

MODULE_PATH = pathlib.Path(__file__).with_name("arm_bench.py")
SPEC = importlib.util.spec_from_file_location("arm_bench", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
arm_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(arm_bench)


def make_result(name, logical_name, label):
    cfg = arm_bench.BenchmarkConfig(
        name=name,
        logical_name=logical_name,
        variant_label=label,
        cmd=["rg", logical_name],
    )
    return cfg


class ArmBenchWorkflowTests(unittest.TestCase):
    def test_probe_decision_metadata_parses_trace_lines(self):
        stderr = "\n".join([
            "DEBUG|rg::flags::hiargs|using 6 thread(s)",
            "DEBUG|rg::flags::hiargs|auto-mmap enabled for 1 explicit file(s): total_bytes=10485760, max_file_bytes=10485760, multiline=true",
            'TRACE|grep_searcher::searcher|Some("hay"): searching via memory map (multiline)',
        ])

        class FakeProc:
            def communicate(self, timeout=None):
                return "", stderr

        with mock.patch.object(arm_bench.subprocess, "Popen", return_value=FakeProc()):
            metadata = arm_bench.probe_decision_metadata(
                ["rg", "-n", "-U", "Sherlock\\nHolmes", "hay"],
                None,
            )

        self.assertEqual(6, metadata["threads_selected"])
        self.assertTrue(metadata["auto_mmap_enabled"])
        self.assertEqual("mmap", metadata["search_strategy"])
        self.assertTrue(metadata["multiline_with_matcher"])

    def test_infer_decision_metadata_for_mmap_multiline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hay = pathlib.Path(tmpdir) / "hay.txt"
            hay.write_text("Sherlock\nHolmes\n", encoding="utf-8")
            metadata = arm_bench.infer_decision_metadata(
                ["rg", "-n", "-U", "--mmap", r"Sherlock\nHolmes", str(hay)],
                None,
                {
                    "is_apple_silicon": True,
                    "total_logical_cpus": 10,
                    "core_topology": {
                        "level0": {"name": "Performance", "logical": 6}
                    },
                },
            )

        self.assertEqual(1, metadata["threads_selected"])
        self.assertEqual("always", metadata["mmap_mode"])
        self.assertTrue(metadata["effective_mmap_enabled"])
        self.assertTrue(metadata["multiline_with_matcher"])
        self.assertEqual("mmap", metadata["search_strategy"])

    def test_summarize_xctrace_toc_extracts_schemas(self):
        summary = arm_bench.summarize_xctrace_toc(
            """
            <trace-toc>
              <run number="1" duration="1.23" template-name="Time Profiler">
                <data>
                  <table schema="time-profile" />
                  <table schema="cpu-usage" />
                </data>
                <tracks>
                  <track name="CPU Strategy" />
                </tracks>
              </run>
            </trace-toc>
            """
        )

        self.assertEqual(["time-profile", "cpu-usage"], summary["available_schemas"])
        self.assertEqual(["CPU Strategy"], summary["tracks"])
        self.assertEqual("1.23", summary["duration"])

    def test_xctrace_profiler_selects_best_delta_pair(self):
        cfg_fast = arm_bench.BenchmarkConfig(
            name="fast", logical_name="pair", variant_label="candidate", cmd=["rg", "fast"]
        )
        cfg_slow = arm_bench.BenchmarkConfig(
            name="slow", logical_name="pair", variant_label="baseline", cmd=["rg", "slow"]
        )
        group = arm_bench.BenchmarkGroup(
            name="synthetic",
            description="synthetic",
            configs=[cfg_fast, cfg_slow],
        )
        result = arm_bench.GroupResult(group)
        result.results["fast"].wall_times = [1.0, 1.1, 1.2]
        result.results["fast"].thermal_states = ["nominal"] * 3
        result.results["slow"].wall_times = [2.0, 2.1, 2.2]
        result.results["slow"].thermal_states = ["nominal"] * 3

        with mock.patch.object(arm_bench, "detect_xctrace_version", return_value="xctrace 1.0"):
            profiler = arm_bench.XctraceProfiler(
                mode="time-profiler",
                output_dir="/tmp/profile-out",
                on_best_delta=True,
            )

        targets = profiler.select_targets(group, result)
        names = {cfg.name for cfg, _, _ in targets}
        self.assertEqual({"fast", "slow"}, names)

    def test_xctrace_profiler_capture_sets_time_limit(self):
        cfg = arm_bench.BenchmarkConfig(
            name="rare",
            cmd=["rg", "-c", "PM_RESUME"],
            cwd="/tmp/linux",
        )
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))

            class Result:
                def __init__(self, returncode, stdout="", stderr=""):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            if cmd[:3] == ["xcrun", "xctrace", "record"]:
                trace_path = pathlib.Path(cmd[cmd.index("--output") + 1])
                trace_path.mkdir(parents=True, exist_ok=True)
                return Result(0)
            if "--toc" in cmd:
                return Result(
                    0,
                    (
                        "<trace-toc><run number='1' duration='3.2'>"
                        "<data><table schema='time-profile' /></data>"
                        "</run></trace-toc>"
                    ),
                )
            return Result(0, "<table />")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(arm_bench, "detect_xctrace_version", return_value="xctrace 1.0"):
                profiler = arm_bench.XctraceProfiler(
                    mode="time-profiler",
                    output_dir=tmpdir,
                )
            with mock.patch.object(arm_bench.subprocess, "run", side_effect=fake_run):
                artifact = profiler.capture(
                    "warm",
                    "directory_io",
                    cfg,
                    sample_idx=0,
                    selection_reason="profile-samples",
                    use_sudo=False,
                    sample_wall_time=2.25,
                )

        record_cmd = calls[0][0]
        self.assertIn("--time-limit", record_cmd)
        self.assertEqual("14s", record_cmd[record_cmd.index("--time-limit") + 1])
        self.assertEqual(14, artifact["record_time_limit_seconds"])
        self.assertEqual(2.25, artifact["sample_wall_time_seconds"])
        self.assertEqual("ok", artifact["status"])

    def test_xctrace_profiler_capture_ignores_successful_nonzero_returncode(self):
        cfg = arm_bench.BenchmarkConfig(
            name="rare",
            cmd=["rg", "-c", "PM_RESUME"],
            cwd="/tmp/linux",
        )

        def fake_run(cmd, **kwargs):
            class Result:
                def __init__(self, returncode, stdout="", stderr=""):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            if cmd[:3] == ["xcrun", "xctrace", "record"]:
                trace_path = pathlib.Path(cmd[cmd.index("--output") + 1])
                trace_path.mkdir(parents=True, exist_ok=True)
                return Result(
                    54,
                    (
                        "Starting recording with the Time Profiler template.\n"
                        "Ctrl-C to stop the recording\n"
                        "Reached specified time limit, ending recording...\n"
                        "Recording completed. Saving output file...\n"
                        "Output file saved as: sample.trace\n"
                    ),
                    "",
                )
            if "--toc" in cmd:
                return Result(
                    0,
                    (
                        "<trace-toc><run number='1' duration='3.2'>"
                        "<data><table schema='time-profile' /></data>"
                        "</run></trace-toc>"
                    ),
                )
            return Result(0, "<table />")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(arm_bench, "detect_xctrace_version", return_value="xctrace 1.0"):
                profiler = arm_bench.XctraceProfiler(
                    mode="time-profiler",
                    output_dir=tmpdir,
                )
            with mock.patch.object(arm_bench.subprocess, "run", side_effect=fake_run):
                artifact = profiler.capture(
                    "warm",
                    "directory_io",
                    cfg,
                    sample_idx=0,
                    selection_reason="profile-samples",
                    use_sudo=False,
                    sample_wall_time=1.0,
                )

        self.assertEqual("ok", artifact["status"])
        self.assertEqual(54, artifact["record_returncode"])
        self.assertTrue(artifact["record_returncode_ignored"])
        self.assertEqual([], artifact["warnings"])

    def test_parse_cache_modes_both(self):
        self.assertEqual(
            ["warm", "cold"],
            arm_bench.parse_cache_modes("both"),
        )

    def test_compare_ab_suites_produces_interleaved_regression(self):
        baseline_cfg = make_result(
            "[baseline] rg literal", "rg literal", "baseline",
        )
        candidate_cfg = make_result(
            "[candidate] rg literal", "rg literal", "candidate",
        )
        group = arm_bench.BenchmarkGroup(
            name="large_file_single_threaded",
            description="synthetic",
            configs=[candidate_cfg, baseline_cfg],
        )
        group_result = arm_bench.GroupResult(group)
        group_result.results[candidate_cfg.name].wall_times = [
            1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27
        ]
        group_result.results[baseline_cfg.name].wall_times = [
            1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07
        ]

        suite = arm_bench.SuiteResult("warm", [group_result])
        warnings = arm_bench.compare_ab_suites(
            [suite], "candidate", "baseline"
        )

        self.assertEqual([], warnings)
        self.assertEqual(1, len(suite.comparisons))
        comparison = suite.comparisons[0]
        self.assertEqual("warm", comparison["cache_mode"])
        self.assertEqual("rg literal", comparison["config"])
        self.assertTrue(comparison["is_regression"])
        self.assertGreater(comparison["pct_change"], 0.05)
        self.assertLess(comparison["p_value"], 0.05)

    def test_compare_ab_suites_prefers_clean_samples(self):
        baseline_cfg = make_result(
            "[baseline] rg literal", "rg literal", "baseline",
        )
        candidate_cfg = make_result(
            "[candidate] rg literal", "rg literal", "candidate",
        )
        group = arm_bench.BenchmarkGroup(
            name="large_file_single_threaded",
            description="synthetic",
            configs=[candidate_cfg, baseline_cfg],
        )
        group_result = arm_bench.GroupResult(group)
        candidate = group_result.results[candidate_cfg.name]
        baseline = group_result.results[baseline_cfg.name]

        baseline.wall_times = [1.00, 1.01, 1.02, 1.03, 1.04, 8.00]
        baseline.thermal_states = ["nominal"] * 5 + ["fair"]
        candidate.wall_times = [1.20, 1.21, 1.22, 1.23, 1.24, 0.20]
        candidate.thermal_states = ["nominal"] * 5 + ["fair"]

        suite = arm_bench.SuiteResult("warm", [group_result])
        warnings = arm_bench.compare_ab_suites(
            [suite], "candidate", "baseline"
        )

        self.assertEqual(1, len(warnings))
        self.assertIn("n < 8", warnings[0])
        comparison = suite.comparisons[0]
        self.assertEqual(5, comparison["n_reference"])
        self.assertEqual(5, comparison["n_candidate"])
        self.assertTrue(comparison["is_regression"])

    def test_check_regressions_uses_cache_mode_keys(self):
        cfg = arm_bench.BenchmarkConfig(
            name="rg literal",
            logical_name="rg literal",
            cmd=["rg", "literal"],
        )
        group = arm_bench.BenchmarkGroup(
            name="large_file_single_threaded",
            description="synthetic",
            configs=[cfg],
        )
        group_result = arm_bench.GroupResult(group)
        group_result.results[cfg.name].wall_times = [
            1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27
        ]
        suite = arm_bench.SuiteResult("cold", [group_result])

        baseline = {
            "cold/large_file_single_threaded/rg literal": [
                1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07
            ]
        }
        old_eprint = arm_bench.eprint
        try:
            arm_bench.eprint = lambda *args, **kwargs: None
            regressions = arm_bench.check_regressions([suite], baseline)
        finally:
            arm_bench.eprint = old_eprint

        self.assertEqual(1, len(regressions))
        self.assertEqual("cold", regressions[0]["cache_mode"])
        self.assertTrue(regressions[0]["is_regression"])

    def test_write_csv_emits_cache_mode_and_variant_columns(self):
        cfg = arm_bench.BenchmarkConfig(
            name="[candidate] rg literal",
            logical_name="rg literal",
            variant_label="candidate",
            cmd=["rg", "literal"],
        )
        group = arm_bench.BenchmarkGroup(
            name="large_file_single_threaded",
            description="synthetic",
            configs=[cfg],
        )
        group_result = arm_bench.GroupResult(group)
        group_result.results[cfg.name].wall_times = [1.0, 1.1, 1.2, 9.0, 1.4]
        group_result.results[cfg.name].user_times = [0.1] * 5
        group_result.results[cfg.name].sys_times = [0.01] * 5
        group_result.results[cfg.name].thermal_states = [
            "nominal", "fair", "nominal", "nominal", "nominal"
        ]
        group_result.results[cfg.name].decision_metadata = {
            "threads_selected": 4,
            "threads_reason": "reported by --trace probe",
            "apple_pcore_count_detected": 6,
            "mmap_mode": "auto",
            "auto_mmap_enabled": True,
            "auto_mmap_reason": "enabled for 1 explicit file(s)",
            "effective_mmap_enabled": True,
            "path_mode": "explicit_files",
            "explicit_file_count": 1,
            "explicit_total_bytes": 1048576,
            "explicit_max_file_bytes": 1048576,
            "multiline_enabled": False,
            "multiline_dotall": False,
            "multiline_with_matcher": False,
            "search_strategy": "slice_by_line",
            "search_strategy_detail": "slice reader: searching via slice-by-line strategy",
            "decision_source": "probe+inference",
        }
        group_result.results[cfg.name].profile_artifacts[0] = {
            "profile_mode": "time-profiler",
            "selection_reason": "profile-samples",
            "trace_path": "/tmp/sample.trace",
            "summary_path": "/tmp/sample.summary.json",
            "export_schema": "time-profile",
            "export_path": "/tmp/sample.export.xml",
        }
        suite = arm_bench.SuiteResult("warm", [group_result])

        system_info = {"chip": "Test", "core_topology": {"logical": 4}}
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = pathlib.Path(tmpdir) / "out.csv"
            arm_bench.write_csv(str(outpath), [suite], system_info)
            with outpath.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(5, len(rows))
        self.assertEqual("warm", rows[0]["cache_mode"])
        self.assertEqual("candidate", rows[0]["binary_label"])
        self.assertEqual("rg literal", rows[0]["logical_name"])
        self.assertEqual("1", rows[0]["clean_sample"])
        self.assertEqual("0", rows[1]["clean_sample"])
        self.assertEqual("0", rows[3]["clean_sample"])
        self.assertEqual("3", rows[0]["clean_n"])
        self.assertEqual("1.000000", rows[0]["best_clean"])
        self.assertEqual("4", rows[0]["threads_selected"])
        self.assertEqual("1", rows[0]["auto_mmap_enabled"])
        self.assertEqual("slice_by_line", rows[0]["search_strategy"])
        self.assertEqual("/tmp/sample.trace", rows[0]["profile_trace_path"])

    def test_write_json_emits_decision_metadata_and_samples(self):
        cfg = arm_bench.BenchmarkConfig(
            name="rg literal",
            logical_name="rg literal",
            cmd=["rg", "literal"],
        )
        group = arm_bench.BenchmarkGroup(
            name="large_file_single_threaded",
            description="synthetic",
            configs=[cfg],
        )
        group_result = arm_bench.GroupResult(group)
        br = group_result.results[cfg.name]
        br.wall_times = [1.0, 1.1]
        br.user_times = [0.1, 0.1]
        br.sys_times = [0.01, 0.02]
        br.thermal_states = ["nominal", "nominal"]
        br.major_faults = [1, 2]
        br.minor_faults = [10, 20]
        br.decision_metadata = {
            "threads_selected": 1,
            "search_strategy": "read_by_line",
        }
        br.profile_artifacts[1] = {
            "profile_mode": "time-profiler",
            "trace_path": "/tmp/profile.trace",
            "summary_path": "/tmp/profile.summary.json",
        }
        suite = arm_bench.SuiteResult("warm", [group_result])

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = pathlib.Path(tmpdir) / "out.json"
            arm_bench.write_json(
                str(outpath),
                [suite],
                {"chip": "Test"},
                [],
                [],
                {"cache_modes": ["warm"]},
            )
            data = json.loads(outpath.read_text())

        config = data["benchmarks"]["large_file_single_threaded"]["configs"]["rg literal"]
        self.assertEqual(1, config["decision_metadata"]["threads_selected"])
        self.assertEqual("read_by_line", config["samples"][0]["decision_metadata"]["search_strategy"])
        self.assertIn("profile_artifact", config["samples"][1])

    def test_load_baseline_skips_unclean_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = pathlib.Path(tmpdir) / "baseline.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, ["benchmark", "name", "cache_mode", "duration", "clean_sample"]
                )
                writer.writeheader()
                writer.writerow({
                    "benchmark": "thread_scaling",
                    "name": "rg -j4",
                    "cache_mode": "warm",
                    "duration": "1.0",
                    "clean_sample": "1",
                })
                writer.writerow({
                    "benchmark": "thread_scaling",
                    "name": "rg -j4",
                    "cache_mode": "warm",
                    "duration": "9.0",
                    "clean_sample": "0",
                })

            baseline = arm_bench.load_baseline(str(csv_path))

        self.assertEqual(
            {"warm/thread_scaling/rg -j4": [1.0]},
            baseline,
        )

    def test_check_background_interference_targets_suite_dir(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)

            class Result:
                returncode = 0
                stdout = "Indexing enabled."

            return Result()

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(arm_bench.subprocess, "run", side_effect=fake_run):
                warnings = arm_bench.check_background_interference(tmpdir)

        self.assertTrue(warnings)
        self.assertEqual(["mdutil", "-s", tmpdir], calls[0])


if __name__ == "__main__":
    unittest.main()
