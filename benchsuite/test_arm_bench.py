import csv
import importlib.util
import json
import pathlib
import plistlib
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
    def test_summarize_powermetrics_capture_extracts_pcore_share(self):
        payload_a = {
            "tasks": [
                {
                    "pid": 4242,
                    "name": "rg",
                    "cputime_sample_ms_per_s": 60.0,
                    "cpu_cycles": 100.0,
                    "pcpu_cycles": 75.0,
                    "cpu_instructions": 200.0,
                    "pcpu_instructions": 150.0,
                    "clusters": [
                        {"name": "P-Cluster", "cputime_sample_ms_per_s": 55.0},
                        {"name": "E-Cluster", "cputime_sample_ms_per_s": 5.0},
                    ],
                }
            ]
        }
        payload_b = {
            "tasks": [
                {
                    "pid": 4242,
                    "name": "rg",
                    "cputime_sample_ms_per_s": 40.0,
                    "cpu_cycles": 100.0,
                    "pcpu_cycles": 50.0,
                    "cpu_instructions": 100.0,
                    "pcpu_instructions": 50.0,
                    "clusters": [
                        {"name": "P-Cluster", "cputime_sample_ms_per_s": 20.0},
                        {"name": "E-Cluster", "cputime_sample_ms_per_s": 20.0},
                    ],
                }
            ]
        }
        raw = plistlib.dumps(payload_a) + b"\0" + plistlib.dumps(payload_b)

        summary = arm_bench.summarize_powermetrics_capture(raw, 4242)

        self.assertEqual("ok", summary["telemetry_status"])
        self.assertTrue(summary["p_core_dispatch_observed"])
        self.assertEqual(2, summary["telemetry_sample_count"])
        self.assertEqual(2, summary["p_core_dispatch_sample_count"])
        self.assertAlmostEqual(0.625, summary["p_core_share"])
        self.assertEqual(75.0, summary["p_core_cycles"])

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
        group_result.results[cfg.name].sample_metadata = [
            {
                "contention_active": True,
                "contention_kind": "cpu_hoggers",
                "contention_workers": 4,
                "telemetry_status": "ok",
                "p_core_dispatch_observed": True,
                "p_core_share": 0.75,
                "p_core_cycles": 75.0,
                "cpu_cycles": 100.0,
                "sample_cpu_ms_per_s": 40.0,
                "cluster_cpu_ms_per_s": {"P-Cluster": 35.0, "E-Cluster": 5.0},
            }
        ] + [{} for _ in range(4)]
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
        self.assertEqual("1", rows[0]["contention"])
        self.assertEqual("cpu_hoggers", rows[0]["contention_kind"])
        self.assertEqual("4", rows[0]["contention_workers"])
        self.assertEqual("ok", rows[0]["telemetry_status"])
        self.assertEqual("1", rows[0]["p_core_dispatch"])
        self.assertEqual("0.7500", rows[0]["p_core_share"])

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

    def test_thread_scaling_contention_requires_sudo_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            linux_dir = pathlib.Path(tmpdir) / "linux"
            linux_dir.mkdir()
            (linux_dir / "Makefile").write_text("all:\n", encoding="utf-8")
            system_info = {
                "is_apple_silicon": True,
                "powermetrics_available": True,
                "sudo_enabled": True,
                "sudo_ready": False,
                "core_topology": {
                    "level0": {"name": "Performance", "logical": 4, "physical": 4},
                    "level1": {"name": "Efficiency", "logical": 4, "physical": 4},
                },
                "total_logical_cpus": 8,
            }

            with self.assertRaises(RuntimeError):
                arm_bench.scenario_thread_scaling_contention("rg", tmpdir, system_info)

    def test_thread_scaling_contention_builds_controller_backed_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            linux_dir = pathlib.Path(tmpdir) / "linux"
            linux_dir.mkdir()
            (linux_dir / "Makefile").write_text("all:\n", encoding="utf-8")
            system_info = {
                "is_apple_silicon": True,
                "powermetrics_available": True,
                "sudo_enabled": True,
                "sudo_ready": True,
                "core_topology": {
                    "level0": {"name": "Performance", "logical": 4, "physical": 4},
                    "level1": {"name": "Efficiency", "logical": 4, "physical": 4},
                },
                "total_logical_cpus": 8,
            }

            group = arm_bench.scenario_thread_scaling_contention("rg", tmpdir, system_info)

        self.assertIn("thread_scaling_contention", arm_bench.ALL_SCENARIOS)
        self.assertEqual("thread_scaling_contention", group.name)
        self.assertTrue(group.configs)
        self.assertTrue(all(cfg.sample_controller_factory is not None for cfg in group.configs))
        self.assertIn("contend", group.configs[0].name)

    def test_execute_config_uses_sample_controller_hooks(self):
        events = []

        class FakeController(arm_bench.SampleController):
            def __init__(self, collect_telemetry):
                self.collect_telemetry = collect_telemetry
                events.append(("init", collect_telemetry))

            def start(self):
                events.append("start")

            def attach_process(self, proc):
                events.append(("attach", proc.pid))

            def stop(self):
                events.append("stop")
                return {"contention_active": True, "telemetry_status": "ok"}

        cfg = arm_bench.BenchmarkConfig(
            name="rg literal",
            logical_name="rg literal",
            cmd=["rg", "literal"],
            sample_controller_factory=lambda collect: FakeController(collect),
        )
        runner = arm_bench.BenchmarkRunner(
            rg_bin="rg",
            suite_dir=".",
            thermal=mock.Mock(),
        )

        def fake_run_timed(cmd, cwd=None, count_lines=False, process_started=None):
            proc = mock.Mock()
            proc.pid = 321
            if process_started is not None:
                process_started(proc)
            return arm_bench.TimingResult(
                wall_time=1.0,
                user_time=0.2,
                sys_time=0.1,
                line_count=None,
                returncode=0,
            )

        with mock.patch.object(arm_bench, "run_timed", side_effect=fake_run_timed):
            _, metadata = runner._execute_config(
                cfg, count_lines=False, collect_telemetry=True
            )

        self.assertEqual(
            [("init", True), "start", ("attach", 321), "stop"],
            events,
        )
        self.assertEqual("ok", metadata["telemetry_status"])

    def test_format_summary_includes_scheduler_placement(self):
        cfg = arm_bench.BenchmarkConfig(
            name="rg -j4 (contended)",
            logical_name="rg -j4",
            cmd=["rg", "-j4", "literal"],
        )
        group = arm_bench.BenchmarkGroup(
            name="thread_scaling_contention",
            description="synthetic",
            configs=[cfg],
        )
        group_result = arm_bench.GroupResult(group)
        result = group_result.results[cfg.name]
        result.wall_times = [1.0, 1.1, 1.2, 1.05, 1.15]
        result.user_times = [0.1] * 5
        result.sys_times = [0.01] * 5
        result.thermal_states = ["nominal"] * 5
        result.sample_metadata = [
            {
                "contention_active": True,
                "telemetry_status": "ok",
                "p_core_dispatch_observed": True,
                "p_core_share": 0.80,
            },
            {
                "contention_active": True,
                "telemetry_status": "ok",
                "p_core_dispatch_observed": True,
                "p_core_share": 0.70,
            },
            {
                "contention_active": True,
                "telemetry_status": "ok",
                "p_core_dispatch_observed": False,
                "p_core_share": 0.0,
            },
            {"contention_active": True, "telemetry_status": "no_matching_task"},
            {"contention_active": True, "telemetry_status": "powermetrics_failed"},
        ]
        suite = arm_bench.SuiteResult("warm", [group_result])

        summary = arm_bench.format_summary(
            [suite],
            {"chip": "Test", "arch": "arm64", "os_version": "14.0", "memory_gb": 16, "rust_version": "rustc test"},
            [],
            [],
            {"primary_profile": "release-lto", "cache_modes": ["warm"], "primary_version": "rg test", "primary_binary": "/tmp/rg", "sample_retries": 1},
        )

        self.assertIn("Scheduler placement:", summary)
        self.assertIn("P-core dispatch observed in 2/3 telemetry sample(s)", summary)

    def test_write_json_includes_sample_metadata_and_placement_summary(self):
        cfg = arm_bench.BenchmarkConfig(
            name="rg -j4 (contended)",
            logical_name="rg -j4",
            cmd=["rg", "-j4", "literal"],
        )
        group = arm_bench.BenchmarkGroup(
            name="thread_scaling_contention",
            description="synthetic",
            configs=[cfg],
        )
        group_result = arm_bench.GroupResult(group)
        result = group_result.results[cfg.name]
        result.wall_times = [1.0, 1.1, 1.2, 1.3, 1.4]
        result.user_times = [0.1] * 5
        result.sys_times = [0.01] * 5
        result.thermal_states = ["nominal"] * 5
        result.sample_metadata = [
            {
                "contention_active": True,
                "telemetry_status": "ok",
                "p_core_dispatch_observed": True,
                "p_core_share": 0.75,
            }
        ] + [{} for _ in range(4)]
        suite = arm_bench.SuiteResult("warm", [group_result])

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = pathlib.Path(tmpdir) / "out.json"
            arm_bench.write_json(
                str(outpath),
                [suite],
                {"chip": "Test"},
                [],
                [],
                {"primary_binary": "/tmp/rg", "primary_version": "rg test", "primary_profile": "release-lto", "cache_modes": ["warm"]},
            )
            payload = json.loads(outpath.read_text(encoding="utf-8"))

        config_data = payload["benchmarks"]["thread_scaling_contention"]["configs"]["rg -j4 (contended)"]
        self.assertIn("sample_metadata", config_data)
        self.assertIn("placement_summary", config_data)
        self.assertTrue(config_data["placement_summary"]["contention_enabled"])


if __name__ == "__main__":
    unittest.main()
