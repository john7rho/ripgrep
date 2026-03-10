import csv
import importlib.util
import pathlib
import sys
import tempfile
import unittest


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
        group_result.results[cfg.name].wall_times = [1.0, 1.1, 1.2, 1.3, 1.4]
        group_result.results[cfg.name].user_times = [0.1] * 5
        group_result.results[cfg.name].sys_times = [0.01] * 5
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


if __name__ == "__main__":
    unittest.main()
