"""V15 suite selection, initialization identity and safe artifact publication."""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import v15_experiments as suite


class V15LauncherTests(unittest.TestCase):
    def make_args(self, folder, *extra):
        return suite.parse_args([
            "--data-root", str(folder / "data"), "--v7-run-root", str(folder / "v7"),
            "--run-root", str(folder / "output"), "--device", "cpu", "--skip-robustness",
            "--models", "v15", "--nv", "1", *extra,
        ])

    def make_plan(self, folder, *extra):
        args = self.make_args(folder, *extra)
        dataset = suite.dataset_path(args, 1)
        init = suite.initialization_path(args, 1, 3407)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        init.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_bytes(b"dataset")
        init.write_bytes(b"trained-v7")
        plan = suite.prepare_job(args, args.run_root, "v15", 3407, 1, dataset, init,
                                 suite.file_digest(dataset), suite.file_digest(init),
                                 {"model.py": "source"}, {"device": "cpu"})
        return args, plan

    @staticmethod
    def fake_run(status="selected", baseline_bytes=b"reference-predictions"):
        def execute(command, **kwargs):
            def value(flag):
                return Path(command[command.index(flag) + 1])
            checkpoint = value("--checkpoint-dir")
            checkpoint.mkdir(parents=True, exist_ok=True)
            for name in ("baseline.pt", "model_last.pt", "training_state.pt"):
                (checkpoint / name).write_bytes(name.encode())
            (checkpoint / "loss_history.csv").write_text("epoch,loss\n1,0.1\n")
            selection = {"status": status, "complete": True,
                         "selected_checkpoint": {"epoch": 1} if status == "selected" else None}
            (checkpoint / "validation_selection.json").write_text(json.dumps(selection))
            value("--baseline-output-pkl").write_bytes(baseline_bytes)
            if status == "selected":
                (checkpoint / "best_model.pt").write_bytes(b"selected-weights")
                result = value("--output-pkl")
                result.parent.mkdir(parents=True, exist_ok=True)
                result.write_bytes(b"selected-predictions")
        return execute

    def test_eight_conditions_and_defaults(self):
        args = suite.parse_args([])
        self.assertEqual(suite.select_variants(args), tuple(suite.VARIANTS))
        self.assertEqual(args.nv, "all")
        self.assertEqual(args.seeds, [3407])
        values = [
            (0, 0, 0, False, "joint"), (.01, 0, 0, False, "joint"),
            (0, .01, 0, False, "joint"), (.01, .01, 0, False, "joint"),
            (.01, .01, .25, False, "joint"), (.01, .01, 0, True, "joint"),
            (.01, .01, .25, True, "joint"), (.01, .01, 0, False, "flow_only"),
        ]
        for name, expected in zip(suite.VARIANTS, values):
            config = suite.build_config(name, 3407, 2, Path("/tmp/data.pkl"))
            actual = (config.model.v13_raw_cos_weight, config.model.v12_tail_weight,
                      config.model.v13_extra_mse_weight, config.model.v12_local_gate,
                      config.training.finetune_scope)
            self.assertEqual(actual, expected)
            self.assertEqual(config.training.train_epochs, 10)
            self.assertEqual(config.training.scheduler, "none")
            self.assertEqual(config.training.early_stopping_patience_epochs, 0)
            self.assertEqual(config.model_name, "lstm_finetune_v15_ecg")
            self.assertEqual(config.training.lr_backbone, 1e-6)
            self.assertEqual(config.training.lr_decoder, 3e-6)
            self.assertEqual(config.training.lr_flow, 1e-5)

    def test_v7_inference_and_original_loss_config_retained(self):
        original = OmegaConf.load(PROJECT / "src/configs" / f"mecge_table1_repro_{suite.V7_MODEL}.yaml")
        candidate = suite.build_config("v15", 3407, 1, Path("/tmp/data.pkl"))
        for key, value in original.model.items():
            self.assertEqual(candidate.model[key], value, key)
        self.assertEqual(original.dataset.pkl_validation_ratio, candidate.dataset.pkl_validation_ratio)
        self.assertEqual(original.dataset.pkl_validation_random_state, candidate.dataset.pkl_validation_random_state)

    def test_named_disabled_factors_override_shared_sweeps(self):
        overrides = ["model.v13_raw_cos_weight=0.02", "model.v12_tail_weight=0.03",
                     "model.v13_extra_mse_weight=0.4", "training.lr_flow=0.00002"]
        for name, factors in suite.VARIANTS.items():
            config = suite.build_config(name, 9, 1, Path("data.pkl"), overrides, 7)
            for enabled, key, value in zip(factors[:3], ("v13_raw_cos_weight", "v12_tail_weight", "v13_extra_mse_weight"),
                                           (.02, .03, .4)):
                self.assertEqual(config.model[key], value if enabled else 0)
            self.assertEqual(config.training.train_epochs, 7)
            self.assertEqual(config.training.lr_flow, .00002)

    def test_unknown_ignored_and_nonfinite_overrides_rejected(self):
        bad = ["training.lr=0.1", "training.scheduler=ReduceLROnPlateau",
               "model.v12_local_gate=true", "model.cfm_inference_steps=2", "training.finetune_scope=flow_only",
               "model.v14_relative_weight=.01", "training.batch_size=1.5", "training.train_epochs=0",
               "training.lr_flow=-1", "model.v13_raw_cos_weight=-1", "training.lr_backbone=true",
               "training.lr_decoder=.inf", "model.v12_tail_weight=.nan"]
        for override in bad:
            with self.subTest(override=override), self.assertRaises(ValueError):
                suite.build_config("v15", 3407, 1, Path("data.pkl"), [override])

    def test_checkpoint_templates_and_explicit_pair_restriction(self):
        args = suite.parse_args(["--v7-checkpoint", "/weights/nv{nv}/seed{seed}.pt", "--seeds", "1", "2"])
        self.assertEqual(suite.initialization_path(args, 2, 1), Path("/weights/nv2/seed1.pt"))
        args = suite.parse_args(["--v7-checkpoint", "/weights/one.pt", "--nv", "1", "--seeds", "42"])
        self.assertEqual(suite.initialization_path(args, 1, 42), Path("/weights/one.pt"))
        default = suite.initialization_path(suite.parse_args([]), 2, 3407)
        self.assertIn(f"{suite.V7_MODEL}__qtdb_train_qtdb_test__nv2__seed3407", str(default))
        self.assertEqual(default.parent.name, suite.V7_REGISTERED)
        for argv in (["--v7-checkpoint", "/one.pt"],
                     ["--nv", "1", "--seeds", "1", "2", "--v7-checkpoint", "/one.pt"],
                     ["--v7-checkpoint", "/{nv.__class__}.pt"],
                     ["--pkl-file", "/one.pkl"],
                     ["--collect-only", "--resume"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                suite.parse_args(argv)

    def test_list_and_dry_run_need_no_input_files_or_output_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                suite.main(["--list"])
                suite.main(["--dry-run", "--run-root", str(folder / "out"),
                            "--data-root", str(folder / "absent-data"),
                            "--v7-run-root", str(folder / "absent-weights")])
            self.assertIn("8 variants x 2 noise versions x 1 seeds", out.getvalue())
            self.assertIn("--init-checkpoint", out.getvalue())
            self.assertFalse((folder / "out").exists())

    def test_preflight_checks_all_inputs_before_any_process(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            argv = ["--models", "v15", "--device", "cpu", "--skip-robustness",
                    "--data-root", str(folder / "data"), "--run-root", str(folder / "out"),
                    "--v7-run-root", str(folder / "weights")]
            args = suite.parse_args(argv)
            dataset = suite.dataset_path(args, 1)
            dataset.parent.mkdir(parents=True)
            dataset.write_bytes(b"nv1")
            init = suite.initialization_path(args, 1, 3407)
            init.parent.mkdir(parents=True)
            init.write_bytes(b"v7")
            with patch.object(suite.subprocess, "run") as run, self.assertRaises(FileNotFoundError) as error:
                suite.main(argv)
            self.assertIn("nv2", str(error.exception))
            run.assert_not_called()
            self.assertFalse((folder / "out").exists())

    def test_protected_roots_and_escaping_symlinks(self):
        for root in (PROJECT, PROJECT / "src", PROJECT / "runs/v14_performance/new", PROJECT / "runs"):
            with self.assertRaises(ValueError):
                suite.check_output_root(root)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "output").mkdir()
            (folder / "outside").mkdir()
            (folder / "output/v15").symlink_to(folder / "outside", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.make_plan(folder)
            with self.assertRaises(ValueError):
                suite.check_output_root(folder / "output", [folder / "output/input.pkl"])

    def test_selected_run_is_hashed_then_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.make_plan(Path(directory))
            original_init = plan["initialization"].read_bytes()
            self.assertEqual(suite.inspect_job(args, plan), "run")
            with patch.object(suite.subprocess, "run", side_effect=self.fake_run()):
                suite.run_job(args, plan)
            self.assertEqual(suite.inspect_job(args, plan), "skip")
            manifest = json.loads(plan["manifest_path"].read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["status"], "selected")
            self.assertIn(str(plan["shared_baseline"].relative_to(plan["root"])), manifest["artifacts_sha256"])
            self.assertEqual(plan["initialization"].read_bytes(), original_init)
            plan["result"].write_bytes(b"changed-result")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                suite.inspect_job(args, plan)

    def test_no_feasible_is_complete_without_best_or_main_result(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.make_plan(Path(directory))
            with patch.object(suite.subprocess, "run", side_effect=self.fake_run("no_feasible_checkpoint")):
                suite.run_job(args, plan)
            self.assertFalse(plan["result"].exists())
            self.assertFalse((plan["checkpoint"] / "best_model.pt").exists())
            self.assertEqual(suite.inspect_job(args, plan), "skip")
            args.eval_only = True
            self.assertEqual(suite.inspect_job(args, plan), "run")
            with patch.object(suite.subprocess, "run", side_effect=self.fake_run("no_feasible_checkpoint")) as run:
                suite.run_job(args, plan)
            self.assertIn("--skip-train", run.call_args.args[0])

    def test_initial_checkpoint_hash_change_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.make_plan(Path(directory), "--resume")
            with patch.object(suite.subprocess, "run", side_effect=self.fake_run()):
                suite.run_job(args, plan)
            plan["identity"]["init_checkpoint_sha256"] = "different"
            plan["baseline_identity"]["init_checkpoint_sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "changed"):
                suite.inspect_job(args, plan)

    def test_resume_requires_managed_identity_and_preserves_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.make_plan(Path(directory))
            plan["checkpoint"].mkdir(parents=True)
            (plan["checkpoint"] / "training_state.pt").write_bytes(b"state")
            suite.atomic_json(plan["manifest_path"], {"identity": plan["identity"], "complete": False})
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                suite.inspect_job(args, plan)
            args.resume = True
            self.assertEqual(suite.inspect_job(args, plan), "run")
            with patch.object(suite.subprocess, "run", side_effect=self.fake_run()) as run:
                suite.run_job(args, plan)
            self.assertIn("--resume", run.call_args.args[0])

    def test_shared_baseline_mismatch_preserves_first_export(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.make_plan(Path(directory))
            with patch.object(suite.subprocess, "run", side_effect=self.fake_run()):
                suite.run_job(args, plan)
            saved = plan["shared_baseline"].read_bytes()
            plan["baseline_result"].write_bytes(b"different-predictions")
            with self.assertRaisesRegex(RuntimeError, "differs across"):
                suite.publish_baseline(plan)
            self.assertEqual(plan["shared_baseline"].read_bytes(), saved)

    def test_input_mutated_after_preflight_does_not_start_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.make_plan(Path(directory))
            plan["initialization"].write_bytes(b"replaced-checkpoint")
            with patch.object(suite.subprocess, "run") as run, self.assertRaisesRegex(RuntimeError, "after preflight"):
                suite.run_job(args, plan)
            run.assert_not_called()
            self.assertFalse(plan["job"].exists())

    def test_full_launch_checks_all_manifests_before_work(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args, plan = self.make_plan(folder)
            bad = dict(plan)
            bad["variant"] = "v15_cos_only"
            bad["identity"] = {**plan["identity"], "variant": "v15_cos_only"}
            with patch.object(suite, "prepare_job", side_effect=[plan, bad]), \
                 patch.object(suite, "inspect_job", side_effect=["run", RuntimeError("changed identity")]), \
                 patch.object(suite, "source_digests", return_value={}), \
                 patch.object(suite, "environment_info", return_value={}), \
                 patch.object(suite.subprocess, "run") as run, \
                 self.assertRaisesRegex(RuntimeError, "changed identity"):
                suite.main(["--models", "v15", "v15_cos_only", "--nv", "1", "--device", "cpu", "--skip-robustness",
                            "--data-root", str(args.data_root), "--v7-run-root", str(args.v7_run_root),
                            "--run-root", str(args.run_root)])
            run.assert_not_called()
            self.assertFalse(plan["job"].exists())

    def test_collect_only_does_not_require_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            module = types.ModuleType("v15_results")
            calls = []
            module.collect_results = lambda *args, **kwargs: calls.append((args, kwargs))
            with patch.dict(sys.modules, {"v15_results": module}):
                suite.main(["--collect-only", "--skip-robustness", "--models", "v15", "--nv", "1",
                            "--data-root", str(folder / "missing-data"), "--run-root", str(folder / "results"),
                            "--v7-run-root", str(folder / "missing-weights")])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["expected_models"], ["lstm_v15"])


if __name__ == "__main__":
    unittest.main()
