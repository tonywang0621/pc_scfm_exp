import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import v14_experiments as pipeline


class V14PipelineTests(unittest.TestCase):
    def run_cli(self, arguments, success=True):
        result = subprocess.run(
            [sys.executable, "-B", str(pipeline.APP / "v14_experiments.py"), *arguments],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_only_seven_v14_variants_and_no_v7_training_option(self):
        self.assertEqual(len(pipeline.CORE), 7)
        self.assertEqual(len(pipeline.VARIANTS), 7)
        self.assertEqual(pipeline.select_variants(pipeline.parse_args([])), pipeline.CORE)
        for suite in ("core", "all"):
            self.assertEqual(pipeline.select_variants(pipeline.parse_args(["--suite", suite])), pipeline.CORE)
        self.assertNotIn("v7_reference", pipeline.VARIANTS)
        self.assertTrue(all(name.startswith("v14") for name in pipeline.VARIANTS))
        self.assertEqual(pipeline.select_variants(pipeline.parse_args(
            ["--models", "v14", "v14_no_topk", "v14"])), ("v14", "v14_no_topk"))
        defaults = pipeline.parse_args([])
        self.assertEqual(defaults.seeds, [3407])
        self.assertEqual(defaults.nv, "all")
        self.assertEqual(defaults.run_root, PROJECT / "runs/v14_performance")

    def test_ablation_configs_change_only_the_declared_factors(self):
        dataset = Path("unused.pkl")
        base = pipeline.build_config("v14", 3407, 1, dataset)
        self.assertEqual(base.model_name, "lstm_relative_balanced_v14_ecg")
        self.assertEqual(base.training.train_epochs, 30)
        self.assertEqual(base.training.early_stopping_patience_epochs, 0)
        self.assertEqual(base.training.selection_metric, "val_score")
        self.assertEqual(base.training.val_score_cos_weight, 0.05)
        self.assertEqual(base.training.val_score_topk_weight, 0.01)
        self.assertEqual(base.training.val_score_topk, 8)
        self.assertFalse(base.model.v12_local_gate)
        self.assertEqual(base.model.v12_tail_weight, 0.01)
        self.assertEqual(base.model.v13_extra_mse_weight, 0.25)
        self.assertEqual(base.model.v13_raw_cos_weight, 0.05)
        self.assertEqual(base.model.v13_ramp_epochs, 10)
        self.assertEqual(base.model.v14_relative_weight, 0.02)
        self.assertEqual(base.model.v14_energy_floor, 0.01)
        self.assertEqual(base.model.v14_relative_ramp_epochs, 10)
        self.assertTrue(base.model.v14_curriculum)
        for variant, (changes, _) in pipeline.VARIANTS.items():
            config = pipeline.build_config(variant, 3407, 2, dataset)
            expected = OmegaConf.create(OmegaConf.to_container(base.model))
            for key, value in changes.items():
                OmegaConf.update(expected, key.removeprefix("model."), value)
            self.assertEqual(OmegaConf.to_container(config.model), OmegaConf.to_container(expected))
            self.assertEqual(OmegaConf.to_container(config.training), OmegaConf.to_container(base.training))
            self.assertEqual(config.dataset.pkl_file, str(dataset))
            self.assertIn("nv2__seed3407", config.exp_name)
        # Each config is loaded independently; overrides cannot leak into the next job.
        pipeline.build_config("v14", 9, 1, dataset, ["model.lambda_mse=4.0"], epochs=2)
        self.assertEqual(pipeline.build_config("v14", 3407, 1, dataset).model.lambda_mse, 0.95)
        original = OmegaConf.load(pipeline.APP / "configs" /
            "mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml")
        inherited = {key: value for key, value in OmegaConf.to_container(base.model).items()
                     if not key.startswith(("v12_", "v13_", "v14_")) and key != "mad_topk"}
        self.assertEqual(inherited, OmegaConf.to_container(original.model))
        control = pipeline.build_config("v14_v13_control", 3407, 1, dataset)
        historical = OmegaConf.load(pipeline.APP / "configs" / "mecge_table1_repro_lstm_v13.yaml")
        historical.model.v12_local_gate = False
        inherited_control = {key: value for key, value in OmegaConf.to_container(control.model).items()
                             if not key.startswith("v14_")}
        self.assertEqual(inherited_control, OmegaConf.to_container(historical.model))
        self.assertEqual(control.model.v14_relative_weight, 0)
        self.assertTrue(control.model.v14_curriculum)

    def test_invalid_objectives_fail_before_launch(self):
        invalid = [
            ["model.v13_extra_mse_weight=-1"], ["model.v13_raw_cos_weight=nan"],
            ["model.v12_tail_weight=inf"], ["model.v13_ramp_epochs=1.5"],
            ["model.v12_tail_ramp_epochs=0"], ["model.mad_topk=0"],
            ["model.loss_fn=time+mse+max"], ["model.loss_fn=time+mse+cos"],
            ["training.selection_metric=test_mad"],
            ["training.val_score_cos_weight=0.01"], ["training.val_score_topk_weight=0"],
            ["training.val_score_topk=1"], ["model.v14_relative_weight=-0.01"],
            ["model.v14_relative_weight=nan"], ["model.v14_energy_floor=0"],
            ["model.v14_energy_floor=inf"], ["model.v14_relative_ramp_epochs=1.5"],
            ["model.v14_curriculum=1"], ["model.v12_local_gate=true"],
            ["model.v12_feature_dapp=false"], ["model.v12_unet_dapp=false"],
            ["model.v12_resconv=false"], ["model.v12_dual_head=false"],
            ["model.use_dapp=false"], ["model.use_time_attention=false"],
            ["model.use_freq_attention=false"], ["model.cfm_inference_steps=2"],
            ["model.cfm_unet_use_attention=true"], ["model.attention_replacement=none"],
            ["training.init_checkpoint=previous/best_model.pt"], ["training.resume=true"],
            ["training.resume_checkpoint=previous/training_state.pt"],
            ["model.cfm_inference_start_noise_scale=0.1"], ["model.cfm_inference_shots=2"],
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                pipeline.build_config("v14", 3407, 1, Path("unused.pkl"), overrides)
        with self.assertRaisesRegex(ValueError, "Original v7 is excluded"):
            pipeline.build_config("v14", 3407, 1, Path("unused.pkl"), [
                "model.v12_local_gate=false", "model.v12_tail_weight=0",
                "model.v13_extra_mse_weight=0", "model.v13_raw_cos_weight=0",
                "model.v14_relative_weight=0",
            ])

    def test_shared_sweeps_cannot_reenable_named_ablations(self):
        overrides = ["model.v13_extra_mse_weight=0.5", "model.v13_raw_cos_weight=0.02",
                     "model.v12_tail_weight=0.03", "model.v14_relative_weight=0.04",
                     "model.v14_curriculum=true",
                     "training.lr=0.0001", "model.cfm_train_noise_scale=0.06"]
        full = pipeline.build_config("v14", 3407, 1, Path("unused.pkl"), overrides)
        self.assertEqual(full.model.v13_extra_mse_weight, 0.5)
        self.assertEqual(full.model.v13_raw_cos_weight, 0.02)
        self.assertEqual(full.model.v12_tail_weight, 0.03)
        self.assertFalse(full.model.v12_local_gate)
        self.assertEqual(full.model.v14_relative_weight, 0.04)
        for variant, (removals, _) in pipeline.VARIANTS.items():
            with self.subTest(variant=variant):
                config = pipeline.build_config(variant, 3407, 1, Path("unused.pkl"), overrides)
                self.assertEqual(config.training.lr, 0.0001)
                self.assertEqual(config.model.cfm_train_noise_scale, 0.06)
                expected = OmegaConf.create(OmegaConf.to_container(full.model))
                for key, value in removals.items():
                    OmegaConf.update(expected, key.removeprefix("model."), value)
                self.assertEqual(OmegaConf.to_container(config.model), OmegaConf.to_container(expected))

    def test_unknown_or_ignored_overrides_cannot_enter_run_identity(self):
        invalid = [
            "model.v14_relative_weigth=1", "training.batch_szie=2", "model.unknown=1",
            "training.unknown=1", "training.lr.typo=0.1", "dataset.pkl_file=other.pkl",
            "training.eval_every_epochs=2", "training.validation_metrics_every_epochs=2",
            "training.save_every_epochs=1", "training.progress_bar=false", "training.progress_bar=1",
        ]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                pipeline.build_config("v14", 3407, 1, Path("unused.pkl"), [override])
        config = pipeline.build_config("v14", 3407, 1, Path("unused.pkl"), [
            "training.gamma=0.98", "training.scheduler=ExponentialLR",
            "model.cfm_inference_start_noise_scale=0", "model.cfm_inference_shots=1",
            "training.eval_every_epochs=1", "training.validation_metrics_every_epochs=1",
            "training.save_every_epochs=5", "training.progress_bar=true",
            "training.init_checkpoint=null", "training.resume=false", "training.resume_checkpoint=null",
        ])
        self.assertEqual(config.training.gamma, 0.98)
        self.assertEqual(config.model.cfm_inference_start_noise_scale, 0)
        self.assertEqual(config.model.cfm_inference_shots, 1)

    def test_epochs_and_batch_size_reject_nonpositive_fractional_and_boolean_values(self):
        for key in ("train_epochs", "batch_size"):
            for value in ("0", "-1", "1.5", "true", "false", "nan", ".nan", ".inf", "'2'"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    pipeline.build_config("v14", 3407, 1, Path("unused.pkl"), [f"training.{key}={value}"])
        for epochs in (True, 1.5, float("nan"), float("inf")):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                pipeline.build_config("v14", 3407, 1, Path("unused.pkl"), epochs=epochs)
        config = pipeline.build_config("v14", 3407, 1, Path("unused.pkl"), ["training.batch_size=2"], epochs=1)
        self.assertEqual(config.training.train_epochs, 1)
        self.assertEqual(config.training.batch_size, 2)

    def test_protected_roots_and_symlink_are_rejected(self):
        for name in (".", "runs", "runs/mecge_table1_repro", "runs/v11_performance",
                     "runs/v12_performance/nv1", "runs/v13_performance", "runs/v13_performance/nv1",
                     "data/new", "src", "scripts", "tests", "docs"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pipeline.check_output_root(PROJECT / name)
        self.assertEqual(pipeline.check_output_root(PROJECT / "runs/v14_performance"),
                         PROJECT / "runs/v14_performance")
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "legacy-alias"
            alias.symlink_to(PROJECT / "runs/v12_performance", target_is_directory=True)
            with self.assertRaises(ValueError):
                pipeline.check_output_root(alias / "new-output")

    def test_dry_run_is_read_only_and_previews_each_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            result = self.run_cli(["--dry-run", "--run-root", str(output)])
            self.assertIn("7 variants x 2 noise versions x 1 seeds", result.stdout)
            self.assertEqual(result.stdout.count("--ssd-output-pkl"), 14)
            self.assertEqual(result.stdout.count("mecge_table1_run_v14_model.py"), 14)
            self.assertNotIn("lstm_v7_reference__", result.stdout)
            self.assertFalse(output.exists())
            for forbidden in (["--suite", "reference"], ["--models", "v7_reference"],
                              ["--models", "lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience"]):
                self.run_cli(["--dry-run", "--run-root", str(output), *forbidden], success=False)
            self.assertFalse(output.exists())
            self.run_cli(["--models", "v14", "--suite", "core", "--dry-run"], success=False)

    def test_sources_include_dependencies_without_changing_legacy_inventory(self):
        from v11_experiments import source_digests as v11_digests
        from v12_experiments import source_digests as v12_digests
        from v13_experiments import source_digests as v13_digests
        sources = pipeline.source_digests()
        for name in ("src/models/lstm_v14/__init__.py", "src/models/lstm_v12/__init__.py",
                     "src/models/lstm_v13/__init__.py", "src/mecge_table1_run_v13_model.py",
                     "src/mecge_table1_run_v14_model.py", "src/v14_experiments.py",
                     "src/v12_results.py", "src/v14_results.py", "src/v14_acceptance.py",
                     "src/v13_acceptance.py", "src/utils_ecg.py",
                     "src/configs/mecge_table1_repro_lstm_v14.yaml"):
            self.assertIn(name, sources)
            self.assertEqual(len(sources[name]), 64)
        for legacy in (v11_digests(), v12_digests(), v13_digests()):
            self.assertFalse(any("v14" in name for name in legacy))

    def test_cli_resume_and_collect_modes_are_unambiguous(self):
        for conflicting in (("--collect-only", "--resume"), ("--collect-only", "--dry-run"),
                            ("--collect-only", "--eval-only"), ("--resume", "--eval-only")):
            with self.subTest(args=conflicting), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pipeline.parse_args(conflicting)
        for resume, eval_only, expected in ((True, False, "--resume"), (False, True, "--skip-train")):
            command = pipeline.job_command("config.yaml", "dataset.pkl", "checkpoints", "log", "primary.pkl",
                                           "secondary.pkl", 3407, "cuda:0", resume, eval_only)
            self.assertIn(expected, command)
            self.assertTrue(command[2].endswith("mecge_table1_run_v14_model.py"))

    def test_collect_only_uses_v14_reporting_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            with patch("v14_results.collect_results") as collect, patch.object(pipeline.subprocess, "run") as train:
                pipeline.main(["--collect-only", "--skip-robustness", "--run-root", str(root)])
                collect.assert_called_once_with(
                    root, {}, baseline_csvs=None, baseline_robustness_csvs=None,
                    expected_models=[f"lstm_{name}" for name in pipeline.CORE], expected_seeds=[3407],
                )
                train.assert_not_called()

    def test_manifest_skip_identity_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            args = pipeline.parse_args(["--models", "v14", "--nv", "1", "--run-root", str(root)])
            job = root / "v14/nv1/seed3407"
            dataset = Path(directory) / "dataset.pkl"
            launched = []

            def fake_train(command, **kwargs):
                launched.append(command)
                for key in ("--output-pkl", "--ssd-output-pkl"):
                    result = Path(command[command.index(key) + 1])
                    result.parent.mkdir(parents=True, exist_ok=True)
                    result.write_bytes(key.encode())
                checkpoint = Path(command[command.index("--checkpoint-dir") + 1])
                checkpoint.mkdir(parents=True, exist_ok=True)
                for name in ("best_model.pt", "best_val_ssd.pt"):
                    (checkpoint / name).write_bytes(name.encode())

            def run():
                pipeline.run_job(args, root, "v14", 3407, 1, dataset, "data-hash", {"source": "hash"}, {})

            with patch.object(pipeline.subprocess, "run", side_effect=fake_train), contextlib.redirect_stdout(io.StringIO()):
                run()
                self.assertTrue(json.loads((job / "manifest.json").read_text())["complete"])
                run()
                self.assertEqual(len(launched), 1)
                args.set = ["training.lr=0.0001"]
                with self.assertRaisesRegex(RuntimeError, "Choose a new --run-root"):
                    run()
                args.set = []
                (job / "checkpoint/best_model.pt").write_bytes(b"changed")
                args.eval_only = True
                with self.assertRaisesRegex(RuntimeError, "Completed checkpoint changed"):
                    run()
                self.assertEqual(len(launched), 1)

    def test_interrupted_job_requires_resume_and_reuses_saved_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            args = pipeline.parse_args(["--models", "v14", "--nv", "1", "--run-root", str(root)])
            job = root / "v14/nv1/seed3407"
            dataset = Path(directory) / "dataset.pkl"

            def interrupted(command, **kwargs):
                checkpoint = Path(command[command.index("--checkpoint-dir") + 1])
                checkpoint.mkdir(parents=True, exist_ok=True)
                (checkpoint / "training_state.pt").write_bytes(b"saved state")
                raise subprocess.CalledProcessError(1, command)

            def resumed(command, **kwargs):
                self.assertIn("--resume", command)
                checkpoint = Path(command[command.index("--checkpoint-dir") + 1])
                self.assertEqual((checkpoint / "training_state.pt").read_bytes(), b"saved state")
                for name in ("best_model.pt", "best_val_ssd.pt"):
                    (checkpoint / name).write_bytes(name.encode())
                for key in ("--output-pkl", "--ssd-output-pkl"):
                    result = Path(command[command.index(key) + 1])
                    result.parent.mkdir(parents=True, exist_ok=True)
                    result.write_bytes(key.encode())

            def run():
                pipeline.run_job(args, root, "v14", 3407, 1, dataset, "data-hash", {"source": "hash"}, {})

            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(pipeline.subprocess, "run", side_effect=interrupted), self.assertRaises(subprocess.CalledProcessError):
                    run()
                self.assertFalse(json.loads((job / "manifest.json").read_text())["complete"])
                with patch.object(pipeline.subprocess, "run") as train, self.assertRaisesRegex(RuntimeError, "Use --resume"):
                    run()
                train.assert_not_called()
                args.resume = True
                with patch.object(pipeline.subprocess, "run", side_effect=resumed):
                    run()
                self.assertTrue(json.loads((job / "manifest.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
