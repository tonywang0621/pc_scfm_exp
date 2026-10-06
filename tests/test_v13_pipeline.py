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
import v13_experiments as pipeline


class V13PipelineTests(unittest.TestCase):
    def run_cli(self, arguments, success=True):
        result = subprocess.run(
            [sys.executable, "-B", str(pipeline.APP / "v13_experiments.py"), *arguments],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_only_six_v13_variants_and_no_v7_training_option(self):
        self.assertEqual(len(pipeline.CORE), 6)
        self.assertEqual(len(pipeline.VARIANTS), 6)
        self.assertEqual(pipeline.select_variants(pipeline.parse_args([])), pipeline.CORE)
        for suite in ("core", "all"):
            self.assertEqual(pipeline.select_variants(pipeline.parse_args(["--suite", suite])), pipeline.CORE)
        self.assertNotIn("v7_reference", pipeline.VARIANTS)
        self.assertTrue(all(name.startswith("v13") for name in pipeline.VARIANTS))
        self.assertEqual(pipeline.select_variants(pipeline.parse_args(
            ["--models", "v13", "v13_no_topk", "v13"])), ("v13", "v13_no_topk"))
        defaults = pipeline.parse_args([])
        self.assertEqual(defaults.seeds, [3407])
        self.assertEqual(defaults.nv, "all")
        self.assertEqual(defaults.run_root, PROJECT / "runs/v13_performance")

    def test_ablation_configs_change_only_the_declared_factors(self):
        dataset = Path("unused.pkl")
        base = pipeline.build_config("v13", 3407, 1, dataset)
        self.assertEqual(base.model_name, "lstm_metric_balanced_v13_ecg")
        self.assertEqual(base.training.train_epochs, 30)
        self.assertEqual(base.training.early_stopping_patience_epochs, 0)
        self.assertEqual(base.training.selection_metric, "val_loss")
        self.assertEqual(base.model.v12_tail_weight, 0.01)
        self.assertEqual(base.model.v13_extra_mse_weight, 0.25)
        self.assertEqual(base.model.v13_raw_cos_weight, 0.01)
        self.assertEqual(base.model.v13_ramp_epochs, 10)
        for variant, (changes, _) in pipeline.VARIANTS.items():
            config = pipeline.build_config(variant, 3407, 2, dataset)
            expected = OmegaConf.create(OmegaConf.to_container(base.model))
            for key, value in changes.items():
                OmegaConf.update(expected, key.removeprefix("model."), value)
            self.assertEqual(OmegaConf.to_container(config.model), OmegaConf.to_container(expected))
            self.assertEqual(config.dataset.pkl_file, str(dataset))
            self.assertIn("nv2__seed3407", config.exp_name)
        # Each config is loaded independently; overrides cannot leak into the next job.
        pipeline.build_config("v13", 9, 1, dataset, ["model.lambda_mse=4.0"], epochs=2)
        self.assertEqual(pipeline.build_config("v13", 3407, 1, dataset).model.lambda_mse, 0.95)
        original = OmegaConf.load(pipeline.APP / "configs" /
            "mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml")
        inherited = {key: value for key, value in OmegaConf.to_container(base.model).items()
                     if not key.startswith(("v12_", "v13_")) and key != "mad_topk"}
        self.assertEqual(inherited, OmegaConf.to_container(original.model))

    def test_invalid_objectives_fail_before_launch(self):
        invalid = [
            ["model.v13_extra_mse_weight=-1"], ["model.v13_raw_cos_weight=nan"],
            ["model.v12_tail_weight=inf"], ["model.v13_ramp_epochs=1.5"],
            ["model.v12_tail_ramp_epochs=0"], ["model.mad_topk=0"],
            ["model.loss_fn=time+mse+max"], ["model.loss_fn=time+mse+cos"],
            ["training.selection_metric=test_mad"],
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                pipeline.build_config("v13", 3407, 1, Path("unused.pkl"), overrides)
        with self.assertRaisesRegex(ValueError, "Original v7 is excluded"):
            pipeline.build_config("v13", 3407, 1, Path("unused.pkl"), [
                "model.v12_local_gate=false", "model.v12_tail_weight=0",
                "model.v13_extra_mse_weight=0", "model.v13_raw_cos_weight=0",
            ])

    def test_shared_sweeps_cannot_reenable_named_ablations(self):
        overrides = ["model.v13_extra_mse_weight=0.5", "model.v13_raw_cos_weight=0.02",
                     "model.v12_tail_weight=0.03", "model.v12_local_gate=true",
                     "training.lr=0.0001", "model.cfm_train_noise_scale=0.06"]
        full = pipeline.build_config("v13", 3407, 1, Path("unused.pkl"), overrides)
        self.assertEqual(full.model.v13_extra_mse_weight, 0.5)
        self.assertEqual(full.model.v13_raw_cos_weight, 0.02)
        self.assertEqual(full.model.v12_tail_weight, 0.03)
        self.assertTrue(full.model.v12_local_gate)
        for variant, (removals, _) in pipeline.VARIANTS.items():
            with self.subTest(variant=variant):
                config = pipeline.build_config(variant, 3407, 1, Path("unused.pkl"), overrides)
                self.assertEqual(config.training.lr, 0.0001)
                self.assertEqual(config.model.cfm_train_noise_scale, 0.06)
                expected = OmegaConf.create(OmegaConf.to_container(full.model))
                for key, value in removals.items():
                    OmegaConf.update(expected, key.removeprefix("model."), value)
                self.assertEqual(OmegaConf.to_container(config.model), OmegaConf.to_container(expected))

    def test_protected_roots_and_symlink_are_rejected(self):
        for name in (".", "runs", "runs/mecge_table1_repro", "runs/v11_performance",
                     "runs/v12_performance/nv1", "data/new", "src", "scripts", "tests", "docs"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pipeline.check_output_root(PROJECT / name)
        self.assertEqual(pipeline.check_output_root(PROJECT / "runs/v13_performance"),
                         PROJECT / "runs/v13_performance")
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "legacy-alias"
            alias.symlink_to(PROJECT / "runs/v12_performance", target_is_directory=True)
            with self.assertRaises(ValueError):
                pipeline.check_output_root(alias / "new-output")

    def test_dry_run_is_read_only_and_previews_each_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            result = self.run_cli(["--dry-run", "--run-root", str(output)])
            self.assertIn("6 variants x 2 noise versions x 1 seeds", result.stdout)
            self.assertEqual(result.stdout.count("--ssd-output-pkl"), 12)
            self.assertEqual(result.stdout.count("mecge_table1_run_v13_model.py"), 12)
            self.assertNotIn("lstm_v7_reference__", result.stdout)
            self.assertFalse(output.exists())
            for forbidden in (["--suite", "reference"], ["--models", "v7_reference"],
                              ["--models", "lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience"]):
                self.run_cli(["--dry-run", "--run-root", str(output), *forbidden], success=False)
            self.assertFalse(output.exists())
            self.run_cli(["--models", "v13", "--suite", "core", "--dry-run"], success=False)

    def test_sources_include_dependencies_without_changing_legacy_inventory(self):
        from v11_experiments import source_digests as v11_digests
        from v12_experiments import source_digests as v12_digests
        sources = pipeline.source_digests()
        for name in ("src/models/lstm_v13/__init__.py", "src/models/lstm_v12/__init__.py",
                     "src/mecge_table1_run_v13_model.py", "src/v13_experiments.py",
                     "src/v12_results.py", "src/utils_ecg.py",
                     "src/configs/mecge_table1_repro_lstm_v13.yaml"):
            self.assertIn(name, sources)
            self.assertEqual(len(sources[name]), 64)
        for legacy in (v11_digests(), v12_digests()):
            self.assertFalse(any("v13" in name for name in legacy))

    def test_manifest_skip_identity_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            args = pipeline.parse_args(["--models", "v13", "--nv", "1", "--run-root", str(root)])
            job = root / "v13/nv1/seed3407"
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
                pipeline.run_job(args, root, "v13", 3407, 1, dataset, "data-hash", {"source": "hash"}, {})

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


if __name__ == "__main__":
    unittest.main()
