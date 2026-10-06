"""v13 runner regression checks, including the unchanged official metric definitions."""

import csv
import pickle
import random
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import mecge_table1_run_v13_model as runner


class TinyDenoiser(torch.nn.Module):
    """Exercise stochastic loss, scheduling and checkpointing without a U-Net."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.rand(()))
        self.epoch, self.epochs = 1, 3

    def set_epoch(self, epoch, total_epochs):
        self.epoch, self.epochs = epoch, total_epochs

    def tail_weight(self):
        return 0.01 * self.epoch / self.epochs if self.training else 0.01

    def denoising(self, noisy):
        return noisy * self.gain

    def compute_loss(self, batch, device):
        noisy, clean = [value.to(device) for value in batch]
        residual = self.denoising(noisy) - clean
        jitter = torch.rand((), device=device) + random.random() + np.random.random()
        return residual.square().mean() + 0.01 * jitter * self.gain + self.tail_weight() * residual.abs().mean()


def training_args(scheduler="none", resume=False):
    return Namespace(device=torch.device("cpu"), optimizer="AdamW", lr=0.01,
                     betas=[0.8, 0.99], weight_decay=0.0, scheduler=scheduler,
                     gamma=0.9, factor=0.5, lr_scheduler_patience_epochs=2,
                     lr_scheduler_min_delta=1e-4, min_lr=0.0, epochs=3,
                     patience=0, early_stopping_min_delta=0.0, grad_clip_norm=None,
                     resume=resume, resume_checkpoint=None)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class V13RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_rng_equal(self, expected, actual):
        self.assertEqual(expected["python"], actual["python"])
        self.assertEqual(expected["numpy"][0], actual["numpy"][0])
        np.testing.assert_array_equal(expected["numpy"][1], actual["numpy"][1])
        self.assertEqual(expected["numpy"][2:], actual["numpy"][2:])
        self.assertTrue(torch.equal(expected["torch_cpu"], actual["torch_cpu"]))
        self.assertEqual(len(expected["torch_cuda"]), len(actual["torch_cuda"]))
        for left, right in zip(expected["torch_cuda"], actual["torch_cuda"]):
            self.assertTrue(torch.equal(left, right))

    def test_metrics_include_remainder_and_do_not_change_rng_or_training_mode(self):
        noisy = torch.arange(5, dtype=torch.float32).view(5, 1, 1).repeat(1, 1, 4)
        dataset = TensorDataset(noisy, torch.zeros_like(noisy))
        original = DataLoader(dataset, batch_size=2, drop_last=True)
        complete = runner.full_validation_loader(original)
        model = TinyDenoiser().train()
        with torch.no_grad():
            model.gain.fill_(1)
        seed_all(42)
        before = runner.capture_rng_state()
        metrics = runner.deterministic_validation_metrics(model, complete, "cpu")
        self.assertEqual(metrics["val_ssd"], 24.0)
        self.assertEqual(metrics["val_mad"], 2.0)
        self.assertEqual(metrics["val_count"], 5)
        self.assertAlmostEqual(metrics["val_prd"], 80.0, places=6)
        self.assertEqual(metrics["val_cossim"], 0.0)
        self.assertTrue(model.training)
        self.assert_rng_equal(before, runner.capture_rng_state())
        with self.assertRaisesRegex(ValueError, "every window"):
            runner.deterministic_validation_metrics(model, original, "cpu")
        self.assert_rng_equal(before, runner.capture_rng_state())

    def test_metric_rng_isolation_covers_stochastic_forward_and_failure(self):
        class StochasticModel(TinyDenoiser):
            def denoising(self, noisy):
                return super().denoising(noisy) + torch.rand_like(noisy) + np.random.random() + random.random()

        model = StochasticModel().train()
        data = TensorDataset(torch.ones(5, 1, 4), torch.zeros(5, 1, 4))
        loader = DataLoader(data, batch_size=2)
        seed_all(56)
        before = runner.capture_rng_state()
        first = runner.deterministic_validation_metrics(model, loader, "cpu")
        self.assert_rng_equal(before, runner.capture_rng_state())
        second = runner.deterministic_validation_metrics(model, loader, "cpu")
        self.assertEqual(first, second)
        self.assert_rng_equal(before, runner.capture_rng_state())
        with patch.object(model, "denoising", side_effect=RuntimeError("forward failure")):
            with self.assertRaisesRegex(RuntimeError, "forward failure"):
                runner.deterministic_validation_metrics(model, loader, "cpu")
        self.assertTrue(model.training)
        self.assert_rng_equal(before, runner.capture_rng_state())

    def test_loader_split_and_batching_preserve_official_protocol(self):
        clean = np.arange(40, dtype=np.float32).reshape(10, 4, 1)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.pkl"
            with dataset.open("wb") as handle:
                pickle.dump([clean + 1, clean, clean[:3] + 1, clean[:3]], handle)
            train, val, test, noisy, target = runner.build_loaders(dataset, 2)
            self.assertEqual(len(train.dataset), 7)
            self.assertEqual(len(val.dataset), 3)
            self.assertTrue(train.drop_last)
            self.assertTrue(val.drop_last)
            self.assertEqual(test.batch_size, 50)
            self.assertFalse(test.drop_last)
            np.testing.assert_array_equal(target, clean[:3])
            np.testing.assert_array_equal(noisy, clean[:3] + 1)
            metric = runner.full_validation_loader(val)
            self.assertIs(metric.dataset, val.dataset)
            self.assertEqual(sum(len(batch[0]) for batch in metric), 3)

    def test_training_checkpoint_and_exact_resume_with_and_without_scheduler(self):
        noisy = torch.linspace(-1, 1, 28).view(7, 1, 4)
        train_loader = DataLoader(TensorDataset(noisy, noisy * 0.7), batch_size=2,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(noisy[:5], noisy[:5] * 0.7), batch_size=2,
                                drop_last=True)
        for scheduler_name in ("none", "ExponentialLR"):
            with self.subTest(scheduler=scheduler_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                whole, partial = root / "whole", root / "partial"
                seed_all(987)
                model = TinyDenoiser()
                expected_history = runner.train_model(model, train_loader, val_loader,
                                                      training_args(scheduler_name), whole)
                expected_parameters = {key: value.clone() for key, value in model.state_dict().items()}
                expected_rng = runner.capture_rng_state()
                seed_all(987)
                interrupted_model = TinyDenoiser()
                original_save = runner.save_training_state

                def save_then_interrupt(*args, **kwargs):
                    original_save(*args, **kwargs)
                    raise RuntimeError("simulated interruption after a complete epoch")

                with patch.object(runner, "save_training_state", side_effect=save_then_interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        runner.train_model(interrupted_model, train_loader, val_loader,
                                           training_args(scheduler_name), partial)
                seed_all(100)
                resumed_model = TinyDenoiser()
                actual_history = runner.train_model(resumed_model, train_loader, val_loader,
                                                    training_args(scheduler_name, resume=True), partial)
                self.assertEqual(actual_history, expected_history)
                for key, expected in expected_parameters.items():
                    self.assertTrue(torch.equal(resumed_model.state_dict()[key], expected))
                self.assert_rng_equal(expected_rng, runner.capture_rng_state())
                self.assertEqual((whole / "loss_history.csv").read_bytes(),
                                 (partial / "loss_history.csv").read_bytes())
                with (partial / "loss_history.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row["epoch"] for row in rows], ["1", "2", "3"])
                self.assertEqual([int(row["val_count"]) for row in rows], [5, 5, 5])
                self.assertEqual([float(row["val_topk_weight"]) for row in rows], [0.01] * 3)
                for name in ("best_model.pt", "best_val_ssd.pt", "model_last.pt"):
                    self.assertTrue((partial / name).is_file())
                state = torch.load(partial / "training_state.pt", weights_only=False)
                self.assertEqual(state["epoch"], 2)
                self.assertEqual(state["history"], expected_history)
                self.assertIsNone(state["scheduler_state_dict"]) if scheduler_name == "none" else self.assertIsNotNone(state["scheduler_state_dict"])
                self.assert_rng_equal(expected_rng, state["rng_state"])

    def test_selectors_export_their_own_checkpoint_predictions(self):
        noisy = torch.ones(5, 1, 4)
        clean = torch.zeros_like(noisy)
        loader = DataLoader(TensorDataset(noisy, clean), batch_size=2)
        original_noisy, original_clean = noisy.permute(0, 2, 1).numpy(), clean.permute(0, 2, 1).numpy()
        model = TinyDenoiser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, gain in (("best_model.pt", 0.0), ("best_val_ssd.pt", 1.0)):
                with torch.no_grad():
                    model.gain.fill_(gain)
                torch.save(model.state_dict(), root / name)
            loss_output, ssd_output = root / "loss.pkl", root / "ssd.pkl"
            before = runner.capture_rng_state()
            runner.test_model(model, loader, root, original_noisy, original_clean, loss_output, "cpu")
            runner.test_model(model, loader, root, original_noisy, original_clean, ssd_output, "cpu",
                              eval_checkpoint=root / "best_val_ssd.pt")
            self.assert_rng_equal(before, runner.capture_rng_state())
            with loss_output.open("rb") as handle:
                loss_payload = pickle.load(handle)
            with ssd_output.open("rb") as handle:
                ssd_payload = pickle.load(handle)
            np.testing.assert_array_equal(loss_payload[2], original_clean)
            np.testing.assert_array_equal(ssd_payload[2], original_noisy)
            np.testing.assert_array_equal(loss_payload[0], ssd_payload[0])
            np.testing.assert_array_equal(loss_payload[1], ssd_payload[1])
            self.assertEqual(loss_payload[2].shape, (5, 4, 1))

    def test_all_metrics_match_official_formulas_across_partial_batches(self):
        from utils_ecg import ssd, maximum_absolute_distance, prd_mecge_official, cosine_similarity
        clean = torch.tensor([[1., 2., 4., 3.], [10., 11., 7., 9.], [-2., 1., 3., 0.]])
        noisy = clean + torch.tensor([[.1, -.2, .3, .1], [.4, -.2, .1, -.3], [.2, .2, -.1, .4]])
        model = TinyDenoiser()
        with torch.no_grad():
            model.gain.fill_(.9)
        expected = {
            "val_ssd": ssd,
            "val_mad": maximum_absolute_distance,
            "val_prd": prd_mecge_official,
            "val_cossim": cosine_similarity,
        }
        for batch_size in (1, 2, 3):
            loader = DataLoader(TensorDataset(noisy[:, None], clean[:, None]), batch_size=batch_size)
            actual = runner.deterministic_validation_metrics(model, loader, "cpu")
            for name, formula in expected.items():
                value = formula(clean.numpy(), (noisy * .9).numpy()).mean()
                self.assertAlmostEqual(actual[name], float(value), delta=2e-5)
            self.assertEqual(actual["val_count"], 3)

    def test_empty_loss_loaders_fail_clearly(self):
        data = TensorDataset(torch.ones(1, 1, 4), torch.zeros(1, 1, 4))
        empty = DataLoader(data, batch_size=2, drop_last=True)
        nonempty = DataLoader(data, batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "training loader has no complete batches"):
                runner.train_model(TinyDenoiser(), empty, nonempty, training_args(), Path(directory))
            with self.assertRaisesRegex(ValueError, "validation loss loader has no complete batches"):
                runner.train_model(TinyDenoiser(), nonempty, empty, training_args(), Path(directory))


if __name__ == "__main__":
    unittest.main()
