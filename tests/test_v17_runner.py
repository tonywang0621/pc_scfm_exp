import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mecge_table1_run_v17_model import make_model, train_model, train_settings, validate
from v17_experiments import build_config
from test_v17_model import SMALL


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_resume_matches_uninterrupted_weights_and_selection(self):
        config = build_config("v17", 3407, 1, Path("unused"), SMALL, epochs=2)
        config.training.batch_size = 2
        clean = torch.sin(torch.linspace(0, 8*torch.pi, 512)).view(1, 1, -1).repeat(7, 1, 1)
        noisy = clean + torch.linspace(-.2, .2, 512).view(1, 1, -1)
        train = DataLoader(TensorDataset(noisy[:4], clean[:4]), batch_size=2)
        val = DataLoader(TensorDataset(noisy[4:], clean[4:]), batch_size=2, drop_last=True)
        seed = lambda: (random.seed(3407), np.random.seed(3407), torch.manual_seed(3407))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed()
            full = make_model(config)
            settings = train_settings(config, "cpu", 3407)
            expected = train_model(full, train, val, settings, root / "full")
            seed()
            partial = make_model(config)
            train_model(partial, train, val, settings, root / "resume", stop_after=1)
            resumed = make_model(config)
            settings.resume = True
            actual = train_model(resumed, train, val, settings, root / "resume")
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            for a, b in zip(expected, actual):
                self.assertEqual({k: v for k, v in a.items() if k != "seconds"},
                                 {k: v for k, v in b.items() if k != "seconds"})
            self.assertEqual((root / "full/selection.json").read_bytes(), (root / "resume/selection.json").read_bytes())

    def test_validation_is_complete_and_does_not_advance_rng(self):
        cfg = build_config("v17", 2, 1, Path("unused"), SMALL)
        model = make_model(cfg).train()
        clean = torch.randn(3, 1, 512)
        loader = DataLoader(TensorDataset(clean+.2, clean), batch_size=2, drop_last=True)
        state = torch.get_rng_state().clone()
        metrics, waves = validate(model, loader, "cpu")
        self.assertEqual(metrics["val_count"], 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertIn("gate_flow_saturated_fraction", metrics)
        self.assertIn("baseline_candidate", waves)

    def test_training_rng_is_independent_of_model_construction_draws(self):
        cfg = build_config("v17", 71, 1, Path("unused"), SMALL, epochs=1)
        clean = torch.randn(4, 1, 512)*.3
        loader = DataLoader(TensorDataset(clean+.1, clean), batch_size=2)
        a, b = make_model(cfg), make_model(cfg)
        b.load_state_dict(a.state_dict())
        settings = train_settings(cfg, "cpu", 71)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(33)
            train_model(a, loader, loader, settings, root / "a")
            torch.manual_seed(123456)
            train_model(b, loader, loader, settings, root / "b")
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
