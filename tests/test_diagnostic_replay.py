"""Check the released diagnostic replay against actual model forwards on CPU."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "small"))
sys.path.insert(0, str(ROOT / "scripts"))
from pretrain_core import make_model
from analyze_checkpoints import merge_batch_results, run_batch


class DiagnosticReplay(unittest.TestCase):
    def test_replay_matches_native(self):
        torch.set_num_threads(2)
        for variant in ("full", "block4", "norm_rms"):
            torch.manual_seed(42)
            model = make_model(ROOT / "vendor/brujula", width=64, layers=4, heads=4, seq_len=8, vocab=257,
                               residual="attnres_full" if variant == "full" else "attnres_block",
                               summary_mode="norm_rms" if variant == "norm_rms" else None, checkpointing=False)
            ids = torch.randint(0, 257, (2, 8))
            targets = torch.randint(0, 257, (2, 8))
            targets[1, 5:] = -100
            result = run_batch(model.native.eval(), ids, targets, "synthetic", variant, [.25, .5, 2., 4.])
            self.assertEqual(result["checks"]["max_aggregate_formula_error"], 0.)
            self.assertEqual(result["checks"]["manual_backbone_vs_native"]["max_abs"], 0.)
            self.assertEqual(len(result["output_rms"]), 8)
            self.assertTrue(result["counterfactual"])
        with self.assertRaises(ValueError):
            merge_batch_results([{}, {}])

    def test_released_parameter_counts(self):
        # Meta tensors instantiate the full architecture without allocating weights.
        with torch.device("meta"):
            small = make_model(ROOT / "vendor/brujula", residual="attnres_full", checkpointing=False)
            large = make_model(ROOT / "vendor/brujula", width=816, layers=18, heads=6, seq_len=1024,
                               residual="attnres_full", publisher_config=True, checkpointing=False)
            config_450m = make_model(
                ROOT / "vendor/brujula", width=1280, layers=28, heads=8,
                seq_len=1024, kv_compression_dim=112, q_compression_dim=320,
                residual="attnres_full", checkpointing=False,
            )
        self.assertEqual(sum(p.numel() for p in small.parameters()), 56836352)
        self.assertEqual(sum(p.numel() for p in large.parameters()), 157433856)
        self.assertEqual(sum(p.numel() for p in config_450m.parameters()), 512416576)


if __name__ == "__main__":
    unittest.main()
