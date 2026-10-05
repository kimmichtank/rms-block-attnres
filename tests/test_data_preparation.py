"""Synthetic local Parquet/tokenizer test; no network or research data required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "small"))
from build_training_index import build
from common import digest
from prepare_pretrain import doc_hash, split_for
from pretrain_core import DocumentChunks


class DataPreparation(unittest.TestCase):
    def test_prepare_resume_deduplicate_and_index(self):
        with tempfile.TemporaryDirectory(prefix="rms-data-test-") as temp:
            p = Path(temp)
            raw = p / "raw" / "sample/100BT"
            raw.mkdir(parents=True)
            tok = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "sample": 2, "word": 3}, unk_token="[UNK]"))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
            fast.save_pretrained(p / "tokenizer")
            chosen = {s: [] for s in ("train", "validation", "test")}
            for i in range(20000):
                text = f"sample {i} word"
                target = chosen[split_for(doc_hash(text))]
                if len(target) < 14:
                    target.append(text)
                if all(len(v) == 14 for v in chosen.values()):
                    break
            self.assertTrue(all(len(v) == 14 for v in chosen.values()))
            excluded = chosen["train"][0]
            rows = sum(chosen.values(), [])
            rows += [rows[2], "  " + rows[2].replace(" ", "   ") + "  ", ""]
            pq.write_table(pa.table({"text": rows}), raw / "000_00000.parquet", row_group_size=12)
            exclusion = p / "exclusions.json"
            exclusion.write_text(json.dumps({"normalization": "nfkc-whitespace-sha256-v1", "hashes": [doc_hash(excluded)]}))

            for script, output, extra in [
                ("prepare_pretrain.py", "bounded", ["--local-only", "--max-files", "1", "--train-tokens", "32", "--heldout-tokens", "16"]),
                ("prepare_full_corpus.py", "full", ["--expected-files", "1"]),
            ]:
                cmd = [sys.executable, str(ROOT / "small" / script), "--data", str(p / "raw"),
                       "--tokenizer", str(p / "tokenizer"), "--out", str(p / output),
                       "--exclusion-hashes", str(exclusion)] + extra
                subprocess.run(cmd + ["--max-groups", "1"], check=True, capture_output=True, text=True)
                self.assertFalse((p / output / "COMPLETE.json").exists())
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                self.assertTrue((p / output / "COMPLETE.json").exists())
                hashes = []
                for docs in (p / output).glob("part-*/docs.jsonl"):
                    hashes.extend(json.loads(line)["hash"] for line in docs.read_text().splitlines())
                self.assertEqual(len(hashes), len(set(hashes)))
                self.assertNotIn(doc_hash(excluded), hashes)

            # Exactly the same prefix selection must yield the same binary/index bytes.
            for output in ("index1", "index2"):
                build(p / "full", p / output, train_tokens=20, eval_tokens=12, seq_len=3, require_full=False)
                ds = DocumentChunks(p / output, "train", 3, 20)
                self.assertEqual(ds.tokens, 20)
            for split in ("train", "validation", "test"):
                for suffix in (".bin", ".npy"):
                    self.assertEqual(digest(p / "index1" / (split + suffix)), digest(p / "index2" / (split + suffix)))


if __name__ == "__main__":
    unittest.main()
