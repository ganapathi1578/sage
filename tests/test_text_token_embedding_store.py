"""Self-contained synthetic test for TokenEmbeddingStore; run with python -m unittest."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.text_token_embedding_store import TokenEmbeddingStore


class TokenEmbeddingStoreTest(unittest.TestCase):
    def test_get_and_padded_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared"
            shared.mkdir()
            # Three sentences: lengths 2, 1, and 3; hidden dimension 4.
            tokens = np.arange(24, dtype=np.float16).reshape(6, 4)
            offsets = np.asarray([0, 2, 3, 6], dtype=np.int64)
            np.save(shared / "part-000000.tokens.npy", tokens)
            np.save(shared / "part-000000.offsets.npy", offsets)
            manifest = {
                "format": "ragged_token_embeddings_v1",
                "complete": True,
                "embedding_dim": 4,
                "storage_dtype": "float16",
                "num_sentences": 3,
                "embedding_store": "shared",
                "parts": [{
                    "part_index": 0, "start_id": 1, "end_id": 3, "rows": 3,
                    "token_count": 6, "embedding_dim": 4,
                    "token_file": "part-000000.tokens.npy",
                    "offset_file": "part-000000.offsets.npy",
                }],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with TokenEmbeddingStore(root) as store:
                self.assertEqual(store.get(1).shape, (2, 4))
                self.assertEqual(store.get(2).shape, (1, 4))
                self.assertEqual(store.get(3).shape, (3, 4))
                batch, mask, lengths = store.get_batch([1, 2, 3, 0])
                self.assertEqual(batch.shape, (4, 3, 4))
                self.assertEqual(mask.shape, (4, 3))
                self.assertEqual(lengths.tolist(), [2, 1, 3, 0])
                self.assertEqual(mask.sum(axis=1).tolist(), [2, 1, 3, 0])


if __name__ == "__main__":
    unittest.main()
