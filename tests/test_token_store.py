import json
from pathlib import Path

import numpy as np

from sageqa.data.token_embedding_store import TokenEmbeddingStore


def test_ragged_token_store_get_and_padded_batch(tmp_path: Path):
    root = tmp_path / "10k"
    shared = root / "shared"
    shared.mkdir(parents=True)
    vectors = np.arange(5 * 3, dtype=np.float16).reshape(5, 3)
    offsets = np.array([0, 2, 5], dtype=np.int64)
    np.save(shared / "tokens.npy", vectors, allow_pickle=False)
    np.save(shared / "offsets.npy", offsets, allow_pickle=False)
    manifest = {
        "format": "ragged_token_embeddings_v1",
        "complete": True,
        "embedding_dim": 3,
        "storage_dtype": "float16",
        "embedding_store": "shared",
        "num_sentences": 2,
        "parts": [{"part_index": 0, "start_id": 1, "end_id": 2, "rows": 2,
                   "token_file": "tokens.npy", "offset_file": "offsets.npy"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with TokenEmbeddingStore(root) as store:
        assert store.get(1).shape == (2, 3)
        assert store.get(2).shape == (3, 3)
        batch, mask, lengths = store.get_batch([1, 2, 0])
        assert batch.shape == (3, 3, 3)
        assert lengths.tolist() == [2, 3, 0]
        assert mask.tolist() == [[True, True, False], [True, True, True], [False, False, False]]
