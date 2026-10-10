# Token-level text embeddings (pre-pooling)

## What is stored

`encode_text_embeddings.py` uses the Transformer module inside the configured SentenceTransformer model and reads its `token_embeddings` output **before** the sentence pooling module. For all-MiniLM-L6-v2, each valid token state is 384-dimensional.

A sentence with L valid tokenizer positions therefore maps to `[L, 384]`, not `[384]`. L refers to tokenizer tokens/subwords, not whitespace-separated words. Valid special tokens such as CLS/SEP are retained; batch-padding positions are removed based on `attention_mask`.

## Storage layout

Token outputs are deliberately separate from the existing pooled embeddings:

```text
data/embeddings/text/all-MiniLM-L6-v2/<scale>/
  manifest.json
  shared/
    part-000000.tokens.npy    # flattened token vectors [total_tokens, 384]
    part-000000.offsets.npy   # sentence boundaries [rows_in_part + 1]
  train/manifest.json
  val/manifest.json
  test/manifest.json
```

Each part contains contiguous sentence IDs. For a sentence at local row `i`, its token sequence is `tokens[offsets[i]:offsets[i+1]]`. Storage padding is not written to disk. Dtype is configured as FP16 or FP32. The manifest records sentence ID ranges, token count, dtype, hidden dimension and output signature so partial writes can resume safely.

## Configuration

Merge `configs/embeddings/token_embedding_settings.yaml` into the existing `configs/embeddings/text_pipeline.yaml`. Keep your current `paths` and `dataset` sections. `token_model_slug` defaults to a separate output namespace so pooled artifacts remain unchanged.

## Run

From the Sage repository root:

```bash
python -m scripts.encode_text_embeddings \
  --config configs/embeddings/text_pipeline.yaml \
  --scale 10k \
  --limit-sentences 100
```

After checking output shapes and retrieval, run the full scale without the debug limit:

```bash
python -m scripts.encode_text_embeddings \
  --config configs/embeddings/text_pipeline.yaml \
  --scale 10k --overwrite
```

The debug-limited and complete runs have different manifests. After verifying the 100-sentence test, use `--overwrite` for the full 10k run to replace that limited test output. Repeat for `100k`, `1m`, and `10m` when validated. Use `--overwrite` only when you intend to rebuild the selected token-output scale.

## Lookup

```python
from utils.text_token_embedding_store import TokenEmbeddingStore

with TokenEmbeddingStore(
    "data/embeddings/text/all-MiniLM-L6-v2/10k"
) as store:
    tokens = store.get(12)  # [L, 384]
    batch, attention_mask, lengths = store.get_batch([12, 31, 52])
    # batch: [B, max_tokens, 384]; attention_mask: [B, max_tokens]
```

IDs are local to the sentence dictionary for the selected scale. Never use a 10k ID with a 10m embedding store or dictionary.

## Validation checks

- Validate `[L, 384]` for examples with multiple different sequence lengths.
- Compare `L` against the tokenizer's `attention_mask.sum()`.
- Confirm no padding positions are stored and the offset array is monotonic.
- Retrieve IDs at the start/end of every shard and across shard boundaries.
- Interrupt/resume a small run and confirm mappings are unchanged.
- Do not delete pooled embeddings until token-level training and evaluation have been validated.
