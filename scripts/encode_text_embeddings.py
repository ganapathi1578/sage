#!/usr/bin/env python3
"""Extract pre-pooling, contextual token embeddings for the sentence dictionary.

Uses the first (Transformer) module of SentenceTransformer and deliberately does
not call the sentence-level pooling module. Each sentence ID maps to a variable-
length [tokens, hidden_dim] matrix stored in a ragged, memory-mappable format.

Run from the Sage repository root:
  python -m scripts.encode_text_token_embeddings \
    --config configs/embeddings/text_pipeline.yaml --scale 10k
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised on user's environment
    raise SystemExit("Missing dependency pyarrow. Install with: python -m pip install pyarrow") from exc

from utils.text_pipeline_common import (
    atomic_write_json,
    config_path,
    json_fingerprint,
    load_config,
    selected_scale,
)

FORMAT_VERSION = 1
TOKEN_FORMAT = "ragged_token_embeddings_v1"


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def dictionary_metadata(path: Path) -> tuple[int, int, int]:
    pf = pq.ParquetFile(path)
    return int(pf.metadata.num_rows), int(path.stat().st_mtime_ns), int(path.stat().st_size)


def iter_dictionary_rows(path: Path, batch_rows: int) -> Iterable[tuple[int, str | None]]:
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    if not {"sentence_id", "sentence"}.issubset(names):
        raise SystemExit(f"Dictionary must contain sentence_id and sentence columns: {path}")
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["sentence_id", "sentence"]):
        ids = batch.column(0).to_pylist()
        texts = batch.column(1).to_pylist()
        yield from zip(ids, texts)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, arr, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def token_part_paths(shared_root: Path, index: int) -> tuple[Path, Path]:
    stem = f"part-{index:06d}"
    return shared_root / f"{stem}.tokens.npy", shared_root / f"{stem}.offsets.npy"


def encode_token_batch(model: Any, texts: list[str], device: str, dtype: np.dtype) -> list[np.ndarray]:
    """Return each sentence's valid contextual token states, before pooling.

    The attention mask removes batch padding. Special tokens (e.g. CLS/SEP) are
    retained because they are valid tokens according to the tokenizer mask.
    """
    import torch

    features = model.tokenize(texts)
    features = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in features.items()
    }
    if "attention_mask" not in features:
        raise RuntimeError("Tokenizer output does not include attention_mask")

    transformer = model[0]
    with torch.inference_mode():
        output = transformer(features)
        token_states = output.get("token_embeddings")
        if token_states is None:
            raise RuntimeError(
                "The first SentenceTransformer module did not return token_embeddings. "
                "Expected a Transformer module before the pooling module."
            )
        attention = features["attention_mask"].to(dtype=torch.bool)
        token_states = token_states.detach()
        result: list[np.ndarray] = []
        for row in range(token_states.shape[0]):
            valid_states = token_states[row][attention[row]]
            seq = valid_states.to(device="cpu").numpy()
            result.append(np.asarray(seq, dtype=dtype, order="C"))
    return result


def valid_existing_part(
    record: dict[str, Any] | None,
    token_path: Path,
    offsets_path: Path,
    start_id: int,
    end_id: int,
    embedding_dim: int,
) -> bool:
    if not record or not token_path.is_file() or not offsets_path.is_file():
        return False
    try:
        if int(record["start_id"]) != start_id or int(record["end_id"]) != end_id:
            return False
        if int(record["embedding_dim"]) != embedding_dim:
            return False
        tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        valid = (
            tokens.ndim == 2
            and tokens.shape[1] == embedding_dim
            and offsets.ndim == 1
            and offsets.shape[0] == end_id - start_id + 2
            and int(offsets[0]) == 0
            and int(offsets[-1]) == tokens.shape[0]
            and np.all(offsets[1:] >= offsets[:-1])
        )
        del tokens, offsets
        return bool(valid)
    except Exception:
        return False


def make_split_manifests(
    cfg: dict[str, Any], analysis_root: Path, output_root: Path, scale: str,
    model_name: str, output_slug: str, embedding_dim: int,
) -> None:
    """Write per-split pointers; shared token arrays remain globally deduplicated."""
    paths_cfg = cfg.get("paths", {})
    dataset_cfg = cfg.get("dataset", {})
    emb_cfg = cfg.get("embedding", {})
    compact_root = analysis_root / str(paths_cfg.get("compact_dir_name", "compact"))
    shared_name = str(emb_cfg.get("shared_dir_name", "shared"))
    split_names = list(emb_cfg.get("splits", dataset_cfg.get("split_names", ["train", "val", "test"])))
    compact_files = sorted(compact_root.glob("**/*.parquet")) if compact_root.is_dir() else []
    split_column = str(dataset_cfg.get("split_column", "split"))
    row_split_files: list[Path] = []
    for file_path in compact_files:
        try:
            if split_column in pq.ParquetFile(file_path).schema_arrow.names:
                row_split_files.append(file_path)
        except Exception:
            continue

    for split in split_names:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        matched: list[str] = []
        for data_path in compact_files:
            rel = data_path.relative_to(compact_root)
            parent_parts = [part.lower() for part in rel.parts[:-1]]
            has_split = split.lower() in parent_parts
            if split.lower() == "val":
                has_split = has_split or any(p in {"valid", "validation"} for p in parent_parts)
            if has_split:
                matched.append(os.path.relpath(data_path.resolve(), split_dir.resolve()))
        filter_column = None
        if not matched and row_split_files:
            matched = [os.path.relpath(p.resolve(), split_dir.resolve()) for p in row_split_files]
            filter_column = split_column
        payload = {
            "format": TOKEN_FORMAT,
            "split": split,
            "scale": scale,
            "model_name": model_name,
            "model_slug": output_slug,
            "embedding_dim": embedding_dim,
            "embedding_store_root": f"../{shared_name}",
            "embedding_manifest": "../manifest.json",
            "compact_data_root": os.path.relpath(compact_root.resolve(), split_dir.resolve()),
            "compact_parquet_files": matched,
            "query_id_column": dataset_cfg.get("query_id_column", "query_sentence_id"),
            "options_id_column": dataset_cfg.get("option_ids_column", "option_sentence_ids"),
            "filter_column": filter_column,
            "filter_value": split if filter_column else None,
            "lookup_note": (
                "TokenEmbeddingStore returns [sequence_length, embedding_dim] arrays. "
                "IDs are from this scale's sentence dictionary."
            ),
        }
        atomic_write_json(split_dir / "manifest.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale", help="Override dataset.scale")
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild this token-output scale")
    parser.add_argument("--limit-sentences", type=int, help="Optional debug limit on dictionary rows")
    args = parser.parse_args()

    cfg, _, sage_root = load_config(args.config)
    scale, _ = selected_scale(cfg, args.scale)
    analysis_root = config_path(cfg, "analysis_root", sage_root) / scale
    paths_cfg = cfg.get("paths", {})
    dataset_cfg = cfg.get("dataset", {})
    emb_cfg = cfg.get("embedding", {})
    dictionary_filename = str(paths_cfg.get("dictionary_filename", "sentence_dictionary.parquet"))
    dictionary_path = analysis_root / dictionary_filename
    if not dictionary_path.is_file():
        raise SystemExit(f"Sentence dictionary not found: {dictionary_path}\nRun build_sentence_ids.py first.")

    model_name = str(emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    revision = str(emb_cfg.get("revision", "main"))
    pooled_slug = str(emb_cfg.get("model_slug", model_name.rsplit("/", 1)[-1]))
    output_slug = str(emb_cfg.get("token_model_slug", f"{pooled_slug}-tokens"))
    output_root = config_path(cfg, "embeddings_root", sage_root) / output_slug / scale
    shared_name = str(emb_cfg.get("shared_dir_name", "shared"))
    shared_root = output_root / shared_name
    overwrite = args.overwrite or bool(emb_cfg.get("token_overwrite", False))
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    num_sentences, dict_mtime_ns, dict_size_bytes = dictionary_metadata(dictionary_path)
    if args.limit_sentences is not None:
        if args.limit_sentences < 0:
            raise SystemExit("--limit-sentences must be >= 0")
        num_sentences = min(num_sentences, args.limit_sentences)

    device = resolve_device(str(emb_cfg.get("device", "auto")))
    dtype_name = str(emb_cfg.get("storage_dtype", "float16")).lower()
    if dtype_name not in {"float16", "float32"}:
        raise SystemExit("embedding.storage_dtype must be float16 or float32")
    storage_dtype = np.float16 if dtype_name == "float16" else np.float32
    rows_per_chunk = max(1, int(emb_cfg.get("token_rows_per_chunk", 5000)))
    batch_size = max(1, int(emb_cfg.get("batch_size", 256)))
    configured_max_seq_length = emb_cfg.get("max_seq_length")

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing PyTorch or sentence-transformers. Install the Sage requirements first."
        ) from exc

    print(f"Dictionary:  {dictionary_path} ({num_sentences:,} sentences)")
    print(f"Model:       {model_name} @ {revision}")
    print(f"Output mode: token-level states before pooling")
    print(f"Device:      {device}")
    print(f"Output:      {output_root}")
    print(f"Storage:     {dtype_name}; {rows_per_chunk:,} sentences per ragged chunk")

    model = SentenceTransformer(model_name, revision=revision, device=device)
    model.eval()
    if configured_max_seq_length is not None:
        model.max_seq_length = int(configured_max_seq_length)
    effective_max_seq_length = int(model.max_seq_length)
    transformer = model[0]
    if not hasattr(transformer, "get_word_embedding_dimension"):
        raise SystemExit(
            "The first SentenceTransformer module is not a compatible Transformer module; "
            "cannot access pre-pooling token embeddings."
        )
    embedding_dim = int(transformer.get_word_embedding_dimension())

    signature_payload = {
        "format_version": FORMAT_VERSION,
        "format": TOKEN_FORMAT,
        "model_name": model_name,
        "revision": revision,
        "output_slug": output_slug,
        "scale": scale,
        "dictionary_path": str(dictionary_path.resolve()),
        "dictionary_rows_used": num_sentences,
        "dictionary_mtime_ns": dict_mtime_ns,
        "dictionary_size_bytes": dict_size_bytes,
        "storage_dtype": dtype_name,
        "max_seq_length": effective_max_seq_length,
        "rows_per_chunk": rows_per_chunk,
        "include_special_tokens": True,
        "debug_limit_sentences": args.limit_sentences,
    }
    signature = json_fingerprint(signature_payload)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists() and not overwrite:
        with manifest_path.open("r", encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        if old_manifest.get("signature") != signature:
            raise SystemExit(
                f"Existing token embedding manifest does not match this configuration: {manifest_path}\n"
                "Use --overwrite to rebuild token outputs, or restore the original configuration."
            )
        if not bool(emb_cfg.get("resume", True)) and not old_manifest.get("complete", False):
            raise SystemExit("Partial token output exists but embedding.resume is false; use --overwrite or enable resume.")
        manifest = old_manifest
    else:
        if any(output_root.iterdir()) and not overwrite:
            raise SystemExit(f"Output exists without a compatible manifest: {output_root}. Use --overwrite to rebuild.")
        manifest = {
            **signature_payload,
            "signature": signature,
            "device_used": device,
            "embedding_dim": embedding_dim,
            "num_sentences": num_sentences,
            "complete": False,
            "embedding_store": shared_name,
            "sentence_id_mapping": "sentence ID N maps to sentence row N-1; ID 0 is reserved for null",
            "storage_layout": {
                "tokens": "part-NNNNNN.tokens.npy: concatenated valid token vectors [total_tokens, embedding_dim]",
                "offsets": "part-NNNNNN.offsets.npy: per-sentence boundaries [rows+1], local to token part",
                "special_tokens": "CLS/SEP and other tokenizer tokens with attention_mask=1 are retained",
                "padding": "batch padding is not stored; reconstruct attention masks from sequence lengths",
            },
            "parts": [],
        }
        atomic_write_json(manifest_path, manifest)

    shared_root.mkdir(parents=True, exist_ok=True)
    parts_by_index = {int(part["part_index"]): part for part in manifest.get("parts", [])}
    buffer_ids: list[int] = []
    buffer_texts: list[str] = []
    expected_next_id = 1
    part_index = 0
    dictionary_batch_rows = max(1, int(dataset_cfg.get("batch_rows", 4096)))

    def flush_chunk(ids: list[int], texts: list[str], index: int) -> None:
        if not ids:
            return
        expected_start = index * rows_per_chunk + 1
        if ids[0] != expected_start or ids != list(range(expected_start, expected_start + len(ids))):
            raise RuntimeError(
                f"Dictionary sentence IDs must be contiguous: chunk={index}, "
                f"first={ids[0]}, expected={expected_start}"
            )
        expected_end = ids[-1]
        token_path, offsets_path = token_part_paths(shared_root, index)
        previous = parts_by_index.get(index)
        if valid_existing_part(previous, token_path, offsets_path, expected_start, expected_end, embedding_dim):
            print(f"  resume {token_path.name} | sentence IDs {expected_start:,}-{expected_end:,}")
            return

        sequences: list[np.ndarray] = []
        offsets = [0]
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start:batch_start + batch_size]
            encoded = encode_token_batch(model, batch_texts, device, storage_dtype)
            for sequence in encoded:
                if sequence.ndim != 2 or sequence.shape[1] != embedding_dim:
                    raise RuntimeError(f"Unexpected token sequence shape {sequence.shape}; expected [L, {embedding_dim}]")
                sequences.append(sequence)
                offsets.append(offsets[-1] + int(sequence.shape[0]))

        if sequences:
            flat_tokens = np.concatenate(sequences, axis=0).astype(storage_dtype, copy=False)
        else:
            flat_tokens = np.empty((0, embedding_dim), dtype=storage_dtype)
        offsets_array = np.asarray(offsets, dtype=np.int64)
        if offsets_array.shape[0] != len(ids) + 1 or int(offsets_array[-1]) != flat_tokens.shape[0]:
            raise RuntimeError("Internal ragged sequence offsets validation failed")

        # Write data files first. The manifest is updated only after both files exist.
        atomic_save_npy(token_path, flat_tokens)
        atomic_save_npy(offsets_path, offsets_array)
        record = {
            "part_index": index,
            "start_id": int(expected_start),
            "end_id": int(expected_end),
            "rows": len(ids),
            "token_count": int(flat_tokens.shape[0]),
            "embedding_dim": embedding_dim,
            "dtype": dtype_name,
            "token_file": token_path.name,
            "offset_file": offsets_path.name,
            "offsets_shape": [len(ids) + 1],
            "token_shape": [int(flat_tokens.shape[0]), embedding_dim],
        }
        parts_by_index[index] = record
        manifest["parts"] = [parts_by_index[key] for key in sorted(parts_by_index)]
        manifest["complete"] = False
        atomic_write_json(manifest_path, manifest)
        avg_len = flat_tokens.shape[0] / len(ids) if ids else 0.0
        print(
            f"  wrote {token_path.name} | IDs {expected_start:,}-{expected_end:,} "
            f"| tokens={flat_tokens.shape[0]:,} | avg_tokens={avg_len:.2f}"
        )

    for sentence_id, sentence in iter_dictionary_rows(dictionary_path, dictionary_batch_rows):
        if len(buffer_ids) >= num_sentences:
            break
        sid = int(sentence_id)
        if sid != expected_next_id:
            raise RuntimeError(f"Dictionary IDs are not sorted/contiguous: expected {expected_next_id}, got {sid}")
        expected_next_id += 1
        buffer_ids.append(sid)
        buffer_texts.append("" if sentence is None else str(sentence))
        if len(buffer_ids) == rows_per_chunk:
            flush_chunk(buffer_ids, buffer_texts, part_index)
            buffer_ids, buffer_texts = [], []
            part_index += 1
    if buffer_ids:
        flush_chunk(buffer_ids, buffer_texts, part_index)
    if expected_next_id - 1 != num_sentences:
        raise RuntimeError(f"Read {expected_next_id - 1:,} dictionary rows but expected {num_sentences:,}")

    expected_parts = (num_sentences + rows_per_chunk - 1) // rows_per_chunk
    if len(parts_by_index) != expected_parts or any(i not in parts_by_index for i in range(expected_parts)):
        raise RuntimeError(f"Expected {expected_parts} token chunks; manifest contains {len(parts_by_index)}")
    # Validate every part before marking the store complete.
    for index in range(expected_parts):
        part = parts_by_index[index]
        token_path = shared_root / str(part["token_file"])
        offsets_path = shared_root / str(part["offset_file"])
        if not valid_existing_part(part, token_path, offsets_path, int(part["start_id"]), int(part["end_id"]), embedding_dim):
            raise RuntimeError(f"Token chunk failed final validation: {token_path}")

    manifest["parts"] = [parts_by_index[i] for i in sorted(parts_by_index)]
    manifest["complete"] = True
    manifest["num_sentences"] = num_sentences
    atomic_write_json(manifest_path, manifest)
    make_split_manifests(cfg, analysis_root, output_root, scale, model_name, output_slug, embedding_dim)
    print(f"\nToken extraction complete: {num_sentences:,} sentences, hidden_dim={embedding_dim}")
    print(f"Format: {TOKEN_FORMAT}; dtype={dtype_name}; maximum sequence length={effective_max_seq_length}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
