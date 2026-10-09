#!/usr/bin/env python3
"""Encode each unique dictionary sentence once into resumable FP16/FP32 NumPy shards."""
from __future__ import annotations

import argparse
import json
import shutil
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("Missing dependency pyarrow. Install with: python -m pip install pyarrow") from exc

from utils.text_pipeline_common import atomic_write_json, config_path, json_fingerprint, load_config, selected_scale


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def dictionary_metadata(dictionary_path: Path) -> tuple[int, int]:
    pf = pq.ParquetFile(dictionary_path)
    return int(pf.metadata.num_rows), int(dictionary_path.stat().st_mtime_ns)


def iter_dictionary_chunks(dictionary_path: Path, input_batch_rows: int):
    pf = pq.ParquetFile(dictionary_path)
    names = set(pf.schema_arrow.names)
    if "sentence_id" not in names or "sentence" not in names:
        raise SystemExit(f"Dictionary must contain sentence_id and sentence columns: {dictionary_path}")
    for batch in pf.iter_batches(batch_size=input_batch_rows, columns=["sentence_id", "sentence"]):
        ids = batch.column(0).to_pylist()
        texts = batch.column(1).to_pylist()
        yield from zip(ids, texts)


def part_name(part_index: int) -> str:
    return f"part-{part_index:06d}.npy"


def write_numpy_atomic(path: Path, array: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
    temp.replace(path)


def part_record(path: Path, part_index: int, start_id: int, arr: np.ndarray) -> dict[str, Any]:
    return {
        "part_index": part_index,
        "file": path.name,
        "start_id": int(start_id),
        "end_id": int(start_id + arr.shape[0] - 1),
        "rows": int(arr.shape[0]),
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale", help="Override dataset.scale")
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild this model+scale output")
    parser.add_argument("--limit-sentences", type=int, help="Optional debug cap on unique dictionary rows")
    args = parser.parse_args()

    cfg, _, sage_root = load_config(args.config)
    scale, _ = selected_scale(cfg, args.scale)
    analysis_root = config_path(cfg, "analysis_root", sage_root) / scale
    dictionary_filename = str(cfg.get("paths", {}).get("dictionary_filename", "sentence_dictionary.parquet"))
    compact_dir_name = str(cfg.get("paths", {}).get("compact_dir_name", "compact"))
    dictionary_path = analysis_root / dictionary_filename
    if not dictionary_path.is_file():
        raise SystemExit(f"Sentence dictionary not found: {dictionary_path}\nRun build_sentence_ids.py first.")

    emb_cfg = cfg.get("embedding", {})
    model_name = str(emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    model_slug = str(emb_cfg.get("model_slug", model_name.rsplit("/", 1)[-1]))
    output_root = config_path(cfg, "embeddings_root", sage_root) / model_slug / scale
    shared_dir_name = str(emb_cfg.get("shared_dir_name", "shared"))
    shared_root = output_root / shared_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output_root / "manifest.json"
    overwrite = args.overwrite or bool(emb_cfg.get("overwrite", False))
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    num_sentences, dict_mtime_ns = dictionary_metadata(dictionary_path)
    if args.limit_sentences is not None:
        if args.limit_sentences < 0:
            raise SystemExit("--limit-sentences must be >= 0")
        num_sentences = min(num_sentences, args.limit_sentences)

    device = resolve_device(str(emb_cfg.get("device", "auto")))
    storage_dtype_name = str(emb_cfg.get("storage_dtype", "float16")).lower()
    if storage_dtype_name not in {"float16", "float32"}:
        raise SystemExit("embedding.storage_dtype must be float16 or float32")
    storage_dtype = np.float16 if storage_dtype_name == "float16" else np.float32
    chunk_rows = max(1, int(emb_cfg.get("rows_per_chunk", 50_000)))
    batch_size = max(1, int(emb_cfg.get("batch_size", 256)))
    max_seq_length = emb_cfg.get("max_seq_length")

    signature_payload = {
        "format_version": 1,
        "model_name": model_name,
        "revision": emb_cfg.get("revision", "main"),
        "model_slug": model_slug,
        "scale": scale,
        "dictionary_path": str(dictionary_path.resolve()),
        "dictionary_rows": num_sentences,
        "dictionary_mtime_ns": dict_mtime_ns,
        "dictionary_size_bytes": int(dictionary_path.stat().st_size),
        "storage_dtype": storage_dtype_name,
        "normalize_embeddings": bool(emb_cfg.get("normalize_embeddings", False)),
        "max_seq_length": max_seq_length,
        "rows_per_chunk": chunk_rows,
    }
    signature = json_fingerprint(signature_payload)

    manifest: dict[str, Any]
    if existing_manifest_path.exists() and not overwrite:
        with existing_manifest_path.open("r", encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        if old_manifest.get("signature") != signature:
            raise SystemExit(
                f"Existing embedding manifest does not match this config: {existing_manifest_path}\n"
                "Use embedding.overwrite: true or --overwrite to rebuild, or restore the original config."
            )
        if not bool(emb_cfg.get("resume", True)) and not old_manifest.get("complete", False):
            raise SystemExit("Partial output exists but embedding.resume is false; use --overwrite or enable resume.")
        manifest = old_manifest
    else:
        if any(output_root.iterdir()) and not overwrite:
            raise SystemExit(f"Output exists without a compatible manifest: {output_root}. Use --overwrite to rebuild.")
        manifest = {
            **signature_payload,
            "signature": signature,
            "device_used": device,
            "storage_dtype": storage_dtype_name,
            "embedding_dim": None,
            "parts": [],
            "complete": False,
            "embedding_store": shared_dir_name,
            "sentence_id_mapping": "sentence_id N maps to row N-1 across contiguous part files; ID 0 is reserved for null",
        }
        atomic_write_json(existing_manifest_path, manifest)

    parts_by_index = {int(p["part_index"]): p for p in manifest.get("parts", [])}
    expected_part_count = (num_sentences + chunk_rows - 1) // chunk_rows
    complete_files_exist = len(parts_by_index) == expected_part_count and all(
        (shared_root / str(part.get("file", part_name(index)))).is_file()
        for index, part in parts_by_index.items()
    )
    if manifest.get("complete") and complete_files_exist:
        write_split_manifests(cfg, analysis_root, output_root, scale, model_name, model_slug)
        print(f"Embeddings already complete: {output_root}")
        return 0

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Missing dependency sentence-transformers. Install requirements.txt first.") from exc

    print(f"Dictionary:  {dictionary_path} ({num_sentences:,} sentences)")
    print(f"Model:       {model_name} @ {emb_cfg.get('revision', 'main')}")
    print(f"Device:      {device}")
    print(f"Output:      {output_root}")
    print(f"Storage:     {storage_dtype_name}; {chunk_rows:,} sentences per chunk")
    model = SentenceTransformer(model_name, revision=emb_cfg.get("revision", "main"), device=device)
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)
    embedding_dim = int(model.get_sentence_embedding_dimension())
    if manifest.get("embedding_dim") not in (None, embedding_dim):
        raise SystemExit(f"Embedding dimension changed: manifest={manifest.get('embedding_dim')} current_model={embedding_dim}")
    manifest["embedding_dim"] = embedding_dim
    manifest["device_used"] = device
    atomic_write_json(existing_manifest_path, manifest)

    buffer_ids: list[int] = []
    buffer_texts: list[str] = []
    expected_next_id = 1
    current_part = 0

    def flush_chunk(ids: list[int], texts: list[str], part_index: int) -> None:
        #split_names = cfg.get("embedding", {}).get("splits",cfg.get("dataset", {}).get("split_names", ["train", "val", "test"]),)
        #split_names = cfg.get("embedding", {}).get("splits",cfg.get("dataset", {}).get("split_names", ["train", "val", "test"]),)
        #split_names = cfg.get("embedding", {}).get("splits",cfg.get("dataset", {}).get("split_names", ["train", "val", "test"]),)

        if not ids:
            return
        expected_start = part_index * chunk_rows + 1
        if ids[0] != expected_start or ids != list(range(expected_start, expected_start + len(ids))):
            raise RuntimeError(f"Dictionary sentence IDs are not contiguous for chunk {part_index}: first={ids[0]} expected={expected_start}")
        out_path = shared_root / part_name(part_index)
        old_part = parts_by_index.get(part_index)
        valid_existing = False
        if old_part and out_path.is_file():
            expected_end = ids[-1]
            try:
                valid_existing = (int(old_part["start_id"]) == ids[0]
                                  and int(old_part["end_id"]) == expected_end
                                  and tuple(old_part["shape"]) == (len(ids), embedding_dim))
                if valid_existing:
                    arr_check = np.load(out_path, mmap_mode="r", allow_pickle=False)
                    valid_existing = arr_check.shape == (len(ids), embedding_dim)
                    del arr_check
            except Exception:
                valid_existing = False
        if not valid_existing:
            arr = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=bool(emb_cfg.get("show_progress_bar", True)),
                convert_to_numpy=True,
                normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", False)),
            )
            arr = np.asarray(arr, dtype=storage_dtype, order="C")
            if arr.ndim != 2 or arr.shape != (len(ids), embedding_dim):
                raise RuntimeError(f"Unexpected embedding shape {arr.shape}; expected {(len(ids), embedding_dim)}")
            shared_root.mkdir(parents=True, exist_ok=True)
            write_numpy_atomic(out_path, arr)
            record = part_record(out_path, part_index, ids[0], arr)
            parts_by_index[part_index] = record
            manifest["parts"] = [parts_by_index[i] for i in sorted(parts_by_index)]
            manifest["complete"] = False
            atomic_write_json(existing_manifest_path, manifest)
            print(f"  wrote {out_path.name} | IDs {ids[0]:,}-{ids[-1]:,} | shape={arr.shape}")
        else:
            print(f"  resume {out_path.name} | IDs {ids[0]:,}-{ids[-1]:,}")
        del texts[:]
        del ids[:]

    for sentence_id, sentence in iter_dictionary_chunks(dictionary_path, int(cfg["dataset"].get("batch_rows", 4096))):
        if len(buffer_ids) >= num_sentences:
            break
        sid = int(sentence_id)
        if sid != expected_next_id:
            raise RuntimeError(f"Dictionary is not sorted/contiguous: expected ID {expected_next_id}, got {sid}")
        expected_next_id += 1
        buffer_ids.append(sid)
        buffer_texts.append("" if sentence is None else str(sentence))
        if len(buffer_ids) == chunk_rows:
            flush_chunk(buffer_ids, buffer_texts, current_part)
            buffer_ids, buffer_texts = [], []
            current_part += 1
    if buffer_ids:
        flush_chunk(buffer_ids, buffer_texts, current_part)
    if expected_next_id - 1 != num_sentences:
        raise RuntimeError(f"Read {expected_next_id - 1:,} dictionary rows but expected {num_sentences:,}")

    expected_parts = (num_sentences + chunk_rows - 1) // chunk_rows
    if len(parts_by_index) != expected_parts:
        raise RuntimeError(f"Expected {expected_parts} embedding chunks; manifest contains {len(parts_by_index)}")
    manifest["parts"] = [parts_by_index[i] for i in sorted(parts_by_index)]
    manifest["complete"] = True
    manifest["num_sentences"] = num_sentences
    atomic_write_json(existing_manifest_path, manifest)

    write_split_manifests(cfg, analysis_root, output_root, scale, model_name, model_slug)
    print(f"\nEmbedding extraction complete: {num_sentences:,} unique sentences, dim={embedding_dim}")
    print(f"Manifest: {existing_manifest_path}")
    split_names = cfg.get("embedding", {}).get("splits",cfg.get("dataset", {}).get("split_names", ["train", "val", "test"]),)

    print("Split manifests written under: " + ", ".join(str(output_root / s) for s in split_names))
    return 0


def write_split_manifests(cfg: dict[str, Any], analysis_root: Path, output_root: Path,
                          scale: str, model_name: str, model_slug: str) -> None:
    """Write small per-split pointers; vectors stay globally deduplicated in shared/."""
    compact_dir_name = str(cfg.get("paths", {}).get("compact_dir_name", "compact"))
    shared_dir_name = str(cfg.get("embedding", {}).get("shared_dir_name", "shared"))
    compact_root = analysis_root / compact_dir_name
    split_names = list(cfg.get("embedding", {}).get("splits", cfg.get("dataset", {}).get("split_names", ["train", "val", "test"])))
    compact_files = sorted(compact_root.glob("**/*.parquet")) if compact_root.is_dir() else []
    row_split_files = []
    split_column = cfg.get("dataset", {}).get("split_column", "split")
    for data_path in compact_files:
        try:
            if split_column in pq.ParquetFile(data_path).schema_arrow.names:
                row_split_files.append(data_path)
        except Exception:
            pass
    for split in split_names:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        matched = []
        for data_path in compact_files:
            relative = data_path.relative_to(compact_root)
            parts = [p.lower() for p in relative.parts[:-1]]
            has_split = split.lower() in parts
            if split.lower() == "val":
                has_split = has_split or any(p in {"valid", "validation"} for p in parts)
            if has_split:
                matched.append(os.path.relpath(data_path.resolve(), split_dir.resolve()))
        filter_column = None
        if not matched and row_split_files:
            # Flat Parquet datasets may store split membership in a column rather than paths.
            matched = [os.path.relpath(data_path.resolve(), split_dir.resolve()) for data_path in row_split_files]
            filter_column = split_column
        split_manifest = {
            "split": split,
            "scale": scale,
            "model_name": model_name,
            "model_slug": model_slug,
            "embedding_store_root": f"../{shared_dir_name}",
            "embedding_manifest": "../manifest.json",
            "compact_data_root": os.path.relpath(compact_root.resolve(), split_dir.resolve()),
            "compact_parquet_files": matched,
            "query_id_column": cfg.get("dataset", {}).get("query_id_column", "query_sentence_id"),
            "options_id_column": cfg.get("dataset", {}).get("option_ids_column", "option_sentence_ids"),
            "filter_column": filter_column,
            "filter_value": split if filter_column else None,
            "lookup_note": "Load vectors with SentenceEmbeddingStore rooted at the parent directory; embeddings are globally deduplicated and addressed by sentence_id.",
        }
        atomic_write_json(split_dir / "manifest.json", split_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
