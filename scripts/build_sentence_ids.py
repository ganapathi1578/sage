#!/usr/bin/env python3
"""Convert text columns in an already split UniProp Parquet scale into sentence IDs.

Expected source layout (the selected scale must already exist):
    UniProp/data/generated/<scale>/train/**/*.parquet
    UniProp/data/generated/<scale>/val/**/*.parquet
    UniProp/data/generated/<scale>/test/**/*.parquet

This script DOES NOT create, re-sample, or redistribute splits. It processes every
row in every Parquet file under the selected scale directory and mirrors the
source-relative paths under:
    UniProp/data/analysis/<scale>/compact/<split>/...

A single sentence dictionary is shared across all source splits for this scale,
so identical normalized text has the same ID in train, val, and test.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterator

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyarrow. Install it with: python -m pip install pyarrow"
    ) from exc

# Shared helpers live at sage/utils/, not sage/scripts/.
from utils.text_pipeline_common import (  # noqa: E402
    atomic_write_json,
    config_path,
    json_fingerprint,
    load_config,
    normalize_sentence,
)


def resolve_scale_input(base_input_root: Path, raw_input_setting: str, scale: str) -> Path:
    """Select exactly one pre-generated scale directory; never scan sibling scales."""
    if "{scale}" in raw_input_setting:
        return base_input_root.resolve()
    return (base_input_root / scale).resolve()


def discover_files(input_root: Path, pattern: str, output_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise SystemExit(
            f"Input scale directory does not exist: {input_root}\n"
            "Expected: <paths.input_root>/<scale>/{train,val,test}/.../*.parquet\n"
            "Check paths.input_root and the --scale argument."
        )

    output_root_resolved = output_root.resolve()
    files: list[Path] = []
    for path in input_root.glob(pattern):
        if not path.is_file() or path.suffix.lower() != ".parquet":
            continue
        resolved = path.resolve()
        # Defensive guard: never re-ingest our own output if roots overlap.
        if resolved == output_root_resolved or output_root_resolved in resolved.parents:
            continue
        files.append(resolved)
    return sorted(files)


def identify_split(path: Path, input_root: Path, split_names: list[str]) -> str:
    """Return the existing split folder name for reporting only; never reallocates rows."""
    aliases = {"validation": "val", "valid": "val"}
    allowed = {str(x).lower() for x in split_names}
    try:
        relative_parts = path.relative_to(input_root).parts[:-1]
    except ValueError:
        relative_parts = path.parts[:-1]
    for part in relative_parts:
        split = aliases.get(part.lower(), part.lower())
        if split in allowed:
            return split
    return "unlabelled"


def create_database(db_path: Path, cache_mb: int) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(f"PRAGMA cache_size=-{max(16, int(cache_mb) * 1024)}")
    conn.execute("PRAGMA mmap_size=1073741824")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sentence (
            sentence_id INTEGER PRIMARY KEY,
            normalized_sentence TEXT NOT NULL UNIQUE,
            sentence TEXT NOT NULL,
            query_frequency INTEGER NOT NULL DEFAULT 0,
            option_frequency INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS processed_input (
            path TEXT PRIMARY KEY,
            rows INTEGER NOT NULL,
            query_occurrences INTEGER NOT NULL,
            option_occurrences INTEGER NOT NULL
        )"""
    )
    conn.commit()
    return conn


def plan_files(
    files: list[Path],
    input_root: Path,
    split_names: list[str],
    query_col: str,
    options_col: str,
) -> list[dict[str, Any]]:
    """Plan all source files as-is. No quotas, row caps, or split reallocation."""
    plan: list[dict[str, Any]] = []
    for path in files:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        if not {query_col, options_col}.issubset(names):
            print(
                f"WARNING: skipping {path}: requires columns {query_col!r} and {options_col!r}",
                file=sys.stderr,
            )
            continue
        plan.append(
            {
                "path": path,
                "rows": int(pf.metadata.num_rows),
                "split": identify_split(path, input_root, split_names),
                "schema": pf.schema_arrow,
            }
        )
    if not plan:
        raise SystemExit(
            f"No Parquet files with {query_col!r} and {options_col!r} columns found under {input_root}"
        )
    return plan


def iter_batches(path: Path, batch_rows: int) -> Iterator[pa.RecordBatch]:
    """Stream all rows in a Parquet file in bounded batches."""
    parquet_file = pq.ParquetFile(path)
    yield from parquet_file.iter_batches(batch_size=batch_rows)


def collect_sentence_counts(
    records: list[dict[str, Any]],
    query_col: str,
    options_col: str,
    norm_cfg: dict[str, Any],
) -> dict[str, list[int]]:
    """Build per-batch occurrence counts: [query_count, option_count]."""
    local: dict[str, list[int]] = {}
    for rec in records:
        query = normalize_sentence(rec.get(query_col), norm_cfg)
        if query is not None:
            local.setdefault(query, [0, 0])[0] += 1

        options = rec.get(options_col) or []
        if not isinstance(options, (list, tuple)):
            continue
        for option in options:
            normalized = normalize_sentence(option, norm_cfg)
            if normalized is not None:
                local.setdefault(normalized, [0, 0])[1] += 1
    return local


def first_pass(
    plan: list[dict[str, Any]], conn: sqlite3.Connection, cfg: dict[str, Any], batch_rows: int
) -> Counter:
    """Index all sentences in every selected-scale source shard, with per-file checkpoints."""
    dataset_cfg = cfg.get("dataset", {})
    norm_cfg = dataset_cfg.get("normalize", {})
    query_col = dataset_cfg.get("query_column", "query")
    options_col = dataset_cfg.get("options_column", "options")
    totals: Counter = Counter()
    insert_sql = """INSERT INTO sentence(
            normalized_sentence, sentence, query_frequency, option_frequency
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT(normalized_sentence) DO UPDATE SET
            query_frequency = query_frequency + excluded.query_frequency,
            option_frequency = option_frequency + excluded.option_frequency"""

    print("PASS 1/2: indexing unique sentences in SQLite (all rows; splits unchanged)")
    for index, item in enumerate(plan, 1):
        path: Path = item["path"]
        path_key = str(path.resolve())
        already = conn.execute(
            "SELECT rows, query_occurrences, option_occurrences FROM processed_input WHERE path = ?",
            (path_key,),
        ).fetchone()
        if already is not None:
            if int(already[0]) != int(item["rows"]):
                raise RuntimeError(
                    f"SQLite checkpoint row count changed for {path}; rebuild this scale with --overwrite."
                )
            totals["examples"] += int(already[0])
            totals["query_occurrences"] += int(already[1])
            totals["option_occurrences"] += int(already[2])
            print(f"  [{index}/{len(plan)}] RESUME {path.relative_to(item['input_root']) if 'input_root' in item else path.name} | rows={int(already[0]):,}")
            continue

        print(f"  [{index}/{len(plan)}] {item['relative_path']} | split={item['split']} | rows={item['rows']:,}")
        file_rows = 0
        file_query_occ = 0
        file_option_occ = 0
        conn.execute("BEGIN IMMEDIATE")
        try:
            for batch in iter_batches(path, batch_rows):
                records = batch.to_pylist()
                local = collect_sentence_counts(records, query_col, options_col, norm_cfg)
                if local:
                    conn.executemany(
                        insert_sql,
                        [(text, text, counts[0], counts[1]) for text, counts in local.items()],
                    )
                file_rows += len(records)
                file_query_occ += sum(counts[0] for counts in local.values())
                file_option_occ += sum(counts[1] for counts in local.values())
            if file_rows != int(item["rows"]):
                raise RuntimeError(
                    f"Read {file_rows:,} rows from {path}, but Parquet metadata reports {item['rows']:,}."
                )
            conn.execute(
                "INSERT INTO processed_input(path, rows, query_occurrences, option_occurrences) VALUES (?, ?, ?, ?)",
                (path_key, file_rows, file_query_occ, file_option_occ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        totals["examples"] += file_rows
        totals["query_occurrences"] += file_query_occ
        totals["option_occurrences"] += file_option_occ
        print(
            f"      committed {file_rows:,} rows | query mentions={file_query_occ:,} "
            f"| option mentions={file_option_occ:,}"
        )

    totals["unique_sentences"] = int(conn.execute("SELECT COUNT(*) FROM sentence").fetchone()[0])
    return totals


class IdLookup:
    """Bounded LRU cache around SQLite lookups to keep compaction memory bounded."""

    def __init__(self, conn: sqlite3.Connection, max_entries: int = 500_000, chunk_size: int = 800):
        self.conn = conn
        self.max_entries = max(0, int(max_entries))
        try:
            variable_limit = int(conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
        except (AttributeError, TypeError):
            variable_limit = 999
        self.chunk_size = min(max(1, int(chunk_size)), max(1, variable_limit - 16))
        self.cache: OrderedDict[str, int] = OrderedDict()

    def lookup(self, sentences: list[str | None]) -> dict[str, int]:
        unique = list(dict.fromkeys(text for text in sentences if text is not None))
        found: dict[str, int] = {}
        missing: list[str] = []
        for text in unique:
            sid = self.cache.get(text)
            if sid is not None:
                self.cache.move_to_end(text)
                found[text] = sid
            else:
                missing.append(text)

        for start in range(0, len(missing), self.chunk_size):
            chunk = missing[start : start + self.chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT normalized_sentence, sentence_id FROM sentence "
                f"WHERE normalized_sentence IN ({placeholders})",
                chunk,
            )
            for text, sentence_id in rows:
                sentence_id = int(sentence_id)
                found[text] = sentence_id
                if self.max_entries:
                    self.cache[text] = sentence_id
                    self.cache.move_to_end(text)
                    while len(self.cache) > self.max_entries:
                        self.cache.popitem(last=False)

        missing_after = [text for text in unique if text not in found]
        if missing_after:
            raise RuntimeError(
                f"ID lookup failed for {len(missing_after)} sentences; example={missing_after[0]!r}"
            )
        return found


def compact_shard(
    item: dict[str, Any],
    input_root: Path,
    output_root: Path,
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    batch_rows: int,
    lookup: IdLookup,
    compact_dir_name: str,
) -> tuple[Path, int]:
    """Replace query/options with IDs while preserving all remaining columns and split path."""
    path: Path = item["path"]
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    dataset_cfg = cfg.get("dataset", {})
    query_col = dataset_cfg.get("query_column", "query")
    options_col = dataset_cfg.get("options_column", "options")
    norm_cfg = dataset_cfg.get("normalize", {})
    id_type_name = str(dataset_cfg.get("sentence_id_type", "int32")).lower()
    if id_type_name not in {"int32", "int64"}:
        raise ValueError("dataset.sentence_id_type must be 'int32' or 'int64'")
    id_type = pa.int64() if id_type_name == "int64" else pa.int32()
    query_id_col = dataset_cfg.get("query_id_column", "query_sentence_id")
    option_ids_col = dataset_cfg.get("option_ids_column", "option_sentence_ids")

    existing_names = set(schema.names)
    if query_id_col in existing_names or option_ids_col in existing_names:
        raise SystemExit(
            f"Input {path} already contains ID output column(s) {query_id_col!r}/{option_ids_col!r}. "
            "The input to this stage must contain the original query/options text columns."
        )
    kept_fields = [field for field in schema if field.name not in {query_col, options_col}]
    out_schema = pa.schema(
        kept_fields
        + [
            pa.field(query_id_col, id_type),
            pa.field(option_ids_col, pa.list_(id_type)),
        ],
        metadata=schema.metadata,
    )

    relative = path.relative_to(input_root)
    out_path = output_root / compact_dir_name / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)

    writer = None
    total_written = 0
    try:
        for batch in iter_batches(path, batch_rows):
            records = batch.to_pylist()
            normalized_records: list[tuple[str | None, list[str | None]]] = []
            all_texts: list[str | None] = []
            for record in records:
                query_norm = normalize_sentence(record.get(query_col), norm_cfg)
                option_values = record.get(options_col) or []
                if not isinstance(option_values, (list, tuple)):
                    option_values = []
                option_norms = [normalize_sentence(value, norm_cfg) for value in option_values]
                normalized_records.append((query_norm, option_norms))
                all_texts.append(query_norm)
                all_texts.extend(option_norms)

            ids = lookup.lookup(all_texts)
            output_records = []
            for record, (query_norm, option_norms) in zip(records, normalized_records):
                compact = {key: value for key, value in record.items() if key not in {query_col, options_col}}
                compact[query_id_col] = ids[query_norm] if query_norm is not None else None
                compact[option_ids_col] = [ids[text] if text is not None else None for text in option_norms]
                output_records.append(compact)

            table = pa.Table.from_pylist(output_records, schema=out_schema)
            if writer is None:
                writer = pq.ParquetWriter(
                    str(temp_path), out_schema, compression=dataset_cfg.get("parquet_compression", "zstd")
                )
            writer.write_table(table, row_group_size=batch_rows)
            total_written += len(output_records)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(
            pa.Table.from_pylist([], schema=out_schema),
            str(temp_path),
            compression=dataset_cfg.get("parquet_compression", "zstd"),
        )
    if total_written != int(item["rows"]):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Compacted row count mismatch for {path}: {total_written} != {item['rows']}")
    temp_path.replace(out_path)
    return out_path, total_written


def export_dictionary(
    conn: sqlite3.Connection,
    output_root: Path,
    compression: str,
    export_csv: bool,
    id_type_name: str,
    dictionary_filename: str,
) -> tuple[int, int]:
    parquet_path = output_root / dictionary_filename
    id_type = pa.int64() if str(id_type_name).lower() == "int64" else pa.int32()
    schema = pa.schema(
        [
            pa.field("sentence_id", id_type),
            pa.field("sentence", pa.string()),
            pa.field("normalized_sentence", pa.string()),
            pa.field("query_frequency", pa.int64()),
            pa.field("option_frequency", pa.int64()),
            pa.field("total_frequency", pa.int64()),
        ]
    )
    sql = """SELECT sentence_id, sentence, normalized_sentence, query_frequency, option_frequency,
                    query_frequency + option_frequency AS total_frequency
             FROM sentence ORDER BY sentence_id"""
    temp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)
    writer = None
    cursor = conn.execute(sql)
    try:
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            values = [
                {
                    "sentence_id": int(row[0]),
                    "sentence": row[1],
                    "normalized_sentence": row[2],
                    "query_frequency": int(row[3]),
                    "option_frequency": int(row[4]),
                    "total_frequency": int(row[5]),
                }
                for row in rows
            ]
            table = pa.Table.from_pylist(values, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(str(temp_path), schema, compression=compression)
            writer.write_table(table, row_group_size=8192)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=schema), str(temp_path), compression=compression)
    temp_path.replace(parquet_path)

    csv_bytes = 0
    if export_csv:
        csv_path = output_root / "sentence_dictionary.csv"
        temp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with temp_csv.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(schema.names)
            cursor = conn.execute(sql)
            while True:
                rows = cursor.fetchmany(8192)
                if not rows:
                    break
                csv_writer.writerows(rows)
        temp_csv.replace(csv_path)
        csv_bytes = csv_path.stat().st_size

    return parquet_path.stat().st_size, csv_bytes


def build_plan_fingerprint(
    scale: str,
    input_root: Path,
    plan: list[dict[str, Any]],
    cfg: dict[str, Any],
    compact_dir_name: str,
    dictionary_filename: str,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset", {})
    return {
        "scale": scale,
        "input_root": str(input_root),
        "query_column": dataset_cfg.get("query_column", "query"),
        "options_column": dataset_cfg.get("options_column", "options"),
        "query_id_column": dataset_cfg.get("query_id_column", "query_sentence_id"),
        "option_ids_column": dataset_cfg.get("option_ids_column", "option_sentence_ids"),
        "split_names": dataset_cfg.get("split_names", ["train", "val", "test"]),
        "sentence_id_type": dataset_cfg.get("sentence_id_type", "int32"),
        "parquet_compression": dataset_cfg.get("parquet_compression", "zstd"),
        "normalize": dataset_cfg.get("normalize", {}),
        "batch_rows": dataset_cfg.get("batch_rows", 4096),
        "compact_dir_name": compact_dir_name,
        "dictionary_filename": dictionary_filename,
        "files": [
            {
                "relative_path": item["relative_path"],
                "split": item["split"],
                "rows": int(item["rows"]),
                "size": int(item["path"].stat().st_size),
                "mtime_ns": int(item["path"].stat().st_mtime_ns),
            }
            for item in plan
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Text pipeline YAML configuration")
    parser.add_argument("--scale", help="Scale folder to process, e.g. smoke, 10k, 100k, 1m")
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild this scale's analysis output")
    args = parser.parse_args()

    cfg, _, sage_root = load_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    paths_cfg = cfg.get("paths", {})
    scale = str(args.scale or dataset_cfg.get("scale", "smoke"))
    if not scale or scale in {".", ".."} or "/" in scale or "\\" in scale:
        raise SystemExit(f"Invalid scale name: {scale!r}")

    registered_scales = dataset_cfg.get("scale_limits", {})
    if registered_scales and scale not in registered_scales:
        raise SystemExit(
            f"Unknown scale {scale!r}. Configured scales: {', '.join(map(str, registered_scales.keys()))}"
        )

    raw_input_value = str(paths_cfg.get("input_root", "UniProp/data/generated"))
    formatted_input_value = raw_input_value.format(scale=scale)
    raw_path = Path(formatted_input_value).expanduser()
    if raw_path.is_absolute():
        input_root = raw_path.resolve()
    else:
        input_base = config_path({"paths": {"input_root": formatted_input_value}}, "input_root", sage_root)
        # Configs normally point at the parent of all scales. If {scale} was
        # supplied, that path is already scale-specific; otherwise append the scale.
        input_root = input_base if "{scale}" in raw_input_value else (input_base / scale).resolve()

    analysis_base = config_path(cfg, "analysis_root", sage_root)
    output_root = analysis_base / scale
    compact_dir_name = str(paths_cfg.get("compact_dir_name", "compact"))
    dictionary_filename = str(paths_cfg.get("dictionary_filename", "sentence_dictionary.parquet"))
    analysis_cfg = cfg.get("analysis", {})
    overwrite = args.overwrite or bool(analysis_cfg.get("overwrite", False))
    resume_enabled = bool(analysis_cfg.get("resume", True))
    batch_rows = max(1, int(dataset_cfg.get("batch_rows", 4096)))
    split_names = [str(name) for name in dataset_cfg.get("split_names", ["train", "val", "test"])]

    files = discover_files(input_root, str(dataset_cfg.get("input_glob", "**/*.parquet")), output_root)
    plan = plan_files(
        files,
        input_root,
        split_names,
        str(dataset_cfg.get("query_column", "query")),
        str(dataset_cfg.get("options_column", "options")),
    )
    for item in plan:
        item["input_root"] = input_root
        item["relative_path"] = item["path"].relative_to(input_root).as_posix()

    # Fail early if a scale folder does not actually contain the expected splits.
    observed_splits = {item["split"] for item in plan}
    expected_splits = {str(name).lower() for name in split_names}
    missing_splits = expected_splits - observed_splits
    if missing_splits:
        print(
            "WARNING: no Parquet files detected for existing split(s): "
            + ", ".join(sorted(missing_splits))
            + ". These split directories will not be fabricated or filled from other scales.",
            file=sys.stderr,
        )

    selected_total = sum(int(item["rows"]) for item in plan)
    split_rows = Counter()
    for item in plan:
        split_rows[item["split"]] += int(item["rows"])

    fingerprint_payload = build_plan_fingerprint(
        scale, input_root, plan, cfg, compact_dir_name, dictionary_filename
    )
    signature = json_fingerprint(fingerprint_payload)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "analysis_state.json"
    db_path = output_root / "sentence_analysis.sqlite"

    if any(output_root.iterdir()):
        if overwrite:
            shutil.rmtree(output_root)
            output_root.mkdir(parents=True, exist_ok=True)
        else:
            if not state_path.is_file():
                raise SystemExit(
                    f"Output already exists without resumable state: {output_root}\n"
                    "It may have been produced by the older scale-mixing implementation. "
                    "Back up anything needed, then rerun with --overwrite."
                )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("signature") != signature:
                raise SystemExit(
                    f"Input/config differs from existing output: {output_root}\n"
                    "If this output was created before scale-specific input selection was fixed, "
                    "rerun with --overwrite to rebuild it from only the requested scale."
                )
            if state.get("complete"):
                print(f"ID conversion already complete: {output_root}")
                print("No work needed. Use --overwrite to rebuild this scale.")
                return 0
            if not resume_enabled:
                raise SystemExit(
                    f"Partial output exists but analysis.resume is false: {output_root}\n"
                    "Set analysis.resume: true or pass --overwrite."
                )
            if not db_path.is_file():
                raise SystemExit(
                    f"Resume state exists but SQLite checkpoint is missing: {db_path}\n"
                    "Pass --overwrite to rebuild."
                )

    if not state_path.exists():
        atomic_write_json(
            state_path,
            {
                "signature": signature,
                "plan": fingerprint_payload,
                "first_pass_complete": False,
                "completed_compact_files": [],
                "complete": False,
            },
        )

    print(f"Sage root:       {sage_root}")
    print(f"Input scale:     {input_root}")
    print(f"Analysis output: {output_root}")
    print(f"Scale:           {scale}")
    print(f"Input Parquet:   {len(plan):,} shards")
    print(f"Selected rows:   {selected_total:,} (all rows in this scale; no re-splitting)")
    print("Existing rows by split:", json.dumps(dict(sorted(split_rows.items())), sort_keys=True))
    print("Split quotas:    none; original split assignments are preserved")

    conn = create_database(db_path, int(analysis_cfg.get("sqlite_cache_mb", 256)))
    try:
        totals = first_pass(plan, conn, cfg, batch_rows)
        totals["input_files"] = len(plan)
        totals["source_rows"] = selected_total
        totals["source_rows_by_split"] = dict(sorted(split_rows.items()))
        totals["unique_sentences"] = int(conn.execute("SELECT COUNT(*) FROM sentence").fetchone()[0])
        totals["scale"] = scale
        totals["row_cap"] = None
        totals["splits_reallocated"] = False

        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["first_pass_complete"] = True
        atomic_write_json(state_path, state)

        print("PASS 2/2: writing ID-coded Parquet shards (original split paths preserved)")
        lookup = IdLookup(conn, int(analysis_cfg.get("id_lookup_cache_entries", 500_000)))
        completed_paths = set(state.get("completed_compact_files", []))
        compact_files = 0
        compact_bytes = 0
        output_rows = 0

        for index, item in enumerate(plan, 1):
            relative = Path(item["relative_path"])
            expected_path = output_root / compact_dir_name / relative
            valid_existing = False
            if expected_path.is_file():
                try:
                    valid_existing = int(pq.ParquetFile(expected_path).metadata.num_rows) == int(item["rows"])
                except Exception:
                    valid_existing = False
            if valid_existing:
                out_path = expected_path
                row_count = int(item["rows"])
                print(f"  [{index}/{len(plan)}] RESUME {relative} | rows={row_count:,}")
            else:
                out_path, row_count = compact_shard(
                    item, input_root, output_root, conn, cfg, batch_rows, lookup, compact_dir_name
                )
                print(f"  [{index}/{len(plan)}] compacted {relative} | rows={row_count:,}")
            if row_count != int(item["rows"]):
                raise RuntimeError(f"Output row count mismatch for {relative}: {row_count} != {item['rows']}")
            completed_paths.add(relative.as_posix())
            state["completed_compact_files"] = sorted(completed_paths)
            atomic_write_json(state_path, state)
            compact_files += 1
            compact_bytes += out_path.stat().st_size
            output_rows += row_count

        if output_rows != selected_total:
            raise RuntimeError(f"Total compact row count mismatch: {output_rows:,} != {selected_total:,}")

        dict_bytes, csv_bytes = export_dictionary(
            conn,
            output_root,
            str(dataset_cfg.get("parquet_compression", "zstd")),
            bool(analysis_cfg.get("export_dictionary_csv", False)),
            str(dataset_cfg.get("sentence_id_type", "int32")),
            dictionary_filename,
        )
        # `totals` is a Counter: Counter.update(mapping) treats mapping values as
        # numeric counts, so nested dictionaries/strings trigger TypeError. Build
        # a normal dict for heterogeneous summary metadata instead.
        summary = dict(totals)
        summary.update(
            {
                "compact_files": compact_files,
                "compact_rows": output_rows,
                "compact_parquet_bytes": compact_bytes,
                "dictionary_parquet_bytes": dict_bytes,
                "dictionary_csv_bytes": csv_bytes,
                "id_type": dataset_cfg.get("sentence_id_type", "int32"),
                "normalization": dataset_cfg.get("normalize", {}),
                "query_column_replaced_by": dataset_cfg.get("query_id_column", "query_sentence_id"),
                "options_column_replaced_by": dataset_cfg.get("option_ids_column", "option_sentence_ids"),
                "dictionary_filename": dictionary_filename,
                "compact_root": str((output_root / compact_dir_name).resolve()),
            }
        )
        atomic_write_json(output_root / "summary.json", summary)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["complete"] = True
        state["summary"] = "summary.json"
        atomic_write_json(state_path, state)

        print("\nID conversion complete")
        print(f"  Scale:               {scale}")
        print(f"  Source rows:         {selected_total:,}")
        print(f"  Output rows:         {output_rows:,}")
        print(f"  Rows by split:       {json.dumps(dict(sorted(split_rows.items())), sort_keys=True)}")
        print(f"  Unique sentences:    {totals['unique_sentences']:,}")
        print(f"  Compact Parquet:     {compact_bytes / 1024**2:.2f} MiB")
        print(f"  Dictionary Parquet:  {dict_bytes / 1024**2:.2f} MiB")
        print(f"  Summary:             {output_root / 'summary.json'}")
    finally:
        conn.close()

    if not bool(analysis_cfg.get("keep_sqlite_database", False)):
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(db_path) + suffix).unlink(missing_ok=True)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
