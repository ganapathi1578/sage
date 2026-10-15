"""Dataset discovery, resumable Parquet-to-SQLite index, and video/dictionary lookup."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

import yaml

from sageqa.config import load_config
from sageqa.data.targets import resolve_target, target_mode_for_record
from services import settings

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - explained by health endpoint and README
    pq = None


class DatasetCatalog:
    def __init__(self) -> None:
        self.analysis_root = settings.ANALYSIS_ROOT
        self.index_root = settings.INDEX_ROOT
        self.index_root.mkdir(parents=True, exist_ok=True)
        self._state_lock = threading.RLock()
        self._states: dict[tuple[str, str], dict[str, Any]] = {}
        self._threads: dict[tuple[str, str], threading.Thread] = {}
        self._dict_lock = threading.RLock()
        self._dictionaries: dict[str, dict[int, str]] = {}
        self._cfg_lock = threading.Lock()
        self._training_cfg: dict[str, Any] | None = None

    def training_cfg(self) -> dict[str, Any]:
        with self._cfg_lock:
            if self._training_cfg is None:
                cfg, _ = load_config(settings.TRAINING_CONFIG)
                self._training_cfg = cfg
            return self._training_cfg

    def data_cfg(self) -> dict[str, Any]:
        return self.training_cfg()["data"]

    def available_scales(self) -> list[str]:
        found: list[str] = []
        if not self.analysis_root.is_dir():
            return []
        for path in self.analysis_root.iterdir():
            if path.is_dir() and (path / "compact").is_dir():
                found.append(path.name)
        preferred = [s for s in settings.SCALE_ORDER if s in found]
        return preferred + sorted(s for s in found if s not in preferred)

    def list_datasets(self) -> dict[str, Any]:
        datasets = []
        video_store_ready = settings.VIDEO_STORE.exists()
        scales = self.available_scales()
        try:
            index_data_cfg = self.data_cfg()
        except Exception:
            index_data_cfg = {}
        text_store_root_ready = any(settings.token_store_ready(scale, index_data_cfg) for scale in scales)
        for scale in scales:
            scale_root = self.analysis_root / scale
            split_rows: dict[str, int | None] = {}
            for split in settings.SPLITS:
                split_dir = scale_root / "compact" / split
                files = sorted(split_dir.rglob("*.parquet")) if split_dir.is_dir() else []
                if not files or pq is None:
                    split_rows[split] = 0 if not files else None
                    continue
                try:
                    split_rows[split] = sum(int(pq.ParquetFile(f).metadata.num_rows) for f in files)
                except Exception:
                    split_rows[split] = None
            dictionary = scale_root / "sentence_dictionary.parquet"
            token_store_ready = settings.token_store_ready(scale, index_data_cfg)
            datasets.append({
                "scale": scale,
                "splits": split_rows,
                "dictionary_ready": dictionary.is_file(),
                "token_embeddings_ready": token_store_ready,
                "video_features_ready": video_store_ready,
            })
        return {
            "datasets": datasets,
            "analysis_root": str(self.analysis_root),
            "video_features_ready": video_store_ready,
            "text_store_root_ready": text_store_root_ready,
        }

    def _split_root(self, scale: str, split: str) -> Path:
        if scale not in self.available_scales():
            raise FileNotFoundError(f"Scale {scale!r} is unavailable under {self.analysis_root}")
        if split not in settings.SPLITS:
            raise ValueError(f"Unsupported split {split!r}")
        path = self.analysis_root / scale / "compact" / split
        if not path.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {path}")
        return path

    @staticmethod
    def _fingerprint(files: list[Path], root: Path) -> str:
        h = hashlib.sha256()
        for path in files:
            st = path.stat()
            h.update(str(path.relative_to(root)).encode("utf-8"))
            h.update(f"\0{st.st_size}\0{st.st_mtime_ns}\n".encode("utf-8"))
        return h.hexdigest()

    def _database_path(self, scale: str, split: str) -> Path:
        safe_scale = "".join(ch for ch in scale if ch.isalnum() or ch in "-_")
        return self.index_root / safe_scale / f"{split}.sqlite3"

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-65536")
        conn.execute("PRAGMA busy_timeout=60000")
        return conn

    @staticmethod
    def _initialise(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS indexed_files (
                relative_path TEXT PRIMARY KEY, file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL, num_rows INTEGER NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                source_row INTEGER NOT NULL,
                video_id TEXT NOT NULL,
                query_sentence_id INTEGER NOT NULL,
                option_sentence_ids TEXT NOT NULL,
                target_labels TEXT NOT NULL,
                target_mode TEXT NOT NULL,
                metadata TEXT NOT NULL,
                task_type TEXT,
                reasoning_family TEXT,
                generator_family TEXT,
                truth_state TEXT,
                UNIQUE(source_file, source_row)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS video_counts (
                video_id TEXT PRIMARY KEY, num_examples INTEGER NOT NULL
            )"""
        )
        conn.commit()

    @staticmethod
    def _get_meta(conn: sqlite3.Connection, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else default

    @staticmethod
    def _set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def index_status(self, scale: str, split: str, *, start: bool = True) -> dict[str, Any]:
        split_root = self._split_root(scale, split)
        files = sorted(split_root.rglob("*.parquet")) if split_root.is_dir() else []
        if pq is None:
            return {"status": "failed", "error": "pyarrow is not installed in the Sage environment."}
        if not files:
            return {"status": "failed", "error": f"No Parquet files found under {split_root}"}
        total_rows = 0
        try:
            total_rows = sum(int(pq.ParquetFile(f).metadata.num_rows) for f in files)
        except Exception as exc:
            return {"status": "failed", "error": f"Could not read Parquet metadata: {exc}"}
        fingerprint = self._fingerprint(files, split_root)
        key = (scale, split)
        with self._state_lock:
            state = self._states.get(key)
            if state and state.get("fingerprint") == fingerprint:
                if state.get("status") == "building" or state.get("status") == "failed":
                    return dict(state)
                if state.get("status") == "complete":
                    return dict(state)
            db_path = self._database_path(scale, split)
            if db_path.is_file():
                try:
                    conn = self._connect(db_path)
                    completed = self._get_meta(conn, "complete") == "1"
                    db_fingerprint = self._get_meta(conn, "fingerprint")
                    rows = int(self._get_meta(conn, "processed_rows", "0"))
                    conn.close()
                    if completed and db_fingerprint == fingerprint:
                        state = {
                            "status": "complete", "scale": scale, "split": split,
                            "processed_rows": rows, "total_rows": total_rows,
                            "progress": 1.0, "fingerprint": fingerprint,
                        }
                        self._states[key] = state
                        return dict(state)
                except sqlite3.DatabaseError:
                    pass
            if not start:
                return {
                    "status": "not_started", "scale": scale, "split": split,
                    "processed_rows": 0, "total_rows": total_rows, "progress": 0.0,
                }
            thread = self._threads.get(key)
            if thread and thread.is_alive():
                return dict(state or {"status": "building", "scale": scale, "split": split, "processed_rows": 0, "total_rows": total_rows, "progress": 0.0})
            self._states[key] = {
                "status": "building", "scale": scale, "split": split,
                "processed_rows": 0, "total_rows": total_rows, "progress": 0.0,
                "fingerprint": fingerprint, "current_file": None,
                "started_at": time.time(),
            }
            thread = threading.Thread(
                target=self._build_index, args=(scale, split, split_root, files, fingerprint, total_rows),
                name=f"sage-index-{scale}-{split}", daemon=True,
            )
            self._threads[key] = thread
            thread.start()
            return dict(self._states[key])

    def _build_index(
        self, scale: str, split: str, split_root: Path, files: list[Path],
        fingerprint: str, total_rows: int,
    ) -> None:
        key = (scale, split)
        db_path = self._database_path(scale, split)
        inserted_this_run = 0
        try:
            # Drop stale or incomplete files only when their source fingerprint changed.
            if db_path.exists():
                conn = self._connect(db_path)
                self._initialise(conn)
                old_fingerprint = self._get_meta(conn, "fingerprint")
                conn.close()
                if old_fingerprint and old_fingerprint != fingerprint:
                    for suffix in ("", "-wal", "-shm"):
                        Path(str(db_path) + suffix).unlink(missing_ok=True)
            conn = self._connect(db_path)
            self._initialise(conn)
            self._set_meta(conn, "fingerprint", fingerprint)
            self._set_meta(conn, "complete", "0")
            conn.commit()

            data_cfg = self.data_cfg()
            columns = data_cfg.get("columns", {})
            target_cfg = data_cfg.get("target", {})
            video_col = str(columns.get("video_id", "video_id"))
            query_col = str(columns.get("query_id", "query_sentence_id"))
            options_col = str(columns.get("option_ids", "option_sentence_ids"))
            target_col = str(target_cfg.get("column", "labels"))
            metadata_cols = list(data_cfg.get("metadata_columns", ["task_type", "reasoning_family", "generator_family", "truth_state"]))
            required_core = {video_col, query_col, options_col, target_col}
            processed_rows = int(self._get_meta(conn, "processed_rows", "0"))

            for file_index, path in enumerate(files, 1):
                relative = str(path.relative_to(split_root))
                st = path.stat()
                prior = conn.execute(
                    "SELECT file_size,mtime_ns,num_rows FROM indexed_files WHERE relative_path=?",
                    (relative,),
                ).fetchone()
                if prior and int(prior["file_size"]) == st.st_size and int(prior["mtime_ns"]) == st.st_mtime_ns:
                    with self._state_lock:
                        self._states[key].update({
                            "processed_rows": processed_rows,
                            "progress": min(0.999, processed_rows / max(1, total_rows)),
                            "current_file": relative,
                            "files_done": file_index,
                            "files_total": len(files),
                        })
                    continue

                parquet = pq.ParquetFile(path)
                schema_names = set(parquet.schema_arrow.names)
                missing = sorted(required_core - schema_names)
                if missing:
                    raise ValueError(f"Required columns {missing} are missing from {relative}; found {sorted(schema_names)}")
                # Training treats metadata columns as optional. Store those which exist
                # in this shard and leave absent metadata as null for filtering.
                available_metadata = [name for name in metadata_cols if name in schema_names]
                read_columns = sorted(required_core | set(available_metadata))
                file_video_counts: dict[str, int] = {}
                file_rows = 0
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute("DELETE FROM examples WHERE source_file=?", (relative,))
                    for batch in parquet.iter_batches(batch_size=2048, columns=read_columns):
                        records = batch.to_pylist()
                        values = []
                        for local_row, row in enumerate(records, start=file_rows):
                            video_value = row.get(video_col)
                            if video_value is None or not str(video_value).strip():
                                raise ValueError(f"Missing video ID at {relative}:{local_row}")
                            option_ids = row.get(options_col)
                            if not isinstance(option_ids, (list, tuple)) or not option_ids:
                                raise ValueError(f"Options must be a non-empty list at {relative}:{local_row}")
                            target_mode = target_mode_for_record(row, target_cfg)
                            labels = resolve_target(row.get(target_col), list(option_ids), target_cfg, row)
                            metadata = {name: row.get(name) for name in metadata_cols if name in row}
                            video_id = str(video_value)
                            qid = 0 if row.get(query_col) is None else int(row[query_col])
                            option_ids = [0 if item is None else int(item) for item in option_ids]
                            task_type = str(metadata.get("task_type") or "") or None
                            reasoning_family = str(metadata.get("reasoning_family") or "") or None
                            generator_family = str(metadata.get("generator_family") or "") or None
                            truth_state = str(metadata.get("truth_state") or "") or None
                            values.append((
                                relative, local_row, video_id, qid,
                                json.dumps(option_ids, separators=(",", ":")),
                                json.dumps(labels, separators=(",", ":")), target_mode,
                                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                                task_type, reasoning_family, generator_family, truth_state,
                            ))
                            file_video_counts[video_id] = file_video_counts.get(video_id, 0) + 1
                        conn.executemany(
                            """INSERT INTO examples(
                                source_file,source_row,video_id,query_sentence_id,option_sentence_ids,
                                target_labels,target_mode,metadata,task_type,reasoning_family,generator_family,truth_state
                            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                            values,
                        )
                        file_rows += len(records)
                        with self._state_lock:
                            self._states[key].update({
                                "processed_rows": processed_rows + file_rows,
                                "progress": min(0.999, (processed_rows + file_rows) / max(1, total_rows)),
                                "current_file": relative,
                                "files_done": file_index,
                                "files_total": len(files),
                            })
                    conn.executemany(
                        "INSERT INTO video_counts(video_id,num_examples) VALUES(?,?) "
                        "ON CONFLICT(video_id) DO UPDATE SET num_examples=num_examples+excluded.num_examples",
                        list(file_video_counts.items()),
                    )
                    conn.execute(
                        "INSERT INTO indexed_files(relative_path,file_size,mtime_ns,num_rows) VALUES(?,?,?,?) "
                        "ON CONFLICT(relative_path) DO UPDATE SET file_size=excluded.file_size,mtime_ns=excluded.mtime_ns,num_rows=excluded.num_rows",
                        (relative, st.st_size, st.st_mtime_ns, file_rows),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                processed_rows += file_rows
                inserted_this_run += file_rows
                self._set_meta(conn, "processed_rows", processed_rows)
                conn.commit()

            conn.executescript(
                "CREATE INDEX IF NOT EXISTS idx_examples_video_id ON examples(video_id, id);"
                "CREATE INDEX IF NOT EXISTS idx_examples_task_type ON examples(task_type, id);"
                "CREATE INDEX IF NOT EXISTS idx_examples_reasoning ON examples(reasoning_family, id);"
                "CREATE INDEX IF NOT EXISTS idx_examples_generator ON examples(generator_family, id);"
                "CREATE INDEX IF NOT EXISTS idx_examples_video_task ON examples(video_id, task_type, id);"
            )
            self._set_meta(conn, "processed_rows", conn.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
            self._set_meta(conn, "complete", "1")
            conn.commit()
            indexed_rows = int(conn.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
            conn.close()
            with self._state_lock:
                self._states[key] = {
                    "status": "complete", "scale": scale, "split": split,
                    "processed_rows": indexed_rows, "total_rows": total_rows,
                    "progress": 1.0, "fingerprint": fingerprint,
                    "elapsed_seconds": round(time.time() - float(self._states[key].get("started_at", time.time())), 2),
                }
        except Exception as exc:
            with self._state_lock:
                state = self._states.get(key, {})
                state.update({"status": "failed", "error": str(exc)[:1600], "scale": scale, "split": split})
                self._states[key] = state

    def _connect_ready(self, scale: str, split: str) -> sqlite3.Connection:
        path = self._database_path(scale, split)
        if not path.is_file():
            raise RuntimeError("Dataset index is not ready yet")
        conn = self._connect(path)
        if self._get_meta(conn, "complete") != "1":
            conn.close()
            raise RuntimeError("Dataset index is still being built")
        return conn

    def get_indexed_videos(self, scale: str, split: str, *, search: str = "", limit: int = 250, offset: int = 0) -> dict[str, Any]:
        conn = self._connect_ready(scale, split)
        try:
            where = "WHERE video_id LIKE ?" if search else ""
            args: list[Any] = [f"%{search}%"] if search else []
            total = int(conn.execute(f"SELECT COUNT(*) FROM video_counts {where}", args).fetchone()[0])
            args.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
            rows = conn.execute(
                f"SELECT video_id,num_examples FROM video_counts {where} ORDER BY video_id LIMIT ? OFFSET ?", args
            ).fetchall()
            return {"videos": [dict(r) for r in rows], "total": total, "limit": max(1, min(int(limit), 500)), "offset": max(0, int(offset))}
        finally:
            conn.close()

    def list_filter_values(self, scale: str, split: str, video_id: str | None = None) -> dict[str, list[str]]:
        conn = self._connect_ready(scale, split)
        try:
            base_where = "WHERE video_id=?" if video_id else ""
            base_args = (video_id,) if video_id else ()
            result = {}
            for column, output in (("task_type", "task_types"), ("reasoning_family", "reasoning_families"), ("generator_family", "generator_families")):
                rows = conn.execute(
                    f"SELECT DISTINCT {column} FROM examples {base_where if not base_where else base_where + ' AND'} {column} IS NOT NULL AND {column}!='' ORDER BY {column}",
                    base_args,
                ).fetchall() if base_where else conn.execute(
                    f"SELECT DISTINCT {column} FROM examples WHERE {column} IS NOT NULL AND {column}!='' ORDER BY {column}"
                ).fetchall()
                result[output] = [str(r[0]) for r in rows]
            return result
        finally:
            conn.close()

    def query_examples(
        self, scale: str, split: str, *, video_id: str | None = None,
        task_type: str | None = None, reasoning_family: str | None = None,
        generator_family: str | None = None, limit: int = 10, offset: int = 0,
    ) -> dict[str, Any]:
        conn = self._connect_ready(scale, split)
        try:
            clauses: list[str] = []
            values: list[Any] = []
            for column, value in (("video_id", video_id), ("task_type", task_type), ("reasoning_family", reasoning_family), ("generator_family", generator_family)):
                if value and value != "all":
                    clauses.append(f"{column}=?")
                    values.append(value)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            count = int(conn.execute(f"SELECT COUNT(*) FROM examples{where}", values).fetchone()[0])
            values.extend([max(1, min(int(limit), 50)), max(0, int(offset))])
            rows = conn.execute(
                "SELECT id,video_id,query_sentence_id,option_sentence_ids,target_labels,target_mode,metadata,task_type,reasoning_family,generator_family,truth_state "
                f"FROM examples{where} ORDER BY id LIMIT ? OFFSET ?", values,
            ).fetchall()
            examples = []
            for row in rows:
                examples.append({
                    "example_id": int(row["id"]),
                    "video_id": row["video_id"],
                    "query_sentence_id": int(row["query_sentence_id"]),
                    "option_sentence_ids": json.loads(row["option_sentence_ids"]),
                    "target_labels": json.loads(row["target_labels"]),
                    "target_mode": row["target_mode"],
                    "metadata": json.loads(row["metadata"]),
                    "task_type": row["task_type"],
                    "reasoning_family": row["reasoning_family"],
                    "generator_family": row["generator_family"],
                    "truth_state": row["truth_state"],
                })
            return {"examples": examples, "total": count, "limit": max(1, min(int(limit), 50)), "offset": max(0, int(offset))}
        finally:
            conn.close()

    def get_example(self, scale: str, split: str, example_id: int) -> dict[str, Any]:
        conn = self._connect_ready(scale, split)
        try:
            row = conn.execute(
                "SELECT id,video_id,query_sentence_id,option_sentence_ids,target_labels,target_mode,metadata,task_type,reasoning_family,generator_family,truth_state FROM examples WHERE id=?",
                (int(example_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"Example {example_id} not found in {scale}/{split}")
            return {
                "example_id": int(row["id"]), "video_id": row["video_id"],
                "query_sentence_id": int(row["query_sentence_id"]),
                "option_sentence_ids": json.loads(row["option_sentence_ids"]),
                "target_labels": json.loads(row["target_labels"]), "target_mode": row["target_mode"],
                "metadata": json.loads(row["metadata"]), "task_type": row["task_type"],
                "reasoning_family": row["reasoning_family"], "generator_family": row["generator_family"],
                "truth_state": row["truth_state"],
            }
        finally:
            conn.close()

    def dictionary(self, scale: str) -> dict[int, str]:
        with self._dict_lock:
            if scale in self._dictionaries:
                return self._dictionaries[scale]
            path = self.analysis_root / scale / "sentence_dictionary.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Sentence dictionary not found: {path}")
            if pq is None:
                raise RuntimeError("pyarrow is required to read sentence dictionaries")
            mapping: dict[int, str] = {}
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=8192, columns=["sentence_id", "sentence"]):
                for sid, sentence in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                    if sid is not None and sentence is not None:
                        mapping[int(sid)] = str(sentence)
            self._dictionaries[scale] = mapping
            return mapping

    def add_texts(self, scale: str, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        mapping = self.dictionary(scale)
        out = []
        for ex in examples:
            query = mapping.get(int(ex["query_sentence_id"]), "[Missing query text]")
            options = [mapping.get(int(sid), "[Missing option text]") for sid in ex["option_sentence_ids"]]
            metadata = dict(ex.get("metadata") or {})
            # Do not disclose expected answers in the dataset-browser endpoint.
            out.append({
                "example_id": ex["example_id"], "video_id": ex["video_id"],
                "query_sentence_id": ex["query_sentence_id"],
                "question": query,
                "option_sentence_ids": ex["option_sentence_ids"],
                "options": options,
                "target_mode": ex["target_mode"],
                "metadata": metadata,
                "task_type": ex.get("task_type"),
                "reasoning_family": ex.get("reasoning_family"),
                "generator_family": ex.get("generator_family"),
            })
        return out

    def resolve_video_path(self, video_id: str) -> Path | None:
        """Resolve only inside explicitly configured video roots; reject traversal."""
        raw = unquote(str(video_id)).replace("\\", "/").strip()
        if not raw or raw.startswith("/") or ".." in PurePosixPath(raw).parts:
            return None
        rel = Path(*PurePosixPath(raw).parts)
        exts = settings.ALLOWED_VIDEO_EXTENSIONS
        candidates: list[Path] = []
        for root in settings.VIDEO_ROOTS:
            direct = root / rel
            candidates.append(direct)
            if direct.suffix.lower() not in exts:
                candidates.extend(root / rel.with_suffix(ext) for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"))
        # Use an extractor manifest source path only when it remains inside a configured root.
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved.is_file() and resolved.suffix.lower() in exts:
                if any(resolved == root or root in resolved.parents for root in settings.VIDEO_ROOTS):
                    return resolved
        return None
