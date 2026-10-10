#!/usr/bin/env python3
"""Run a small config-defined ablation sweep serially."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from sageqa.config import find_repo_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Sweep YAML under configs/training/sweeps")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sweep_path = Path(args.config).expanduser().resolve()
    raw = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}
    repo_root = find_repo_root(sweep_path.parent)
    base_config = (sweep_path.parent / str(raw["base_config"])).resolve()
    runs = raw.get("runs", [])
    if not runs:
        raise SystemExit("Sweep config has no runs")
    for index, run in enumerate(runs, 1):
        command = [sys.executable, "-m", "scripts.train", "--config", str(base_config)]
        for key, value in (run.get("overrides") or {}).items():
            yaml_value = yaml.safe_dump(value, default_flow_style=True).splitlines()[0]
            command.extend(["--set", f"{key}={yaml_value}"])
        print(f"\n[{index}/{len(runs)}] {run.get('name', 'unnamed')}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
