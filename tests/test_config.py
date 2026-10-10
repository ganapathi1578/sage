from pathlib import Path

from sageqa.config import load_config


def test_experiment_config_composes():
    root = Path(__file__).resolve().parents[1]
    cfg, repo_root = load_config(root / "configs/training/experiments/llama_10k.yaml")
    assert cfg["model"]["name"] == "llama_bidir"
    assert cfg["data"]["scale"] == "10k"
    assert cfg["model"]["d_model"] == 384
    assert repo_root == root
