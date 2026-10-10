import torch

from sageqa.models.registry import build_model


def synthetic_batch(text_dim=32, video_dim=24, batch_size=2, t=3, p=4, k=3):
    torch.manual_seed(7)
    return {
        "video_features": torch.randn(batch_size, t, p, video_dim),
        "video_xy": torch.rand(batch_size, p, 2),
        "video_times": torch.arange(t, dtype=torch.float32)[None].repeat(batch_size, 1),
        "video_mask": torch.ones(batch_size, t, dtype=torch.bool),
        "query_tokens": torch.randn(batch_size, 5, text_dim),
        "query_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
        "option_tokens": torch.randn(batch_size, k, 6, text_dim),
        "option_token_mask": torch.ones(batch_size, k, 6, dtype=torch.bool),
        "option_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "target_index": torch.tensor([1, 0]),
        "video_ids": ["v0", "v1"],
        "query_sentence_ids": [1, 2],
        "option_sentence_ids": [[3, 4, 5], [6, 7, 0]],
        "metadata": [{"task_type": "spatial"}, {"task_type": "temporal"}],
    }


def test_llama_bidir_output_and_option_permutation():
    cfg = {"name": "llama_bidir", "video_dim": 24, "text_dim": 32, "d_model": 32,
           "num_layers": 2, "num_heads": 4, "ffn_hidden": 64, "dropout": 0.0,
           "option_chunk_size": 2}
    model = build_model(cfg).eval()
    batch = synthetic_batch()
    with torch.no_grad():
        scores = model(batch)
        perm = torch.tensor([2, 0, 1])
        batch_perm = dict(batch)
        batch_perm["option_tokens"] = batch["option_tokens"][:, perm]
        batch_perm["option_token_mask"] = batch["option_token_mask"][:, perm]
        batch_perm["option_mask"] = batch["option_mask"][:, perm]
        batch_perm["option_sentence_ids"] = [[row[i] for i in perm.tolist()] for row in batch["option_sentence_ids"]]
        scores_perm = model(batch_perm)
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores[0]).all()
    assert scores[1, 2] == torch.finfo(scores.dtype).min
    assert torch.allclose(scores_perm, scores[:, perm], atol=1e-5, rtol=1e-5)


def test_channel_vector_output_and_option_permutation():
    cfg = {"name": "channel_vector", "video_dim": 24, "text_dim": 32, "channels": 8,
           "vector_dim": 4, "num_layers": 2, "num_heads": 2, "ffn_channels": 12,
           "dropout": 0.0, "option_chunk_size": 1}
    model = build_model(cfg).eval()
    batch = synthetic_batch()
    with torch.no_grad():
        scores = model(batch)
        perm = torch.tensor([1, 2, 0])
        batch_perm = dict(batch)
        batch_perm["option_tokens"] = batch["option_tokens"][:, perm]
        batch_perm["option_token_mask"] = batch["option_token_mask"][:, perm]
        batch_perm["option_mask"] = batch["option_mask"][:, perm]
        scores_perm = model(batch_perm)
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores[0]).all()
    assert torch.allclose(scores_perm, scores[:, perm], atol=1e-5, rtol=1e-5)


def test_spatial_patch_storage_order_does_not_change_scores():
    cfg = {"name": "llama_bidir", "video_dim": 24, "text_dim": 32, "d_model": 32,
           "num_layers": 2, "num_heads": 4, "ffn_hidden": 64, "dropout": 0.0,
           "option_chunk_size": 2}
    model = build_model(cfg).eval()
    batch = synthetic_batch()
    with torch.no_grad():
        scores = model(batch)
        perm = torch.tensor([2, 0, 3, 1])
        permuted = dict(batch)
        permuted["video_features"] = batch["video_features"][:, :, perm]
        permuted["video_xy"] = batch["video_xy"][:, perm]
        scores_permuted = model(permuted)
    assert torch.allclose(scores, scores_permuted, atol=1e-5, rtol=1e-5)


def test_channel_vector_invariant_linear_attention_backward():
    cfg = {"name": "channel_vector", "video_dim": 24, "text_dim": 32, "channels": 8,
           "vector_dim": 4, "num_layers": 1, "num_heads": 2, "ffn_channels": 12,
           "dropout": 0.0, "option_chunk_size": 1, "attention": "invariant_linear"}
    model = build_model(cfg).train()
    batch = synthetic_batch()
    scores = model(batch)
    loss = torch.nn.functional.cross_entropy(scores, batch["target_index"])
    loss.backward()
    assert scores.shape == (2, 3)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
