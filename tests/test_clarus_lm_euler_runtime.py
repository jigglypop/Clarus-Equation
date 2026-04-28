from __future__ import annotations

import torch

from clarus.clarus_lm_runtime import _transfer_attention, load_clarus_lm_generator


def _char_tokenizer(vocab_size: int) -> dict[str, dict[str, object]]:
    chars = [chr(ord("a") + idx) for idx in range(vocab_size)]
    stoi = {char: idx for idx, char in enumerate(chars)}
    itos = {idx: char for idx, char in enumerate(chars)}
    return {"stoi": stoi, "itos": itos}


def test_clarus_lm_loader_registers_euler_minimal_attention_backend(tmp_path):
    from examples.ai.clarus_lm import ClarusLM

    config = {
        "vocab_size": 16,
        "dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 8,
        "ffn_hidden_dim": None,
        "mix_rank": None,
        "lambda_curv": 0.0,
        "lambda_mix": 0.0,
        "sparsity": 1.0,
        "bias": True,
        "dense": False,
        "act_fn": "silu",
        "attention_backend": "euler_minimal",
        "euler_head_types": "alibi",
        "euler_xi_init": 4.0,
        "euler_learnable_xi": False,
        "euler_rope_base": 10000.0,
        "use_abs_pos": False,
    }
    model = ClarusLM(**config)
    path = tmp_path / "euler_lm.pt"
    torch.save(
        {
            "config": config,
            "model": model.state_dict(),
            "tokenizer": _char_tokenizer(config["vocab_size"]),
        },
        path,
    )

    gen = load_clarus_lm_generator(str(path), device="cpu")

    assert gen.model.attention_backend == "euler_minimal"
    assert gen.model.use_abs_pos is False
    assert gen.model.blocks[0].attn.__class__.__name__ == "ClarusEulerMinimalAttention"
    assert gen.model.blocks[0].attn.inner._uniform_type == 1


def test_clarus_lm_standard_attention_remains_default():
    from examples.ai.clarus_lm import ClarusAttention, ClarusLM

    model = ClarusLM(vocab_size=16, dim=32, n_layers=1, n_heads=4, max_seq_len=8)

    assert model.attention_backend == "standard"
    assert model.use_abs_pos is True
    assert isinstance(model.blocks[0].attn, ClarusAttention)


def test_gpt2_attention_transfer_keeps_euler_minimal_biasless_qkv():
    from examples.ai.clarus_lm import ClarusLM

    model = ClarusLM(
        vocab_size=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
        attention_backend="euler_minimal",
        euler_head_types="alibi",
        use_abs_pos=False,
    )
    attn = model.blocks[0].attn

    with torch.no_grad():
        _transfer_attention(
            attn,
            torch.randn(32, 96),
            torch.randn(96),
            torch.randn(32, 32),
            torch.randn(32),
        )

    assert attn.qkv.bias is None
    assert attn.proj.bias is None
    assert "blocks.0.attn.inner.qkv.bias" not in model.state_dict()
    logits, loss = model(torch.randint(0, 16, (1, 4)))
    assert logits.shape == (1, 4, 16)
    assert loss is None
