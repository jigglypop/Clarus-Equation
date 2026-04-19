"""Bridge for running ClarusLM checkpoints from the runtime daemon path.

Supports two tokenizer modes:
  - char-level (stoi/itos dicts) -- legacy character-level checkpoints
  - BPE (PreTrainedTokenizerFast) -- GPT-2 / KoGPT-2 artifacts
"""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch


_CLM_CLASS = None

_REQUIRED_CKPT_KEYS = ("config", "model")
_REQUIRED_CONFIG_KEYS = ("vocab_size",)
_OPTIONAL_CONFIG_KEYS = {
    "dim": 256,
    "n_layers": 6,
    "n_heads": 8,
    "max_seq_len": 512,
    "ffn_hidden_dim": None,
    "mix_rank": None,
    "lambda_curv": 0.0,
    "lambda_mix": 0.0,
    "sparsity": 1.0,
}


def _load_clarus_lm_class():
    global _CLM_CLASS
    if _CLM_CLASS is not None:
        return _CLM_CLASS

    try:
        from clarus_lm import ClarusLM
        _CLM_CLASS = ClarusLM
        return _CLM_CLASS
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "examples" / "ai" / "clarus_lm.py"
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("clarus_lm_example", str(module_path))
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    _CLM_CLASS = getattr(module, "ClarusLM", None)
    return _CLM_CLASS


# ---------------------------------------------------------------------------
# Tokenizer abstraction
# ---------------------------------------------------------------------------

class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...


class CharTokenizer:
    """Character-level tokenizer from stoi/itos dicts."""

    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        self._stoi = stoi
        self._itos = itos
        self._fallback = stoi.get(" ", 0)

    def encode(self, text: str) -> list[int]:
        return [self._stoi.get(c, self._fallback) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos.get(int(i), "?") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)


class HFTokenizerWrapper:
    """Wraps a PreTrainedTokenizerFast for the same interface."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self._tok)


# ---------------------------------------------------------------------------
# Checkpoint validation
# ---------------------------------------------------------------------------

def _coerce_char_tokenizer(tokenizer_data: Any) -> CharTokenizer | None:
    if not isinstance(tokenizer_data, dict):
        return None
    stoi = tokenizer_data.get("stoi")
    itos_raw = tokenizer_data.get("itos")
    if not isinstance(stoi, dict) or not isinstance(itos_raw, dict):
        return None
    itos: dict[int, str] = {}
    for key, value in itos_raw.items():
        try:
            itos[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return CharTokenizer(stoi, itos)


def _load_hf_tokenizer(tokenizer_json: str, specials: dict | None = None) -> HFTokenizerWrapper | None:
    try:
        from tokenizers import Tokenizer as BackendTokenizer
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        return None

    backend = BackendTokenizer.from_str(tokenizer_json)
    kwargs = {}
    if specials:
        kwargs = {k: v for k, v in specials.items() if v is not None}
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)
    return HFTokenizerWrapper(tok)


def validate_checkpoint(ckpt: dict, *, path: str = "<unknown>") -> list[str]:
    """Return a list of warning/error strings for checkpoint issues."""
    issues: list[str] = []

    for key in _REQUIRED_CKPT_KEYS:
        if key not in ckpt:
            issues.append(f"[ERROR] missing top-level key '{key}' in {path}")

    has_char_tok = isinstance(ckpt.get("tokenizer"), dict) and "stoi" in ckpt.get("tokenizer", {})
    has_hf_tok = isinstance(ckpt.get("tokenizer_json"), str)
    if not has_char_tok and not has_hf_tok:
        issues.append(f"[ERROR] no tokenizer found (need 'tokenizer' with stoi/itos or 'tokenizer_json') in {path}")

    config = ckpt.get("config")
    if isinstance(config, dict):
        for key in _REQUIRED_CONFIG_KEYS:
            if key not in config:
                issues.append(f"[ERROR] config missing required key '{key}' in {path}")

        for key, default in _OPTIONAL_CONFIG_KEYS.items():
            if key not in config:
                issues.append(
                    f"[WARN] config missing optional key '{key}', will use default={default}"
                )

        vocab_size = config.get("vocab_size")
        if vocab_size is not None and (not isinstance(vocab_size, int) or vocab_size < 1):
            issues.append(f"[ERROR] config.vocab_size={vocab_size} is invalid in {path}")

    model_state = ckpt.get("model")
    if isinstance(model_state, dict):
        if not model_state:
            issues.append(f"[ERROR] model state_dict is empty in {path}")
        if "tok_emb.weight" not in model_state:
            issues.append(
                f"[WARN] model state_dict has no 'tok_emb.weight' -- "
                f"may be incompatible (keys: {list(model_state.keys())[:5]}...)"
            )
    elif model_state is not None:
        issues.append(f"[WARN] model entry is not a state_dict (type={type(model_state).__name__})")

    if has_char_tok:
        tok_data = ckpt["tokenizer"]
        stoi = tok_data.get("stoi", {})
        itos = tok_data.get("itos", {})
        if isinstance(stoi, dict) and isinstance(itos, dict) and len(stoi) != len(itos):
            issues.append(f"[WARN] tokenizer stoi({len(stoi)}) and itos({len(itos)}) sizes differ")
        if isinstance(config, dict):
            vs = config.get("vocab_size")
            if isinstance(vs, int) and isinstance(stoi, dict) and len(stoi) != vs:
                issues.append(f"[WARN] tokenizer size({len(stoi)}) != config.vocab_size({vs})")

    return issues


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@dataclass
class ClarusLMGenerator:
    model: torch.nn.Module
    tokenizer: TokenizerLike
    device: torch.device

    @property
    def vocab_size(self) -> int:
        return int(self.model.tok_emb.weight.shape[0])

    @torch.no_grad()
    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 top_k: int, seed: int | None = None) -> str:
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [0]

        max_len = max(1, int(self.model.max_seq_len))
        if len(tokens) > max_len:
            tokens = tokens[-max_len:]

        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        gen_kwargs = dict(
            temperature=float(temperature),
            top_k=max(1, int(top_k)),
            c3_passes=1,
            c3_candidates=1,
        )
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(int(seed))
                out = self.model.generate(idx, int(max_tokens), **gen_kwargs)
        else:
            out = self.model.generate(idx, int(max_tokens), **gen_kwargs)

        if out.numel() <= idx.numel():
            return ""
        gen_ids = out[0, idx.shape[1]:].tolist()
        return self.tokenizer.decode(gen_ids).strip()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _build_model(config: dict, device: str) -> torch.nn.Module:
    model_cls = _load_clarus_lm_class()
    if model_cls is None:
        raise ImportError("unable to import ClarusLM implementation")

    return model_cls(
        vocab_size=int(config["vocab_size"]),
        dim=int(config.get("dim", _OPTIONAL_CONFIG_KEYS["dim"])),
        n_layers=int(config.get("n_layers", _OPTIONAL_CONFIG_KEYS["n_layers"])),
        n_heads=int(config.get("n_heads", _OPTIONAL_CONFIG_KEYS["n_heads"])),
        max_seq_len=int(config.get("max_seq_len", _OPTIONAL_CONFIG_KEYS["max_seq_len"])),
        ffn_hidden_dim=config.get("ffn_hidden_dim", _OPTIONAL_CONFIG_KEYS["ffn_hidden_dim"]),
        mix_rank=config.get("mix_rank", _OPTIONAL_CONFIG_KEYS["mix_rank"]),
        lambda_curv=float(config.get("lambda_curv", _OPTIONAL_CONFIG_KEYS["lambda_curv"])),
        lambda_mix=float(config.get("lambda_mix", _OPTIONAL_CONFIG_KEYS["lambda_mix"])),
        sparsity=float(config.get("sparsity", _OPTIONAL_CONFIG_KEYS["sparsity"])),
        bias=bool(config.get("bias", True)),
        dense=bool(config.get("dense", False)),
        act_fn=str(config.get("act_fn", "silu")),
    ).to(device)


def load_clarus_lm_generator(path: str, *, device: str = "cpu") -> ClarusLMGenerator:
    if not isinstance(path, str) or not path:
        raise ValueError("clarus_lm checkpoint path is required")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"clarus_lm checkpoint must be a dict, got {type(ckpt).__name__}")

    issues = validate_checkpoint(ckpt, path=path)
    errors = [i for i in issues if i.startswith("[ERROR]")]
    warns = [i for i in issues if i.startswith("[WARN]")]

    for w in warns:
        warnings.warn(w, stacklevel=2)
    if errors:
        raise ValueError(
            f"clarus_lm checkpoint has {len(errors)} error(s):\n" + "\n".join(errors)
        )

    config = ckpt["config"]
    model = _build_model(config, device)

    state = ckpt.get("model")
    if state is None:
        raise ValueError("clarus_lm checkpoint is missing model weights")
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(
            f"clarus_lm state_dict mismatch: {exc}\n"
            f"checkpoint keys ({len(state)}): {list(state.keys())[:8]}..."
        ) from exc
    model.eval()

    tokenizer: TokenizerLike
    if isinstance(ckpt.get("tokenizer_json"), str):
        hf = _load_hf_tokenizer(ckpt["tokenizer_json"], ckpt.get("tokenizer_specials"))
        if hf is None:
            raise ImportError("HF tokenizer requested but transformers/tokenizers not available")
        tokenizer = hf
    else:
        char = _coerce_char_tokenizer(ckpt.get("tokenizer"))
        if char is None:
            raise ValueError("clarus_lm checkpoint tokenizer could not be parsed")
        tokenizer = char

    return ClarusLMGenerator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
    )


def init_from_ce_artifact(
    ce_artifact_path: str,
    *,
    n_layers: int = 6,
    n_heads: int = 8,
    max_seq_len: int = 512,
    device: str = "cpu",
    save_path: str | None = None,
) -> ClarusLMGenerator:
    """Create a ClarusLM initialized with GPT-2 embeddings only.

    ClarusBlocks are randomly initialized. Use init_from_gpt2() to
    transfer ALL GPT-2 parameters.
    """
    ce = torch.load(ce_artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(ce, dict):
        raise ValueError("ce artifact must be a dict")

    dim = int(ce["d"])
    vocab = int(ce["vocab"])
    emb_weight = ce.get("emb_weight")
    pos_weight = ce.get("pos_weight")
    tokenizer_json = ce.get("tokenizer_json")
    tokenizer_specials = ce.get("tokenizer_specials")

    if emb_weight is None:
        raise ValueError("ce artifact has no emb_weight (wte)")
    if tokenizer_json is None:
        raise ValueError("ce artifact has no tokenizer_json")

    config = {
        "vocab_size": vocab,
        "dim": dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "max_seq_len": max_seq_len,
    }
    model = _build_model(config, "cpu")

    with torch.no_grad():
        model.tok_emb.weight.copy_(emb_weight[:vocab, :dim].float())
        if pos_weight is not None:
            src_len = pos_weight.shape[0]
            tgt_len = max_seq_len
            if src_len >= tgt_len:
                model.pos_emb.weight.copy_(pos_weight[:tgt_len, :dim].float())
            else:
                model.pos_emb.weight[:src_len].copy_(pos_weight[:, :dim].float())

    model = model.to(device)
    model.eval()

    tokenizer = _load_hf_tokenizer(tokenizer_json, tokenizer_specials)
    if tokenizer is None:
        raise ImportError("HF tokenizer required but transformers/tokenizers not available")

    generator = ClarusLMGenerator(
        model=model, tokenizer=tokenizer, device=torch.device(device),
    )
    if save_path:
        torch.save({
            "model": model.state_dict(),
            "config": config,
            "tokenizer_json": tokenizer_json,
            "tokenizer_specials": tokenizer_specials,
        }, save_path)
    return generator


# ---------------------------------------------------------------------------
# Full GPT-2 parameter transfer
# ---------------------------------------------------------------------------

def _copy_to_spectral_normed(target_module: torch.nn.Module, weight: torch.Tensor):
    """Copy weight into a spectral_norm'd linear layer."""
    if hasattr(target_module, "weight_orig"):
        target_module.weight_orig.copy_(weight)
    else:
        target_module.weight.copy_(weight)


def _transfer_lbo_from_ln(lbo: torch.nn.Module, ln_weight: torch.Tensor, ln_bias: torch.Tensor):
    """Transfer LayerNorm parameters into LBONorm.

    scale/bias <- ln weight/bias directly.
    h <- 0 so LBO starts as pure LayerNorm (V projection has no effect).
    """
    lbo.scale.copy_(ln_weight)
    lbo.bias.copy_(ln_bias)
    lbo.h.fill_(0.0)
    lbo.alpha_conf.fill_(0.01)


def _transfer_attention(clarus_attn, gpt2_c_attn_weight, gpt2_c_attn_bias,
                         gpt2_c_proj_weight, gpt2_c_proj_bias):
    """Transfer GPT-2 attention weights into ClarusAttention.

    GPT-2 c_attn: (dim, 3*dim) column-major -> ClarusAttention qkv: (3*dim, dim)
    GPT-2 c_proj: (dim, dim) column-major -> ClarusAttention proj: (dim, dim)

    ``proj`` is wrapped in ``spectral_norm`` per CE: the unitary constraint
    |det T|^2 <= 1 is the structural hallucination suppressor of ClarusLM
    (docs/6_뇌/agi.md). We copy GPT-2 weights into ``weight_orig``; the
    spectral parameterization then renormalizes them to sigma_max <= 1 at
    forward time. Resulting outputs differ from raw GPT-2 by exactly the
    margin the unitary constraint enforces -- this is the intended behavior.
    """
    clarus_attn.qkv.weight.copy_(gpt2_c_attn_weight.t())
    if clarus_attn.qkv.bias is not None:
        clarus_attn.qkv.bias.copy_(gpt2_c_attn_bias)
    else:
        clarus_attn.qkv.bias = torch.nn.Parameter(gpt2_c_attn_bias.clone())

    _copy_to_spectral_normed(clarus_attn.proj, gpt2_c_proj_weight.t())

    if gpt2_c_proj_bias is not None:
        if clarus_attn.proj.bias is not None:
            clarus_attn.proj.bias.copy_(gpt2_c_proj_bias)
        else:
            clarus_attn.proj.bias = torch.nn.Parameter(gpt2_c_proj_bias.clone())


def _transfer_ffn_dense(gauge: torch.nn.Module, gpt2_fc_weight, gpt2_fc_bias,
                         gpt2_proj_weight, gpt2_proj_bias):
    """Transfer GPT-2 MLP weights into GaugeLattice (dense mode).

    Direct copy: fc_up <- c_fc, fc_down <- c_proj. ``gauge.phi`` (LBONorm)
    is preserved per CE: it provides the Laplace-Beltrami diffusion
    (h - eta * Delta_g h) that carries the Clarus field Phi through the
    network and feeds the curvature regularizer. ``_transfer_lbo_from_ln``
    initializes phi as identity-equivalent (h=0, scale=1, bias=0) so it
    starts as a pass-through; CE adapts h during training/calibration.
    """
    gauge.fc_up.weight.copy_(gpt2_fc_weight.t().float())
    gauge.fc_up.bias.copy_(gpt2_fc_bias.float())
    gauge.fc_down.weight.copy_(gpt2_proj_weight.t().float())
    gauge.fc_down.bias.copy_(gpt2_proj_bias.float())
    _transfer_lbo_from_ln(gauge.phi, torch.ones(gauge.dim), torch.zeros(gauge.dim))


def _calibrate_spectral_norm(model: torch.nn.Module, n_steps: int = 3):
    """Settle spectral_norm u/v vectors by running dummy forward passes.

    spectral_norm performs one power iteration of sigma_max(W) per forward
    while ``training=True``. After weight transfer or reload the cached u/v
    are stale; a few power iterations bring sigma to convergence (`u| -> 1`).
    Empirically 3 steps suffice for GPT-2-derived ClarusLM checkpoints.
    """
    max_seq = min(int(model.max_seq_len), 32)
    vocab = int(model.tok_emb.weight.shape[0])
    was_training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(n_steps):
            dummy = torch.randint(0, vocab, (1, max_seq))
            model(dummy)
    if not was_training:
        model.eval()


_KOGPT2_TOKENIZER_DEFAULTS = {
    "bos_token": "</s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
}


def _load_compatible_tokenizer(model_name: str):
    """Load a fast tokenizer with sane special-token defaults per family.

    ``AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')`` returns a fast
    tokenizer whose ``bos/eos/unk`` collapse to ``<|endoftext|>`` (id=51200)
    and produces byte-fallback ids for Korean text. The repo ships the actual
    SentencePiece-style BPE; we just have to pass the canonical special-token
    strings explicitly. We do that here so `init_from_gpt2` works for KoGPT2
    out of the box without users having to know the override.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if "kogpt2" in model_name.lower():
        return PreTrainedTokenizerFast.from_pretrained(
            model_name, **_KOGPT2_TOKENIZER_DEFAULTS
        )
    return AutoTokenizer.from_pretrained(model_name)


def init_from_gpt2(
    model_name: str = "gpt2",
    *,
    max_seq_len: int = 1024,
    sparsity: float = 1.0,
    device: str = "cpu",
    save_path: str | None = None,
    tokenizer=None,
) -> ClarusLMGenerator:
    """Create a ClarusLM with ALL parameters transferred from a GPT-2 model.

    Args:
        model_name: HF Hub id (e.g. ``'gpt2'``, ``'skt/kogpt2-base-v2'``).
        max_seq_len: positional embedding crop / extension length.
        sparsity: TopK ratio for ClarusLM SiLU activations.
        device: target device for the returned generator.
        save_path: optional .pt path to persist the converted artifact.
        tokenizer: optional pre-loaded fast tokenizer; if ``None`` we pick a
            compatible loader per model family (see
            :func:`_load_compatible_tokenizer`). Passing your own tokenizer is
            the supported way to override special-token mappings.

    Weight mapping:
      wte              -> tok_emb (tied with head)
      wpe              -> pos_emb
      h[i].ln_1        -> blocks[i].norm1 (LBONorm, h=0 so acts as LayerNorm)
      h[i].attn.c_attn -> blocks[i].attn.qkv
      h[i].attn.c_proj -> blocks[i].attn.proj
      h[i].ln_2        -> blocks[i].norm2
      h[i].mlp.c_fc    -> blocks[i].ffn.fc_up      (dense=True)
      h[i].mlp.c_proj  -> blocks[i].ffn.fc_down    (dense=True)
      ln_f             -> norm
    """
    from transformers import AutoModelForCausalLM

    tok = tokenizer if tokenizer is not None else _load_compatible_tokenizer(model_name)
    gpt2 = AutoModelForCausalLM.from_pretrained(model_name)
    gpt2.eval()

    dim = gpt2.config.n_embd
    n_layers = gpt2.config.n_layer
    n_heads = gpt2.config.n_head
    vocab = gpt2.config.vocab_size

    config = {
        "vocab_size": vocab,
        "dim": dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "max_seq_len": max_seq_len,
        "sparsity": sparsity,
        "bias": True,
        "dense": True,
        "act_fn": "gelu",
        "ffn_hidden_dim": None,
        "mix_rank": None,
        "lambda_curv": 0.0,
        "lambda_mix": 0.0,
    }
    model = _build_model(config, "cpu")

    sd = gpt2.state_dict()

    with torch.no_grad():
        model.tok_emb.weight.copy_(sd["transformer.wte.weight"].float())

        wpe = sd.get("transformer.wpe.weight")
        if wpe is not None:
            src_len = wpe.shape[0]
            if src_len >= max_seq_len:
                model.pos_emb.weight.copy_(wpe[:max_seq_len].float())
            else:
                model.pos_emb.weight[:src_len].copy_(wpe.float())

        for i in range(n_layers):
            block = model.blocks[i]
            pfx = f"transformer.h.{i}"

            _transfer_lbo_from_ln(
                block.norm1,
                sd[f"{pfx}.ln_1.weight"].float(),
                sd[f"{pfx}.ln_1.bias"].float(),
            )

            _transfer_attention(
                block.attn,
                sd[f"{pfx}.attn.c_attn.weight"].float(),
                sd[f"{pfx}.attn.c_attn.bias"].float(),
                sd[f"{pfx}.attn.c_proj.weight"].float(),
                sd[f"{pfx}.attn.c_proj.bias"].float(),
            )

            _transfer_lbo_from_ln(
                block.norm2,
                sd[f"{pfx}.ln_2.weight"].float(),
                sd[f"{pfx}.ln_2.bias"].float(),
            )

            _transfer_ffn_dense(
                block.ffn,
                sd[f"{pfx}.mlp.c_fc.weight"].float(),
                sd[f"{pfx}.mlp.c_fc.bias"].float(),
                sd[f"{pfx}.mlp.c_proj.weight"].float(),
                sd[f"{pfx}.mlp.c_proj.bias"].float(),
            )

        _transfer_lbo_from_ln(
            model.norm,
            sd["transformer.ln_f.weight"].float(),
            sd["transformer.ln_f.bias"].float(),
        )

    model = model.to(device)
    _calibrate_spectral_norm(model)
    model.eval()

    backend_tok = tok.backend_tokenizer
    tokenizer_json = backend_tok.to_str()
    tokenizer_specials = {}
    for attr in ("pad_token", "eos_token", "bos_token", "unk_token", "mask_token", "sep_token"):
        val = getattr(tok, attr, None)
        if val:
            tokenizer_specials[attr] = val

    hf_tok = _load_hf_tokenizer(tokenizer_json, tokenizer_specials)
    if hf_tok is None:
        raise ImportError("HF tokenizer required")

    generator = ClarusLMGenerator(
        model=model, tokenizer=hf_tok, device=torch.device(device),
    )

    if save_path:
        torch.save({
            "model": model.state_dict(),
            "config": config,
            "tokenizer_json": tokenizer_json,
            "tokenizer_specials": tokenizer_specials,
        }, save_path)

    return generator
