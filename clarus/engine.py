"""CE-AGI runtime aligned with the Equation spec.

Phase 1:
  CE relaxation on the converted Hopfield engine.

Phase 2:
  Decode the relaxed state through the source language model using
  direct / layer-inject / multiround modes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    from .ce_ops import (
        pack_sparse as ce_pack_sparse,
        pq_reconstruct_tokens,
        pq_scores,
    )
    from .hopfield import (
        block_hidden,
        build_codebook,
        decode_direct,
        decode_inject,
        generate_multiround,
        relax as hopfield_relax,
        resolve_device,
        safe_print,
        update_phi,
    )
except ImportError:
    from clarus.ce_ops import (
        pack_sparse as ce_pack_sparse,
        pq_reconstruct_tokens,
        pq_scores,
    )
    from clarus.hopfield import (
        block_hidden,
        build_codebook,
        decode_direct,
        decode_inject,
        generate_multiround,
        relax as hopfield_relax,
        resolve_device,
        safe_print,
        update_phi,
    )


DEFAULT_PROMPTS = (
    "인공지능의 미래는",
    "맛있는 음식을",
    "한국의 역사에서",
)


@dataclass
class PromptContext:
    prompt: str
    prompt_ids: torch.Tensor
    h_true: torch.Tensor | None
    m0: torch.Tensor
    phi: torch.Tensor
    best_layer: int
    layer_scores: dict[int, float]


class CEEngine:
    def __init__(self, path: str, device: str = "cpu", backend: str = "torch"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.data = data
        self.device = resolve_device(device)
        self.backend = backend

        self.model_name = data["model_name"]
        self.d = int(data["d"])
        self.vocab = int(data["vocab"])
        self.tau = float(data["tau"])
        self.portal = float(data["portal"])
        self.bypass = float(data["bypass"])
        self.t_wake = float(data["t_wake"])
        self.n_layer = int(data["n_layer"])
        self.h_norm_ref = float(data.get("hidden_norm_ref", 50.0))
        self.decoder_prev_scale = float(data.get("decoder_prev_scale", 0.35))
        self.decoder_prev_proj = None
        if data.get("decoder_prev_proj") is not None:
            self.decoder_prev_proj = data["decoder_prev_proj"].float().to(self.device)
        self.decoder_state_proj = None
        if data.get("decoder_state_proj") is not None:
            self.decoder_state_proj = data["decoder_state_proj"].float().to(self.device)
        self.decoder_token_ids = None
        if data.get("decoder_token_ids") is not None:
            self.decoder_token_ids = data["decoder_token_ids"].long().to(self.device)
        self.decoder_token_state_proj = None
        if data.get("decoder_token_state_proj") is not None:
            self.decoder_token_state_proj = data["decoder_token_state_proj"].float().to(self.device)
        self.decoder_token_prev_proj = None
        if data.get("decoder_token_prev_proj") is not None:
            self.decoder_token_prev_proj = data["decoder_token_prev_proj"].float().to(self.device)
        self.decoder_token_bias = None
        if data.get("decoder_token_bias") is not None:
            self.decoder_token_bias = data["decoder_token_bias"].float().to(self.device)
        self.decoder_token_scale = float(data.get("decoder_token_scale", 1.0))

        self.W = data["W"].float().to(self.device)
        self.W_pack = self._load_w_pack(data)
        self._dense_relax_w = None
        if self.W_pack[0].numel() == self.W.numel():
            self._dense_relax_w = self.W
        emb_weight = data.get("emb_weight")
        self.emb = emb_weight.float().to(self.device) if emb_weight is not None else None
        self.ln_w = data["ln_f_weight"].float().to(self.device)
        self.ln_b = data["ln_f_bias"].float().to(self.device)
        self.pq_centroids = None
        self.pq_codes = None
        if data.get("pq_centroids") is not None and data.get("pq_codes") is not None:
            self.pq_centroids = data["pq_centroids"].to(self.device)
            self.pq_codes = data["pq_codes"].to(self.device)

        self._stored_eigvecs = data.get("W_eigvecs")
        if self._stored_eigvecs is not None:
            self._stored_eigvecs = self._stored_eigvecs.float()
        self._eigvec_cache: dict[int, torch.Tensor] = {}

        self.model = None
        self.tok = None
        self.pad_token_id = data.get("pad_token_id")
        self.eos_token_id = data.get("eos_token_id")
        self.model_memory_bytes = 0
        self._load_model()

    def _load_w_pack(self, data):
        values = data.get("W_values")
        col_idx = data.get("W_col_idx")
        row_ptr = data.get("W_row_ptr")
        if values is None or col_idx is None or row_ptr is None:
            values, col_idx, row_ptr = ce_pack_sparse(data["W"].float(), backend="torch")
        return (
            values.to(self.device),
            col_idx.to(self.device),
            row_ptr.to(self.device),
        )

    def _load_model(self):
        from tokenizers import Tokenizer
        from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

        clone_config = self.data.get("clone_config")
        clone_state = self.data.get("clone_state")
        tokenizer_json = self.data.get("tokenizer_json")
        tokenizer_specials = self.data.get("tokenizer_specials") or {}

        if tokenizer_json is not None:
            backend_tok = Tokenizer.from_str(tokenizer_json)
            tok_kwargs = {k: v for k, v in tokenizer_specials.items() if v is not None}
            self.tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, **tok_kwargs)
        else:
            self.tok = PreTrainedTokenizerFast.from_pretrained(self.model_name)

        if clone_config is not None and clone_state is not None:
            cfg_dict = dict(clone_config)
            model_type = cfg_dict.pop("model_type", "gpt2")
            cfg = AutoConfig.for_model(model_type, **cfg_dict)
            self.model = AutoModelForCausalLM.from_config(cfg)
            state = {}
            for name, value in clone_state.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    state[name] = value.float()
                else:
                    state[name] = value
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected clone_state keys: {unexpected[:5]}")
            self.model_source = "artifact"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
            )
            self.model_source = "pretrained"
        self.model.to(self.device).eval()
        self.model_memory_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        if self.eos_token_id is None:
            self.eos_token_id = self.tok.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tok.pad_token_id
        cfg = self.model.config
        if hasattr(cfg, "n_layer"):
            self.n_layer = int(cfg.n_layer)
        else:
            self.n_layer = int(cfg.num_hidden_layers)
        self.inject_layer = self.n_layer // 2

    def _get_w_eigvecs(self, metric_rank: int) -> torch.Tensor | None:
        hess_rank = min(metric_rank // 2, 8)
        if hess_rank <= 0:
            return None
        if self._stored_eigvecs is not None and self._stored_eigvecs.shape[0] >= hess_rank:
            return self._stored_eigvecs[:hess_rank].to(self.device)
        if hess_rank not in self._eigvec_cache:
            _, eigvecs = torch.linalg.eigh(self.W.cpu())
            self._eigvec_cache[hess_rank] = eigvecs[:, :hess_rank].T.contiguous().to(self.device)
        return self._eigvec_cache[hess_rank]

    def memory_usage(self) -> dict[str, float]:
        values, col_idx, row_ptr = self.W_pack
        w_dense_bytes = self.W.numel() * self.W.element_size()
        w_packed_bytes = (
            values.numel() * values.element_size()
            + col_idx.numel() * col_idx.element_size()
            + row_ptr.numel() * row_ptr.element_size()
        )
        w_layers_bytes = sum(
            w.numel() * w.element_size() for w in self.data.get("W_layers", [])
        )
        emb_bytes = 0 if self.emb is None else self.emb.numel() * self.emb.element_size()
        pq_bytes = 0
        if self.pq_centroids is not None and self.pq_codes is not None:
            pq_bytes = (
                self.pq_centroids.numel() * self.pq_centroids.element_size()
                + self.pq_codes.numel() * self.pq_codes.element_size()
            )
        clone_artifact_bytes = 0
        clone_state = self.data.get("clone_state")
        if clone_state is not None:
            clone_artifact_bytes = sum(
                value.numel() * value.element_size() for value in clone_state.values()
            )
        prev_proj_bytes = (
            0 if self.decoder_prev_proj is None
            else self.decoder_prev_proj.numel() * self.decoder_prev_proj.element_size()
        )
        state_proj_bytes = (
            0 if self.decoder_state_proj is None
            else self.decoder_state_proj.numel() * self.decoder_state_proj.element_size()
        )
        token_head_bytes = 0
        for tensor in (
            self.decoder_token_ids,
            self.decoder_token_state_proj,
            self.decoder_token_prev_proj,
            self.decoder_token_bias,
        ):
            if tensor is not None:
                token_head_bytes += tensor.numel() * tensor.element_size()
        ln_bytes = (self.ln_w.numel() + self.ln_b.numel()) * self.ln_w.element_size()
        runtime_core = w_packed_bytes + ln_bytes + prev_proj_bytes + state_proj_bytes + token_head_bytes
        runtime_total = runtime_core + emb_bytes + pq_bytes
        file_core = w_dense_bytes + w_layers_bytes + ln_bytes + prev_proj_bytes + state_proj_bytes + token_head_bytes
        file_total = file_core + emb_bytes + pq_bytes + clone_artifact_bytes
        return {
            "W_dense_MB": w_dense_bytes / 1024 / 1024,
            "W_packed_MB": w_packed_bytes / 1024 / 1024,
            "W_layers_MB": w_layers_bytes / 1024 / 1024,
            "Embedding_MB": emb_bytes / 1024 / 1024,
            "PQ_MB": pq_bytes / 1024 / 1024,
            "CloneArtifact_MB": clone_artifact_bytes / 1024 / 1024,
            "PrevProj_MB": prev_proj_bytes / 1024 / 1024,
            "StateProj_MB": state_proj_bytes / 1024 / 1024,
            "TokenHead_MB": token_head_bytes / 1024 / 1024,
            "ln_f_KB": ln_bytes / 1024,
            "runtime_core_MB": runtime_core / 1024 / 1024,
            "runtime_total_MB": runtime_total / 1024 / 1024,
            "file_total_MB": file_total / 1024 / 1024,
            "model_MB": self.model_memory_bytes / 1024 / 1024,
        }

    def save_artifact(self, path: str):
        torch.save(self.data, path)

    def has_standalone_lexicon(self) -> bool:
        return self.emb is not None or (
            self.pq_centroids is not None and self.pq_codes is not None
        )

    def token_embedding(self, token_ids: int | list[int] | torch.Tensor) -> torch.Tensor:
        if self.emb is not None:
            if not torch.is_tensor(token_ids):
                token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
            token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
            return self.emb.index_select(0, token_ids)
        if self.pq_centroids is not None and self.pq_codes is not None:
            if not torch.is_tensor(token_ids):
                token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
            token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
            return pq_reconstruct_tokens(
                self.pq_centroids,
                self.pq_codes,
                token_ids,
            ).to(self.device)
        raise RuntimeError("No lexical memory is available for token embedding lookup")

    def lexical_scores(self, query: torch.Tensor) -> torch.Tensor:
        if self.emb is not None:
            return self.emb @ query
        if self.pq_centroids is not None and self.pq_codes is not None:
            return pq_scores(query, self.pq_centroids, self.pq_codes)
        raise RuntimeError("No lexical memory is available for scoring")

    def build_runtime_codebook(self, m_ref: torch.Tensor, top_k: int) -> torch.Tensor:
        if self.has_standalone_lexicon():
            scores = self.lexical_scores(m_ref)
            top_ids = torch.topk(scores, min(top_k, scores.numel())).indices
            return self.token_embedding(top_ids)
        return build_codebook(self.model, m_ref, top_k=top_k, verbose=False)

    def ce_hidden(self, m_star: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(m_star, (self.d,), self.ln_w, self.ln_b)

    def teacher_embedding(self, token_ids: torch.Tensor | list[int]) -> torch.Tensor:
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
        return self.model.transformer.wte.weight.index_select(0, token_ids)

    def teacher_next_logits(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(prompt_ids)
        with torch.no_grad():
            logits = self.model(prompt_ids, attention_mask=attention_mask).logits[:, -1, :]
        return logits.squeeze(0)

    def decoder_token_correction(
        self,
        ce_hidden: torch.Tensor,
        prev_emb: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.decoder_token_ids is None:
            return None
        correction = None
        if self.decoder_token_state_proj is not None:
            correction = ce_hidden @ self.decoder_token_state_proj
        if self.decoder_token_prev_proj is not None:
            prev_piece = self.decoder_prev_scale * (prev_emb @ self.decoder_token_prev_proj)
            correction = prev_piece if correction is None else correction + prev_piece
        if self.decoder_token_bias is not None:
            correction = self.decoder_token_bias if correction is None else correction + self.decoder_token_bias
        if correction is None:
            return None
        return self.decoder_token_scale * correction

    def standalone_logits(
        self,
        ce_hidden: torch.Tensor,
        prev_id: int,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        repeat_ids: list[int] | None = None,
        repeat_penalty: float = 3.0,
    ) -> torch.Tensor:
        prev_emb = self.token_embedding([prev_id]).squeeze(0)
        state_query = ce_hidden @ self.decoder_state_proj if self.decoder_state_proj is not None else ce_hidden
        if self.decoder_prev_proj is not None:
            prev_query = prev_emb @ self.decoder_prev_proj
        else:
            prev_query = prev_emb
        query = state_query + self.decoder_prev_scale * prev_query
        logits = self.lexical_scores(query)
        correction = self.decoder_token_correction(ce_hidden, prev_emb)
        if correction is not None:
            logits = logits.clone()
            logits[self.decoder_token_ids] += correction
        logits = logits / max(temperature, 1e-6)
        if repeat_ids:
            logits = logits.clone()
            for seen_id in set(repeat_ids):
                logits[seen_id] -= repeat_penalty
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits = logits.clone()
            logits[logits < v[-1]] = float("-inf")
        return logits

    def apply_decoder_refine(self, prev_proj: torch.Tensor, state_proj: torch.Tensor):
        self.decoder_prev_proj = prev_proj.to(self.device)
        self.decoder_state_proj = state_proj.to(self.device)
        self.data["decoder_prev_proj"] = prev_proj.detach().cpu()
        self.data["decoder_state_proj"] = state_proj.detach().cpu()

    def apply_token_head(
        self,
        token_ids: torch.Tensor,
        *,
        state_proj: torch.Tensor | None = None,
        prev_proj: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        scale: float | None = None,
    ):
        self.decoder_token_ids = token_ids.long().to(self.device)
        self.decoder_token_state_proj = None if state_proj is None else state_proj.float().to(self.device)
        self.decoder_token_prev_proj = None if prev_proj is None else prev_proj.float().to(self.device)
        self.decoder_token_bias = None if bias is None else bias.float().to(self.device)
        if scale is not None:
            self.decoder_token_scale = float(scale)
        self.data["decoder_token_ids"] = token_ids.detach().cpu().long()
        self.data["decoder_token_state_proj"] = None if state_proj is None else state_proj.detach().cpu()
        self.data["decoder_token_prev_proj"] = None if prev_proj is None else prev_proj.detach().cpu()
        self.data["decoder_token_bias"] = None if bias is None else bias.detach().cpu()
        self.data["decoder_token_scale"] = float(self.decoder_token_scale)

    def standalone_generate(
        self,
        prompt_ids: torch.Tensor,
        m_star: torch.Tensor,
        *,
        max_tok: int,
        temperature: float,
        top_k: int,
        repeat_penalty: float,
        refresh_interval: int = 0,
        refresh_args=None,
        refresh_init_layer: int | None = None,
        refresh_phi: torch.Tensor | None = None,
    ) -> tuple[str, list[int], dict[str, float | int | None]]:
        if not self.has_standalone_lexicon():
            raise RuntimeError("Standalone decoder requires embeddings or PQ lexical memory")

        refresh_interval = max(int(refresh_interval), 0)
        h = F.layer_norm(m_star, (self.d,), self.ln_w, self.ln_b)
        prev_id = int(prompt_ids[0, -1].item())
        running_ids = prompt_ids.clone()
        out_ids: list[int] = []
        phi_state = None if refresh_phi is None else refresh_phi.detach().clone().to(self.device)
        init_layer = refresh_init_layer
        refresh_count = 0
        refresh_steps = 0
        refresh_time_s = 0.0
        refresh_cos: list[float] = []

        for _ in range(max_tok):
            logits = self.standalone_logits(
                h,
                prev_id,
                temperature=temperature,
                top_k=top_k,
                repeat_ids=out_ids[-6:],
                repeat_penalty=repeat_penalty,
            )
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break
            out_ids.append(next_id)
            prev_id = next_id
            next_token = torch.tensor([[next_id]], device=self.device)
            running_ids = torch.cat([running_ids, next_token], dim=1)
            if (
                refresh_interval > 0
                and refresh_args is not None
                and init_layer is not None
                and len(out_ids) < max_tok
                and len(out_ids) % refresh_interval == 0
            ):
                refresh_ctx = self.context_from_ids(
                    running_ids,
                    init_layer=init_layer,
                    phi=phi_state,
                    need_teacher=False,
                )
                refresh_result = self.relax_context(refresh_ctx, refresh_args)
                h = self.ce_hidden(refresh_result["m_star"])
                phi_state = refresh_result["phi_updated"]
                init_layer = refresh_ctx.best_layer
                refresh_count += 1
                refresh_steps += int(refresh_result["steps"])
                refresh_time_s += float(refresh_result["elapsed_s"])
                if refresh_result["cos_ms_h"] is not None:
                    refresh_cos.append(float(refresh_result["cos_ms_h"]))

        meta = {
            "refresh_interval": refresh_interval,
            "refresh_count": refresh_count,
            "refresh_steps": refresh_steps,
            "refresh_time_s": refresh_time_s,
            "refresh_cos_mean": None if not refresh_cos else sum(refresh_cos) / len(refresh_cos),
            "refresh_phi_norm": None if phi_state is None else float(phi_state.norm().item()),
        }
        return self.tok.decode(out_ids, skip_special_tokens=True), out_ids, meta

    def legacy_generate(
        self,
        token_ids: list[int],
        *,
        n_tokens: int,
        residual_scale: float,
        temperature: float,
    ) -> list[int]:
        if self.emb is None or not self.data.get("W_layers"):
            raise RuntimeError("Legacy generator requires full embedding table and W_layers")

        generated = list(token_ids)
        used = set()
        m = self.emb[token_ids].mean(dim=0)
        phi = self.emb.var(dim=0).clamp(min=1e-8).sqrt()

        for _ in range(n_tokens):
            m_out = m.clone()
            for w_layer in self.data["W_layers"]:
                w_dev = w_layer.to(self.device)
                delta = w_dev @ m_out
                m_out = m_out + residual_scale * delta
                m_out = m_out / (m_out.norm() + 1e-8) * m.norm()
            m_out = m_out * (self.h_norm_ref / (m_out.norm() + 1e-8))

            phi_hat = F.normalize(phi, dim=0)
            m_out = m_out + self.portal * phi_hat * self.h_norm_ref
            m_out = m_out + self.bypass * phi

            h = F.layer_norm(m_out, (self.d,), self.ln_w, self.ln_b)
            logits = h @ self.emb.T
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            candidates = torch.topk(probs, 50)

            next_id = None
            for cid in candidates.indices.tolist():
                if cid not in used:
                    next_id = cid
                    break
            if next_id is None:
                next_id = int(candidates.indices[0].item())

            generated.append(next_id)
            used.add(next_id)
            new_emb = self.emb[next_id]
            m = 0.3 * m + 0.7 * new_emb

        return generated

    def prompt_context(self, prompt: str) -> PromptContext:
        prompt_ids = self.tok.encode(prompt, return_tensors="pt").to(self.device)
        return self.context_from_ids(prompt_ids, prompt=prompt)

    def _analyze_prompt_ids(
        self,
        prompt_ids: torch.Tensor,
        *,
        candidate_layers: list[int],
        need_teacher: bool,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor | None]:
        with torch.no_grad():
            seq_len = prompt_ids.shape[1]
            pos_ids = torch.arange(seq_len, device=prompt_ids.device).unsqueeze(0)
            emb = self.model.transformer.wte(prompt_ids) + self.model.transformer.wpe(pos_ids)
            phi_base = emb.squeeze(0).var(dim=0)

            capture = sorted(set(int(layer) for layer in candidate_layers))
            h = emb
            captured: dict[int, torch.Tensor] = {}
            target_layer = capture[-1] if capture else -1
            if need_teacher:
                target_layer = max(target_layer, self.n_layer - 1)

            for layer_idx in range(target_layer + 1):
                h = block_hidden(self.model.transformer.h[layer_idx], h)
                if layer_idx in capture:
                    captured[layer_idx] = h[:, -1, :].squeeze(0).clone()

            h_true = None
            if need_teacher:
                h_true = self.model.transformer.ln_f(h)[:, -1, :]

        return phi_base, captured, h_true

    def context_from_ids(
        self,
        prompt_ids: torch.Tensor,
        prompt: str | None = None,
        *,
        init_layer: int | None = None,
        phi: torch.Tensor | None = None,
        need_teacher: bool = True,
    ) -> PromptContext:
        h_true = None
        if need_teacher:
            candidate_layers = (
                [int(init_layer)]
                if init_layer is not None
                else sorted({
                    0,
                    self.n_layer // 4,
                    self.n_layer // 2,
                    (3 * self.n_layer) // 4,
                    self.n_layer - 1,
                })
            )
            phi_base, captured, h_true = self._analyze_prompt_ids(
                prompt_ids,
                candidate_layers=candidate_layers,
                need_teacher=True,
            )
        elif init_layer is None:
            raise ValueError("need_teacher=False requires a fixed init_layer")
        else:
            phi_base, captured, h_true = self._analyze_prompt_ids(
                prompt_ids,
                candidate_layers=[int(init_layer)],
                need_teacher=False,
            )

        layer_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        layer_scores: dict[int, float] = {}
        if init_layer is not None:
            best_layer = int(init_layer)
            m0 = captured[best_layer]
            score = float("nan") if h_true is None else F.cosine_similarity(m0.unsqueeze(0), h_true).item()
            layer_states[best_layer] = (
                m0,
                phi_base if phi is None else phi.detach().float().to(self.device),
            )
            layer_scores[best_layer] = score
        else:
            candidates = sorted({
                0,
                self.n_layer // 4,
                self.n_layer // 2,
                (3 * self.n_layer) // 4,
                self.n_layer - 1,
            })
            for candidate_layer in candidates:
                m0 = captured[candidate_layer]
                score = F.cosine_similarity(m0.unsqueeze(0), h_true).item()
                layer_states[candidate_layer] = (m0, phi_base)
                layer_scores[candidate_layer] = score
            best_layer = max(layer_scores, key=layer_scores.get)
        m0, phi = layer_states[best_layer]
        return PromptContext(
            prompt=prompt if prompt is not None else self.tok.decode(prompt_ids[0], skip_special_tokens=True),
            prompt_ids=prompt_ids,
            h_true=h_true,
            m0=m0,
            phi=phi,
            best_layer=best_layer,
            layer_scores=layer_scores,
        )

    def relax_context(self, ctx: PromptContext, args):
        dt_eff = min(float(args.dt), 0.9 * self.tau)
        cb_weight = self.portal if args.cb_weight is None else float(args.cb_weight)
        codebook = self.build_runtime_codebook(ctx.m0, top_k=args.cb_topk)
        t0 = time.time()
        m_star, hist, n_steps = hopfield_relax(
            self.W_pack,
            ctx.m0,
            ctx.phi,
            ctx.m0,
            codebook,
            args.beta,
            cb_weight,
            tau=self.tau,
            dt=dt_eff,
            max_steps=args.steps,
            backend=args.backend,
            metric_rank=args.metric_rank,
            lambda0=args.lambda0,
            lambda_phi=args.lambda_phi,
            lambda_var=args.lambda_var,
            noise_scale=args.noise_scale,
            seed=args.seed,
            w_eigvecs=self._get_w_eigvecs(args.metric_rank),
            dense_w=self._dense_relax_w,
        )
        elapsed = time.time() - t0
        cos_ms = None
        if ctx.h_true is not None:
            cos_ms = F.cosine_similarity(m_star.unsqueeze(0), ctx.h_true).item()
        phi_var = hist.get("phi_var")
        if phi_var:
            phi_updated = update_phi(ctx.phi, m_star, phi_var=m_star.new_tensor(phi_var))
        else:
            phi_updated = update_phi(ctx.phi, m_star)
        return {
            "m_star": m_star,
            "hist": hist,
            "steps": n_steps,
            "elapsed_s": elapsed,
            "cos_m0_h": ctx.layer_scores[ctx.best_layer],
            "cos_ms_h": cos_ms,
            "phi_updated": phi_updated,
            "dt_eff": dt_eff,
        }

    def select_mode(self, phi_updated: torch.Tensor, args) -> str:
        if args.decode_mode != "auto":
            return args.decode_mode
        if phi_updated.norm().item() >= args.phi_threshold:
            return "multiround"
        return "direct"

    @staticmethod
    def _copy_args(args, **updates):
        payload = vars(args).copy()
        payload.update(updates)
        return SimpleNamespace(**payload)

    def decode_outputs(self, ctx: PromptContext, relax_result: dict, args):
        outputs: dict[str, str] = {}
        meta: dict[str, object] = {}
        chosen_mode = self.select_mode(relax_result["phi_updated"], args)

        def run_direct():
            text = decode_direct(
                relax_result["m_star"],
                self.model,
                self.tok,
                ctx.prompt_ids,
                max_tok=args.tokens,
                temperature=args.temperature,
            )
            outputs["direct"] = ctx.prompt + text

        def run_inject():
            text = decode_inject(
                self.model,
                self.tok,
                ctx.prompt_ids,
                relax_result["m_star"],
                inject_layer=self.inject_layer,
                ce_strength=args.ce_strength,
                max_tok=args.tokens,
                temperature=args.temperature,
            )
            outputs["inject"] = ctx.prompt + text

        def run_multiround():
            text, energies, phi_norms = generate_multiround(
                self.model,
                self.tok,
                self.W_pack,
                ctx.prompt_ids,
                self.n_layer,
                cb_topk=args.cb_topk,
                beta=args.beta,
                cb_w=self.portal if args.cb_weight is None else float(args.cb_weight),
                tau=self.tau,
                dt=relax_result["dt_eff"],
                relax_steps=args.multiround_steps,
                max_tok=args.tokens,
                temperature=args.temperature,
                backend=args.backend,
                metric_rank=args.metric_rank,
                lambda0=args.lambda0,
                lambda_phi=args.lambda_phi,
                lambda_var=args.lambda_var,
                noise_scale=args.noise_scale,
                seed=args.seed,
                w_eigvecs=self._get_w_eigvecs(args.metric_rank),
                phi_init=relax_result["phi_updated"],
                phi_threshold=args.sleep_threshold,
                sleep_decay=args.sleep_decay,
                top_k=args.top_k,
                repeat_penalty=args.repeat_penalty,
            )
            outputs["multiround"] = ctx.prompt + text
            meta["multiround_energies"] = energies
            meta["multiround_phi_norms"] = phi_norms

        def run_standalone():
            refresh_args = None
            if args.standalone_refresh_interval > 0:
                refresh_args = self._copy_args(
                    args,
                    steps=args.standalone_refresh_steps,
                    cb_topk=args.standalone_refresh_cb_topk,
                    metric_rank=args.standalone_refresh_metric_rank,
                    noise_scale=args.standalone_refresh_noise_scale,
                )
            text, token_ids, standalone_meta = self.standalone_generate(
                ctx.prompt_ids,
                relax_result["m_star"],
                max_tok=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repeat_penalty=args.repeat_penalty,
                refresh_interval=args.standalone_refresh_interval,
                refresh_args=refresh_args,
                refresh_init_layer=ctx.best_layer,
                refresh_phi=relax_result["phi_updated"],
            )
            outputs["standalone"] = ctx.prompt + text
            meta["standalone_token_ids"] = token_ids
            meta["standalone_refresh_interval"] = standalone_meta["refresh_interval"]
            meta["standalone_refresh_count"] = standalone_meta["refresh_count"]
            meta["standalone_refresh_steps"] = standalone_meta["refresh_steps"]
            meta["standalone_refresh_time_s"] = standalone_meta["refresh_time_s"]
            meta["standalone_refresh_cos_mean"] = standalone_meta["refresh_cos_mean"]
            meta["standalone_refresh_phi_norm"] = standalone_meta["refresh_phi_norm"]

        if chosen_mode == "all":
            run_direct()
            run_inject()
            run_multiround()
            if self.has_standalone_lexicon():
                run_standalone()
        elif chosen_mode == "direct":
            run_direct()
        elif chosen_mode == "inject":
            run_inject()
        elif chosen_mode == "multiround":
            run_multiround()
        elif chosen_mode == "standalone":
            run_standalone()

        return chosen_mode, outputs, meta

    def reference_generate(self, prompt: str, max_new_tokens: int) -> str:
        ids = self.tok.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(ids)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.pad_token_id,
            )
        return self.tok.decode(out[0], skip_special_tokens=True)


def build_prompt_list(args) -> list[str]:
    base = list(args.prompts) if args.prompts else [args.prompt, *DEFAULT_PROMPTS]
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in base:
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def main():
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="CE Equation-spec runtime")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--prompt", default="오늘 날씨가")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--tokens", type=int, default=15)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--multiround-steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="torch", choices=["auto", "torch", "rust", "cuda"])
    ap.add_argument("--compare-gpt2", action="store_true")
    ap.add_argument("--cb-topk", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--cb-weight", type=float, default=None)
    ap.add_argument("--metric-rank", type=int, default=16)
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--lambda-phi", dest="lambda_phi", type=float, default=0.5)
    ap.add_argument("--lambda-var", dest="lambda_var", type=float, default=0.25)
    ap.add_argument("--noise-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ce-strength", type=float, default=0.3)
    ap.add_argument(
        "--decode-mode",
        default="auto",
        choices=["auto", "direct", "inject", "multiround", "standalone", "all"],
    )
    ap.add_argument("--phi-threshold", type=float, default=1.0)
    ap.add_argument("--sleep-threshold", type=float, default=2.0)
    ap.add_argument("--sleep-decay", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repeat-penalty", type=float, default=3.0)
    ap.add_argument("--standalone-refresh-interval", type=int, default=2)
    ap.add_argument("--standalone-refresh-steps", type=int, default=48)
    ap.add_argument("--standalone-refresh-cb-topk", type=int, default=128)
    ap.add_argument("--standalone-refresh-metric-rank", type=int, default=0)
    ap.add_argument("--standalone-refresh-noise-scale", type=float, default=0.0)
    ap.add_argument(
        "--residual",
        type=float,
        default=0.15,
        help="Legacy sequential residual argument retained for CLI compatibility.",
    )
    args = ap.parse_args()

    eng = CEEngine(args.engine, device=args.device, backend=args.backend)
    mem = eng.memory_usage()
    prompts = build_prompt_list(args)

    safe_print("\n=== CE Hopfield Engine (Equation Spec Runtime) ===")
    safe_print(f"  model={eng.model_name}")
    safe_print(f"  d={eng.d}  layers={eng.n_layer}  vocab={eng.vocab}")
    safe_print(f"  tau={eng.tau:.4f}  hidden_norm_ref={eng.h_norm_ref:.2f}")
    safe_print(f"  backend={args.backend}  device={eng.device}  model_source={eng.model_source}")

    safe_print("\n--- Memory ---")
    for key, value in mem.items():
        safe_print(f"  {key}: {value:.2f}")

    safe_print(
        f"\n--- Generation (steps={args.steps}, dt={min(args.dt, 0.9 * eng.tau):.4f}, "
        f"temp={args.temperature}, mode={args.decode_mode}) ---"
    )

    results = []
    for prompt in prompts:
        ctx = eng.prompt_context(prompt)
        relax_result = eng.relax_context(ctx, args)
        chosen_mode, outputs, decode_meta = eng.decode_outputs(ctx, relax_result, args)

        safe_print(f"\n  [{prompt}]")
        safe_print(
            f"    init_layer={ctx.best_layer}  cos(m0,h)={relax_result['cos_m0_h']:.4f}  "
            f"cos(m*,h)={relax_result['cos_ms_h']:.4f}"
        )
        safe_print(
            f"    relax_steps={relax_result['steps']}  "
            f"time={relax_result['elapsed_s']:.2f}s  "
            f"phi={ctx.phi.norm().item():.2f}->{relax_result['phi_updated'].norm().item():.2f}"
        )
        if relax_result["hist"]["E"]:
            safe_print(
                f"    energy={relax_result['hist']['E'][0]:.4f}"
                f"->{relax_result['hist']['E'][-1]:.4f}"
            )

        if chosen_mode == "all":
            for name in ("direct", "inject", "multiround", "standalone"):
                if name in outputs:
                    safe_print(f"    [{name}] -> {outputs[name]}")
        else:
            output_text = outputs.get(chosen_mode, "")
            safe_print(f"    [{chosen_mode}] -> {output_text}")
        if "standalone_refresh_count" in decode_meta:
            safe_print(
                f"    [standalone-refresh] interval={decode_meta['standalone_refresh_interval']}  "
                f"count={decode_meta['standalone_refresh_count']}  "
                f"time={decode_meta['standalone_refresh_time_s']:.2f}s"
            )

        reference = None
        if args.compare_gpt2:
            reference = eng.reference_generate(prompt, args.tokens)
            safe_print(f"    [gpt2] -> {reference}")

        results.append(
            {
                "prompt": prompt,
                "best_init_layer": ctx.best_layer,
                "init_layer_sweep": {str(k): round(v, 4) for k, v in ctx.layer_scores.items()},
                "mode": chosen_mode,
                "cos_m0_h": round(relax_result["cos_m0_h"], 4),
                "cos_ms_h": round(relax_result["cos_ms_h"], 4),
                "phi_norm": {
                    "initial": round(ctx.phi.norm().item(), 4),
                    "updated": round(relax_result["phi_updated"].norm().item(), 4),
                },
                "relax_steps": relax_result["steps"],
                "relax_time_s": round(relax_result["elapsed_s"], 4),
                "energy_start": round(relax_result["hist"]["E"][0], 4) if relax_result["hist"]["E"] else None,
                "energy_end": round(relax_result["hist"]["E"][-1], 4) if relax_result["hist"]["E"] else None,
                "outputs": outputs,
                "multiround_phi_norms": [
                    round(v, 4) for v in decode_meta.get("multiround_phi_norms", [])[:32]
                ],
                "multiround_energies": [
                    round(v, 4) for v in decode_meta.get("multiround_energies", [])[:32]
                ],
                "gpt2_reference": reference,
            }
        )

    if args.compare_gpt2:
        safe_print("\n--- Summary ---")
        safe_print(f"  CE runtime: {mem['runtime_total_MB']:.1f} MB")
        safe_print(f"  GPT2 model: {mem['model_MB']:.1f} MB")
        safe_print(f"  Ratio: {mem['runtime_total_MB'] / max(mem['model_MB'], 1e-6) * 100:.1f}%")

    result_path = os.path.join(os.path.dirname(args.engine), "engine_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "engine": os.path.basename(args.engine),
                "model_name": eng.model_name,
                "device": str(eng.device),
                "backend": args.backend,
                "decode_mode": args.decode_mode,
                "tokens": args.tokens,
                "steps": args.steps,
                "temperature": args.temperature,
                "memory": mem,
                "prompts": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    safe_print(f"\n  Results -> {result_path}")


if __name__ == "__main__":
    main()
