# LLM Context Chunk

---
## File: `reality_stone/python/reality_stone/models/transformer_converter.py`

```python
import numpy as np
import json
import os
import copy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from reality_stone._rust import (
        verify_metric_consistency,
        analyze_layer,
        create_compression_plan,
        extract_global_basis
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from reality_stone.layers.rsulf_cuda import RSULFLayerCUDA, RSULFWrapperCUDA
import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class ConversionStats:
    total_layers: int = 0
    converted: int = 0
    failed: List[int] = field(default_factory=list)
    layer_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    original_params: int = 0
    compressed_params: int = 0


class RSULFTransformerConverter:
    def __init__(
        self,
        d_model: int = 4096,
        r: int = 1024,
        eta: float = 0.01,
        alpha: float = 0.02,
        beta: float = 0.01,
        gamma: float = 0.99,
        seq_len: int = 128,
        window: int = 8,
        calibration_samples: int = 1024,
        num_heads: int = 1,
        pfc_mode: str = "bilinear",
        pfc_curvature: float = 0.0,
        pfc_max_rel: float = 0.02,
        pfc_window: int = 0,
        pfc_layers: int = 3,
        pfc_speed_gate: float = 1.0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 4,
        verbose: bool = False,
        exact: bool = False,
        use_geodesic_flow: bool = False,
        geodesic_blend: float = 0.0,
    ):
        if not HAS_RUST:
            raise RuntimeError("reality_stone._rust not available")
        
        self.d_model = d_model
        self.r = r
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seq_len = seq_len
        self.window = window
        self.calibration_samples = int(max(1, calibration_samples))
        self.num_heads = int(max(1, num_heads))
        self.pfc_mode = str(pfc_mode).lower().strip()
        self.pfc_curvature = float(pfc_curvature)
        self.pfc_max_rel = float(pfc_max_rel)
        self.pfc_window = int(max(0, pfc_window))
        self.pfc_layers = int(max(0, pfc_layers))
        self.pfc_speed_gate = float(max(0.0, pfc_speed_gate))
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        self.exact = exact
        self.use_geodesic_flow = bool(use_geodesic_flow)
        self.geodesic_blend = float(max(0.0, min(1.0, geodesic_blend)))
        self.stats = ConversionStats()

    def extract_weights(self, layer) -> Dict[str, np.ndarray]:
        weights = {}
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'):
            d_model = int(self.d_model)
            c_attn_w = np.ascontiguousarray(
                layer.attn.c_attn.weight.detach().cpu().numpy().astype(np.float32)
            )
            if c_attn_w.shape == (d_model, 3 * d_model):
                wq = c_attn_w[:, :d_model].T
                wk = c_attn_w[:, d_model:2 * d_model].T
                wv = c_attn_w[:, 2 * d_model:3 * d_model].T
            elif c_attn_w.shape == (3 * d_model, d_model):
                wq = c_attn_w[:d_model, :]
                wk = c_attn_w[d_model:2 * d_model, :]
                wv = c_attn_w[2 * d_model:3 * d_model, :]
            else:
                raise ValueError(f"Unexpected GPT2 c_attn.weight shape: {c_attn_w.shape}")
            weights["WQ"] = np.ascontiguousarray(wq)
            weights["WK"] = np.ascontiguousarray(wk)
            weights["WV"] = np.ascontiguousarray(wv)
            if hasattr(layer.attn.c_attn, "bias") and layer.attn.c_attn.bias is not None:
                c_attn_b = np.ascontiguousarray(
                    layer.attn.c_attn.bias.detach().cpu().numpy().astype(np.float32)
                )
                weights["bQ"] = np.ascontiguousarray(c_attn_b[:d_model])
                weights["bK"] = np.ascontiguousarray(c_attn_b[d_model:2 * d_model])
                weights["bV"] = np.ascontiguousarray(c_attn_b[2 * d_model:3 * d_model])
            c_proj_w = np.ascontiguousarray(layer.attn.c_proj.weight.detach().cpu().numpy().astype(np.float32))
            if c_proj_w.shape != (d_model, d_model):
                raise ValueError(f"Unexpected GPT2 c_proj.weight shape: {c_proj_w.shape}")
            weights["WO"] = np.ascontiguousarray(c_proj_w.T)
            if hasattr(layer.attn.c_proj, "bias") and layer.attn.c_proj.bias is not None:
                weights["bO"] = np.ascontiguousarray(layer.attn.c_proj.bias.detach().cpu().numpy().astype(np.float32))
            w1_w = np.ascontiguousarray(layer.mlp.c_fc.weight.detach().cpu().numpy().astype(np.float32))
            if w1_w.shape == (d_model, 4 * d_model):
                w1 = w1_w.T
            elif w1_w.shape == (4 * d_model, d_model):
                w1 = w1_w
            else:
                raise ValueError(f"Unexpected GPT2 c_fc.weight shape: {w1_w.shape}")
            weights["W1"] = np.ascontiguousarray(w1)
            if hasattr(layer.mlp.c_fc, "bias") and layer.mlp.c_fc.bias is not None:
                weights["b1"] = np.ascontiguousarray(layer.mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32))
            w2_w = np.ascontiguousarray(layer.mlp.c_proj.weight.detach().cpu().numpy().astype(np.float32))
            if w2_w.shape == (4 * d_model, d_model):
                w2 = w2_w.T
            elif w2_w.shape == (d_model, 4 * d_model):
                w2 = w2_w
            else:
                raise ValueError(f"Unexpected GPT2 c_proj.weight shape: {w2_w.shape}")
            weights["W2"] = np.ascontiguousarray(w2)
            if hasattr(layer.mlp.c_proj, "bias") and layer.mlp.c_proj.bias is not None:
                weights["b2"] = np.ascontiguousarray(layer.mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32))
            weights["ffn_mode"] = "gelu_new"
            if hasattr(layer, 'ln_1'):
                weights["ln_1_weight"] = layer.ln_1.weight.detach().cpu().numpy().astype(np.float32)
                weights["ln_1_bias"] = layer.ln_1.bias.detach().cpu().numpy().astype(np.float32)
            if hasattr(layer, 'ln_2'):
                weights["ln_2_weight"] = layer.ln_2.weight.detach().cpu().numpy().astype(np.float32)
                weights["ln_2_bias"] = layer.ln_2.bias.detach().cpu().numpy().astype(np.float32)
            return weights

        q = layer.self_attn.q_proj
        k = layer.self_attn.k_proj
        v = layer.self_attn.v_proj if hasattr(layer.self_attn, "v_proj") else None
        o = layer.self_attn.o_proj if hasattr(layer.self_attn, "o_proj") else None

        weights["WQ"] = q.weight.detach().cpu().numpy().astype(np.float32)
        weights["WK"] = k.weight.detach().cpu().numpy().astype(np.float32)
        if getattr(q, "bias", None) is not None:
            weights["bQ"] = q.bias.detach().cpu().numpy().astype(np.float32)
        if getattr(k, "bias", None) is not None:
            weights["bK"] = k.bias.detach().cpu().numpy().astype(np.float32)

        if v is not None:
            weights["WV"] = v.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(v, "bias", None) is not None:
                weights["bV"] = v.bias.detach().cpu().numpy().astype(np.float32)
        if o is not None:
            weights["WO"] = o.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(o, "bias", None) is not None:
                weights["bO"] = o.bias.detach().cpu().numpy().astype(np.float32)

        if hasattr(layer.mlp, "gate_proj") and hasattr(layer.mlp, "up_proj") and hasattr(layer.mlp, "down_proj"):
            weights["W1"] = layer.mlp.up_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["W2"] = layer.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["WG"] = layer.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(layer.mlp.gate_proj, "bias", None) is not None:
                weights["bG"] = layer.mlp.gate_proj.bias.detach().cpu().numpy().astype(np.float32)
            weights["ffn_mode"] = "swiglu"
        elif hasattr(layer.mlp, "gate_proj") and hasattr(layer.mlp, "down_proj"):
            weights["W1"] = layer.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["W2"] = layer.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
            weights["ffn_mode"] = "silu"
        else:
            weights["W1"] = layer.mlp.fc1.weight.detach().cpu().numpy().astype(np.float32)
            weights["W2"] = layer.mlp.fc2.weight.detach().cpu().numpy().astype(np.float32)
            weights["ffn_mode"] = "gelu"

        if hasattr(layer, "input_layernorm"):
            weights["ln_1_weight"] = layer.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(layer.input_layernorm, "bias", None) is not None:
                weights["ln_1_bias"] = layer.input_layernorm.bias.detach().cpu().numpy().astype(np.float32)
            weights["norm_mode"] = "rmsnorm"
        elif hasattr(layer, "ln_1"):
            weights["ln_1_weight"] = layer.ln_1.weight.detach().cpu().numpy().astype(np.float32)
            weights["ln_1_bias"] = layer.ln_1.bias.detach().cpu().numpy().astype(np.float32)
            weights["norm_mode"] = "layernorm"

        if hasattr(layer, "post_attention_layernorm"):
            weights["ln_2_weight"] = layer.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            if getattr(layer.post_attention_layernorm, "bias", None) is not None:
                weights["ln_2_bias"] = layer.post_attention_layernorm.bias.detach().cpu().numpy().astype(np.float32)
        elif hasattr(layer, "ln_2"):
            weights["ln_2_weight"] = layer.ln_2.weight.detach().cpu().numpy().astype(np.float32)
            weights["ln_2_bias"] = layer.ln_2.bias.detach().cpu().numpy().astype(np.float32)

        return weights

    def verify_weights(self, weights: Dict[str, np.ndarray], idx: int) -> Tuple[bool, Dict]:
        result = {"valid": True, "issues": []}
        for name, w in weights.items():
            if not isinstance(w, np.ndarray):
                continue
            if np.isnan(w).any():
                result["valid"] = False
                result["issues"].append(f"{name} NaN")
            if np.isinf(w).any():
                result["valid"] = False
                result["issues"].append(f"{name} Inf")
        return result["valid"], result

    def convert_layer(self, layer, idx: int) -> Tuple[Optional[RSULFLayerCUDA], Dict[str, Any]]:
        layer_stat = {"idx": idx, "success": False}
        
        try:
            weights = self.extract_weights(layer)
            valid, check = self.verify_weights(weights, idx)
            if not valid:
                layer_stat["error"] = f"weight_verify: {check['issues']}"
                return None, layer_stat

            d_out, d_model = weights["WQ"].shape
            best_r = int(max(1, min(d_model, self.r)))
            best_consistency = {"fold_accuracy": 1.0, "symmetry_error": 0.0}

            layer_stat["r"] = best_r
            layer_stat["fold_accuracy"] = float(best_consistency["fold_accuracy"])
            layer_stat["symmetry_error"] = float(best_consistency["symmetry_error"])

            rsulf = RSULFLayerCUDA(
                wq=weights["WQ"],
                wk=weights["WK"],
                w1=weights["W1"],
                w2=weights["W2"],
                d_model=self.d_model,
                r=best_r,
                eta=self.eta,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                seq_len=self.seq_len,
                window=self.window,
                calibration_samples=self.calibration_samples,
                num_heads=self.num_heads,
                pfc_mode=self.pfc_mode,
                pfc_curvature=self.pfc_curvature,
                pfc_max_rel=self.pfc_max_rel,
                pfc_window=self.pfc_window,
                pfc_speed_gate=self.pfc_speed_gate,
                norm_mode=str(weights.get("norm_mode", "layernorm")),
                ffn_mode=str(weights.get("ffn_mode", "gelu")),
                use_fast=bool((not self.exact) and (self.calibration_samples > 1)),
                use_geodesic_flow=self.use_geodesic_flow,
                geodesic_blend=self.geodesic_blend,
            )
            if "WV" in weights and "WO" in weights:
                rsulf.set_attention_weights(weights["WV"], weights["WO"])
            rsulf.set_biases(
                bq=weights.get("bQ"),
                bk=weights.get("bK"),
                bv=weights.get("bV"),
                bo=weights.get("bO"),
                b1=weights.get("b1"),
                b2=weights.get("b2"),
            )
            if "WG" in weights:
                rsulf.set_ffn_gate(weights["WG"], weights.get("bG"))
            if "ln_1_weight" in weights:
                rsulf.set_ln1(weights["ln_1_weight"], weights.get("ln_1_bias"))
            if "ln_2_weight" in weights:
                rsulf.set_ln2(weights["ln_2_weight"], weights.get("ln_2_bias"))
            
            compressed, original, ratio = rsulf.param_count()
            layer_stat["compressed"] = compressed
            layer_stat["original"] = original
            layer_stat["ratio"] = float(ratio)
            layer_stat["curvature"] = float(rsulf.curvature)
            layer_stat["eta"] = float(rsulf.eta)
            layer_stat["alpha"] = float(rsulf.alpha)
            
            x_test = torch.randn(4, self.d_model)
            out, _ = rsulf(x_test)
            if torch.isnan(out).any() or torch.isinf(out).any():
                layer_stat["error"] = "forward_nan"
                return None, layer_stat
            
            layer_stat["success"] = True
            return rsulf, layer_stat
            
        except Exception as e:
            layer_stat["error"] = str(e)
            return None, layer_stat

    def convert_model(self, model) -> "RSULFModel":
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            transformer_layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            transformer_layers = model.transformer.h
        else:
             # Fallback for generic HF models or simple stacks
             if hasattr(model, "layers"):
                 transformer_layers = model.layers
             else:
                 raise AttributeError("Could not find transformer layers in model. Checked: model.model.layers, model.transformer.h")

        self.stats.total_layers = len(transformer_layers)
        
        if self.d_model == 4096:
             if hasattr(model.config, "hidden_size"):
                  self.d_model = model.config.hidden_size
             elif hasattr(model.config, "d_model"):
                  self.d_model = model.config.d_model
             elif hasattr(model.config, "n_embd"): # GPT-2
                  self.d_model = model.config.n_embd
        if hasattr(model, "config"):
            if hasattr(model.config, "n_head"):
                self.num_heads = int(model.config.n_head)
            elif hasattr(model.config, "num_attention_heads"):
                self.num_heads = int(model.config.num_attention_heads)
        
        print(f"Collecting weights from {len(transformer_layers)} layers...")
        all_wq = []
        all_wk = []
        layer_weights = []
        original_blocks = []
        
        for idx, layer in enumerate(transformer_layers):
            try:
                weights = self.extract_weights(layer)
                valid, check = self.verify_weights(weights, idx)
                if valid:
                    all_wq.append(weights["WQ"])
                    all_wk.append(weights["WK"])
                    layer_weights.append(weights)
                    original_blocks.append(layer)
                else:
                    print(f"Skipping layer {idx} due to invalid weights: {check['issues']}")
                    all_wq.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                    all_wk.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                    layer_weights.append(None)
                    original_blocks.append(None)
            except Exception as e:
                print(f"Error extracting layer {idx}: {e}")
                all_wq.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                all_wk.append(np.zeros((self.d_model, self.d_model), dtype=np.float32))
                layer_weights.append(None)
                original_blocks.append(None)

        if self.exact:
            print("Exact mode: disabling global basis and using full rank per layer")
            layers = []
            pbar_convert = tqdm(total=len(layer_weights), desc="Converting", unit="layer", disable=not self.verbose)
            for idx, weights in enumerate(layer_weights):
                if weights is None:
                    self.stats.failed.append(idx)
                    pbar_convert.set_postfix(idx=idx, status="skip")
                    pbar_convert.update(1)
                    continue
                try:
                    d_out, d_model = weights["WQ"].shape
                    if self.verbose:
                        print(f"[RSULF] layer {idx:02d}: start")
                    best_r = int(max(1, min(d_model, self.r)))
                    total_layers = len(layer_weights)
                    k = int(self.pfc_layers)
                    if k <= 0:
                        pfc_c = 0.0
                    else:
                        start = max(0, total_layers - k)
                        if idx < start:
                            pfc_c = 0.0
                        else:
                            t = float((idx - start) + 1) / float(max(1, total_layers - start))
                            pfc_c = float(self.pfc_curvature) * t
                    rsulf = RSULFLayerCUDA(
                        wq=weights["WQ"],
                        wk=weights["WK"],
                        w1=weights["W1"],
                        w2=weights["W2"],
                        d_model=self.d_model,
                        r=best_r,
                        eta=self.eta,
                        alpha=self.alpha,
                        beta=self.beta,
                        gamma=self.gamma,
                        seq_len=self.seq_len,
                        window=self.window,
                        global_basis=None,
                        original_block=None,
                        calibration_samples=self.calibration_samples,
                        num_heads=self.num_heads,
                        pfc_mode=self.pfc_mode,
                        pfc_curvature=pfc_c,
                        pfc_max_rel=self.pfc_max_rel,
                        pfc_window=self.pfc_window,
                        pfc_speed_gate=self.pfc_speed_gate,
                        use_fast=False,
                        norm_mode=str(weights.get("norm_mode", "layernorm")),
                        ffn_mode=str(weights.get("ffn_mode", "gelu")),
                        use_geodesic_flow=self.use_geodesic_flow,
                        geodesic_blend=self.geodesic_blend,
                    )
                    if "WV" in weights and "WO" in weights:
                        rsulf.set_attention_weights(weights["WV"], weights["WO"])
                    rsulf.set_biases(
                        bq=weights.get("bQ"),
                        bk=weights.get("bK"),
                        bv=weights.get("bV"),
                        bo=weights.get("bO"),
                        b1=weights.get("b1"),
                        b2=weights.get("b2"),
                    )
                    if "ln_1_weight" in weights:
                        rsulf.set_ln1(weights["ln_1_weight"], weights.get("ln_1_bias"))
                    if "ln_2_weight" in weights:
                        rsulf.set_ln2(weights["ln_2_weight"], weights.get("ln_2_bias"))
                    compressed, original, ratio = rsulf.param_count()
                    if self.verbose:
                        print(f"[RSULF] layer {idx:02d}: ok ratio={ratio:.1f}x")
                    layers.append(rsulf)
                    self.stats.converted += 1
                    self.stats.original_params += original
                    self.stats.compressed_params += compressed
                    pbar_convert.set_postfix(idx=idx, ratio=f"{ratio:.1f}x", status="ok")
                except Exception as e:
                    print(f"[RSULF] layer {idx:02d}: fail {e}")
                    self.stats.failed.append(idx)
                    self.stats.errors.append({"layer": idx, "error": str(e)})
                    pbar_convert.set_postfix(idx=idx, status="fail")
                if self.checkpoint_dir and (idx + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(layers, idx + 1)
                pbar_convert.update(1)
            pbar_convert.close()
            return RSULFModel(layers, self.stats)
        print("Phase 1: Analyzing layers...")
        analyses = []
        pbar_analyze = tqdm(total=len(layer_weights), desc="Analyzing", unit="layer", disable=not self.verbose)
        for idx, weights in enumerate(layer_weights):
            if weights:
                analysis = analyze_layer(
                    weights["WQ"], weights["WK"], weights["W1"], weights["W2"], 
                    idx, self.r
                )
                analyses.append(analysis)
                acc = analysis.get("expected_accuracy", 0.0)
                pbar_analyze.set_postfix(idx=idx, acc=f"{acc:.4f}")
            pbar_analyze.update(1)
        pbar_analyze.close()
            
        print("Phase 2: Planning compression...")
        if analyses:
            plan = create_compression_plan(analyses, 0.95)
            print(f"Plan: ratio={plan.get('expected_compression_ratio', 0):.2f}x, acc={plan.get('min_expected_accuracy', 0):.4f}")
        
        print("Phase 3: Extracting Global Basis...")
        global_basis = None
        try:
            valid_wq = [w for w in all_wq if w.shape[0] > 0 and w.any()]
            valid_wk = [w for w in all_wk if w.shape[0] > 0 and w.any()]
            if valid_wq:
                global_basis = extract_global_basis(valid_wq, valid_wk, self.r)
                self.global_basis = global_basis
                print(f"Global Basis extracted: rank={global_basis['rank']}")
        except Exception as e:
            print(f"Global Basis extraction failed: {e}. Falling back to local.")

        layers = []
        acc_by_idx = {a.get("layer_idx", i): a.get("expected_accuracy", 0.0) for i, a in enumerate(analyses)}
        rank_by_idx = {
            a.get("layer_idx", i): int(a.get("recommended_rank", self.r))
            for i, a in enumerate(analyses)
        }
        pbar_convert = tqdm(total=len(layer_weights), desc="Converting", unit="layer", disable=not self.verbose)
        for idx, weights in enumerate(layer_weights):
            if weights is None:
                self.stats.failed.append(idx)
                pbar_convert.set_postfix(idx=idx, status="skip")
                pbar_convert.update(1)
                continue
            try:
                d_out, d_model = weights["WQ"].shape
                base_r = rank_by_idx.get(idx, self.r)
                best_r = int(max(1, min(d_model, self.r, base_r)))
                if self.verbose:
                    print(f"[RSULF] layer {idx:02d}: start")
                rsulf = RSULFLayerCUDA(
                    wq=weights["WQ"],
                    wk=weights["WK"],
                    w1=weights["W1"],
                    w2=weights["W2"],
                    d_model=self.d_model,
                    r=best_r,
                    eta=self.eta,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                    seq_len=self.seq_len,
                    window=self.window,
                    global_basis=global_basis,
                    calibration_samples=self.calibration_samples,
                    num_heads=self.num_heads,
                    pfc_mode=self.pfc_mode,
                    pfc_curvature=self.pfc_curvature,
                    pfc_max_rel=self.pfc_max_rel,
                    pfc_window=self.pfc_window,
                    pfc_speed_gate=self.pfc_speed_gate,
                    norm_mode=str(weights.get("norm_mode", "layernorm")),
                    ffn_mode=str(weights.get("ffn_mode", "gelu")),
                    use_fast=bool(self.calibration_samples > 1),
                    use_geodesic_flow=self.use_geodesic_flow,
                    geodesic_blend=self.geodesic_blend,
                )
                if "WV" in weights and "WO" in weights:
                    rsulf.set_attention_weights(weights["WV"], weights["WO"])
                rsulf.set_biases(
                    bq=weights.get("bQ"),
                    bk=weights.get("bK"),
                    bv=weights.get("bV"),
                    bo=weights.get("bO"),
                    b1=weights.get("b1"),
                    b2=weights.get("b2"),
                )
                if "WG" in weights:
                    rsulf.set_ffn_gate(weights["WG"], weights.get("bG"))
                if "ln_1_weight" in weights:
                    rsulf.set_ln1(weights["ln_1_weight"], weights.get("ln_1_bias"))
                if "ln_2_weight" in weights:
                    rsulf.set_ln2(weights["ln_2_weight"], weights.get("ln_2_bias"))
                
                compressed, original, ratio = rsulf.param_count()
            
                if self.verbose:
                    print(f"[RSULF] layer {idx:02d}: ok ratio={ratio:.1f}x")
                
                layers.append(rsulf)
                self.stats.converted += 1
                self.stats.original_params += original
                self.stats.compressed_params += compressed
                acc = acc_by_idx.get(idx, 0.0)
                pbar_convert.set_postfix(idx=idx, acc=f"{acc:.4f}", ratio=f"{ratio:.1f}x", status="ok")
                
            except Exception as e:
                print(f"[RSULF] layer {idx:02d}: fail {e}")
                self.stats.failed.append(idx)
                self.stats.errors.append({"layer": idx, "error": str(e)})
                pbar_convert.set_postfix(idx=idx, status="fail")
            
            if self.checkpoint_dir and (idx + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(layers, idx + 1)
            pbar_convert.update(1)
        
        pbar_convert.close()
        return RSULFModel(layers, self.stats)

    def _save_checkpoint(self, layers: List[RSULFLayerCUDA], count: int):
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{count}.json")
        data = {
            "count": count,
            "stats": {
                "converted": self.stats.converted,
                "failed": self.stats.failed,
                "original_params": self.stats.original_params,
                "compressed_params": self.stats.compressed_params,
            },
            "layers": []
        }
        for layer in layers:
            comp = layer._layer.export_components()
            layer_data = {}
            for k, v in comp.items():
                if isinstance(v, np.ndarray):
                    layer_data[k] = v.tolist()
                else:
                    layer_data[k] = v
            data["layers"].append(layer_data)
        with open(path, "w") as f:
            json.dump(data, f)

    def analyze_errors(self) -> Dict[str, Any]:
        analysis = {"total": len(self.stats.errors), "by_type": {}}
        for err in self.stats.errors:
            msg = err.get("error", "unknown")
            key = msg.split(":")[0] if ":" in msg else msg
            if key not in analysis["by_type"]:
                analysis["by_type"][key] = []
            analysis["by_type"][key].append(err["layer"])
        return analysis

    def verify_conversion(self, wq: np.ndarray, wk: np.ndarray) -> Dict:
        return verify_metric_consistency(
            wq.astype(np.float32),
            wk.astype(np.float32),
            self.r
        )


class RSULFModel(torch.nn.Module):
    def __init__(self, layers: List[RSULFLayerCUDA], stats: Optional[ConversionStats] = None):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.wrappers = torch.nn.ModuleList([
            RSULFWrapperCUDA(layer) for layer in layers
        ])
        self.stats = stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for wrapper in self.wrappers:
            x = wrapper(x)
        return x

    def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
        for wrapper in self.wrappers:
            x_t = wrapper.forward_step(x_t)
        return x_t

    def reset_memory(self):
        for wrapper in self.wrappers:
            wrapper.v_mem = None

    def init_step_cache(self, batch: int, max_len: int, device: torch.device, dtype: torch.dtype):
        for wrapper in self.wrappers:
            if hasattr(wrapper, "init_step_cache"):
                wrapper.init_step_cache(batch, max_len, device, dtype)


class TorchRiemannianDecoder(nn.Module):
    def __init__(self, u: np.ndarray, a: np.ndarray, bt: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.u = nn.Parameter(torch.from_numpy(np.asarray(u, dtype=np.float32)), requires_grad=False)
        self.a = nn.Parameter(torch.from_numpy(np.asarray(a, dtype=np.float32)), requires_grad=False)
        self.bt = nn.Parameter(torch.from_numpy(np.asarray(bt, dtype=np.float32)), requires_grad=False)
        self.bias = nn.Parameter(torch.from_numpy(np.asarray(bias, dtype=np.float32)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            b, s, d = x.shape
            flat = x.reshape(-1, d).to(dtype=torch.float32)
            y = flat @ self.u
            y = y @ self.a.T
            y = y @ self.bt.T
            y = y + self.bias.unsqueeze(0)
            return y.view(b, s, -1)
        if x.dim() == 2:
            flat = x.to(dtype=torch.float32)
            y = flat @ self.u
            y = y @ self.a.T
            y = y @ self.bt.T
            y = y + self.bias.unsqueeze(0)
            return y
        raise ValueError("decoder input must be 2D or 3D")


class SyntaxHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))
        return x + self.fc2(h)


class RSULFCausalLM(nn.Module):
    def __init__(
        self,
        rsulf: RSULFModel,
        token_embedding: nn.Embedding,
        lm_head: nn.Linear,
        final_norm: Optional[nn.Module] = None,
        pos_embedding: Optional[nn.Embedding] = None,
        decoder=None,
        apply_final_norm: bool = True,
    ):
        super().__init__()
        self.rsulf = rsulf
        self.token_embedding = token_embedding
        self.pos_embedding = pos_embedding
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.decoder = decoder
        self.apply_final_norm = bool(apply_final_norm)
        self.syntax_head = SyntaxHead(int(token_embedding.weight.size(1)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.rsulf, "reset_memory"):
            self.rsulf.reset_memory()
        x = self.token_embedding(input_ids)
        if self.pos_embedding is not None:
            pos = torch.arange(input_ids.size(1), device=input_ids.device, dtype=torch.long)
            x = x + self.pos_embedding(pos)[None, :, :]
        x = self.rsulf(x)
        x = self.syntax_head(x)
        if self.decoder is not None:
            x_np = x.detach().to("cpu", dtype=torch.float32).reshape(-1, x.size(-1)).numpy().astype(np.float32)
            y_np = self.decoder.forward(x_np)
            x = torch.from_numpy(y_np).to(device=input_ids.device, dtype=x.dtype).view(input_ids.size(0), input_ids.size(1), -1)
        if self.final_norm is not None and bool(self.apply_final_norm):
            x = self.final_norm(x)
        return self.lm_head(x)

    def _decode_hidden(self, h: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            return h
        if isinstance(self.decoder, nn.Module):
            y = self.decoder(h)
            return y.to(dtype=h.dtype)
        h_np = h.detach().to("cpu", dtype=torch.float32).reshape(-1, h.size(-1)).numpy().astype(np.float32)
        y_np = self.decoder.forward(h_np)
        y = torch.from_numpy(y_np).to(device=h.device, dtype=h.dtype).view(h.size(0), h.size(1), -1)
        return y

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        return self.generate_sample(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.15,
        )

    def generate_sample(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        from reality_stone.utils.sampling import sample_next_token
        device = input_ids.device
        out = input_ids
        self.rsulf.reset_memory()
        self.rsulf.init_step_cache(
            batch=int(out.size(0)),
            max_len=int(out.size(1) + max(1, int(max_new_tokens)) + 1),
            device=device,
            dtype=self.token_embedding.weight.dtype,
        )
        for pos_idx in range(out.size(1)):
            tok = out[:, pos_idx : pos_idx + 1]
            x_t = self.token_embedding(tok)
            if self.pos_embedding is not None:
                pos = torch.tensor([pos_idx], device=device, dtype=torch.long)
                x_t = x_t + self.pos_embedding(pos)[None, :, :]
            x_t = self.rsulf.forward_step(x_t)

        finished = torch.zeros(out.size(0), device=device, dtype=torch.bool)
        for step in range(int(max_new_tokens)):
            if out.size(1) == 0:
                break
            if step == 0:
                h = x_t
            else:
                tok = out[:, -1:]
                x_t = self.token_embedding(tok)
                if self.pos_embedding is not None:
                    pos = torch.tensor([out.size(1) - 1], device=device, dtype=torch.long)
                    x_t = x_t + self.pos_embedding(pos)[None, :, :]
                x_t = self.rsulf.forward_step(x_t)
                h = x_t
            h = self.syntax_head(h)
            h = self._decode_hidden(h)
            if self.final_norm is not None and bool(self.apply_final_norm):
                h = self.final_norm(h)
            logits = self.lm_head(h)[:, -1, :]
            next_id = sample_next_token(
                logits,
                generated_ids=out,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
            )
            if eos_token_id is not None:
                eos = torch.full_like(next_id, int(eos_token_id))
                next_id = torch.where(finished.unsqueeze(1), eos, next_id)
            out = torch.cat([out, next_id], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_id.squeeze(1) == int(eos_token_id))
                if bool(finished.all().item()):
                    break
        return out


def save_rsulf_causal_lm(path: str, rs_lm: RSULFCausalLM, decoder_state: dict | None = None) -> None:
    state = rs_lm.state_dict()
    state_cpu = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            state_cpu[k] = v.detach().cpu()
        else:
            state_cpu[k] = v

    layers = getattr(getattr(rs_lm, "rsulf", None), "layers", None)
    layer_meta = []
    if layers is not None:
        for layer in layers:
            layer_meta.append(
                {
                    "d_model": int(getattr(layer, "d_model", 0)),
                    "r": int(getattr(layer, "r", 0)),
                    "ffn_dim": int(getattr(getattr(layer, "b1", None), "numel", lambda: 0)()),
                    "num_heads": int(getattr(layer, "num_heads", 1)),
                    "pfc_mode": str(getattr(layer, "pfc_mode", "accel")),
                    "pfc_curvature": float(getattr(layer, "pfc_curvature", 0.0)),
                    "pfc_max_rel": float(getattr(layer, "pfc_max_rel", 0.02)),
                    "pfc_window": int(getattr(layer, "pfc_window", 0)),
                    "pfc_speed_gate": float(getattr(layer, "pfc_speed_gate", 1.0)),
                    "norm_mode": str(getattr(layer, "norm_mode", "layernorm")),
                    "ffn_mode": str(getattr(layer, "ffn_mode", "gelu")),
                }
            )

    token_emb = getattr(rs_lm, "token_embedding", None)
    pos_emb = getattr(rs_lm, "pos_embedding", None)
    meta = {
        "vocab_size": int(token_emb.weight.size(0)) if token_emb is not None else 0,
        "d_model": int(token_emb.weight.size(1)) if token_emb is not None else 0,
        "max_positions": int(pos_emb.weight.size(0)) if pos_emb is not None else 0,
        "num_layers": int(len(layer_meta)),
        "layer_meta": layer_meta,
        "apply_final_norm": bool(getattr(rs_lm, "apply_final_norm", True)),
    }

    payload = {"meta": meta, "state_dict": state_cpu}
    if decoder_state is not None:
        payload["decoder_state"] = {
            "u": np.asarray(decoder_state["u"], dtype=np.float32),
            "a": np.asarray(decoder_state["a"], dtype=np.float32),
            "bt": np.asarray(decoder_state["bt"], dtype=np.float32),
            "bias": np.asarray(decoder_state["bias"], dtype=np.float32),
        }
    torch.save(payload, path)


def load_rsulf_causal_lm(path: str, device: str | torch.device | None = None) -> RSULFCausalLM:
    payload = torch.load(path, map_location="cpu")
    meta = payload.get("meta") or {}
    state = payload.get("state_dict") or {}
    layer_meta = meta.get("layer_meta") or []

    vocab_size = int(meta.get("vocab_size") or 0)
    d_model = int(meta.get("d_model") or 0)
    max_positions = int(meta.get("max_positions") or 0)

    if vocab_size <= 0 or d_model <= 0 or max_positions <= 0:
        raise ValueError("Invalid checkpoint meta (vocab_size/d_model/max_positions)")
    if not layer_meta:
        raise ValueError("Invalid checkpoint meta (layer_meta missing)")

    layers: list[RSULFLayerCUDA] = []
    for lm in layer_meta:
        dm = int(lm.get("d_model") or d_model)
        r = int(lm.get("r") or dm)
        ffn_dim = int(lm.get("ffn_dim") or (4 * dm))
        wq0 = np.zeros((dm, dm), dtype=np.float32)
        wk0 = np.zeros((dm, dm), dtype=np.float32)
        w10 = np.zeros((ffn_dim, dm), dtype=np.float32)
        w20 = np.zeros((dm, ffn_dim), dtype=np.float32)
        layer = RSULFLayerCUDA(
            wq=wq0,
            wk=wk0,
            w1=w10,
            w2=w20,
            d_model=dm,
            r=r,
            eta=0.0,
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            seq_len=0,
            window=0,
            global_basis=None,
            original_block=None,
            use_fast=False,
            calibration_samples=0,
            num_heads=int(lm.get("num_heads") or 1),
            pfc_mode=str(lm.get("pfc_mode") or "accel"),
            pfc_curvature=float(lm.get("pfc_curvature") or 0.0),
            pfc_max_rel=float(lm.get("pfc_max_rel") or 0.02),
            pfc_window=int(lm.get("pfc_window") or 0),
            pfc_speed_gate=float(lm.get("pfc_speed_gate") or 1.0),
            norm_mode=str(lm.get("norm_mode") or "layernorm"),
            ffn_mode=str(lm.get("ffn_mode") or "gelu"),
        )
        layers.append(layer)

    rsulf = RSULFModel(layers, stats=None)

    token_embedding = nn.Embedding(vocab_size, d_model)
    pos_embedding = nn.Embedding(max_positions, d_model)
    final_norm = nn.LayerNorm(d_model, elementwise_affine=True)
    lm_head = nn.Linear(d_model, vocab_size, bias=False)

    rs_lm = RSULFCausalLM(
        rsulf=rsulf,
        token_embedding=token_embedding,
        lm_head=lm_head,
        final_norm=final_norm,
        pos_embedding=pos_embedding,
        decoder=None,
        apply_final_norm=bool(meta.get("apply_final_norm", True)),
    )
    rs_lm.load_state_dict(state, strict=False)

    decoder_state = payload.get("decoder_state")
    if decoder_state is not None:
        rs_lm.decoder = TorchRiemannianDecoder(
            np.asarray(decoder_state["u"], dtype=np.float32),
            np.asarray(decoder_state["a"], dtype=np.float32),
            np.asarray(decoder_state["bt"], dtype=np.float32),
            np.asarray(decoder_state["bias"], dtype=np.float32),
        )

    if device is not None:
        rs_lm = rs_lm.to(device)
    rs_lm.eval()
    return rs_lm


def build_rsulf_causal_lm(model: nn.Module, converter: RSULFTransformerConverter) -> RSULFCausalLM:
    rsulf = converter.convert_model(model)
    return wrap_rsulf_as_causal_lm(model, rsulf)

def wrap_rsulf_as_causal_lm(model: nn.Module, rsulf: RSULFModel) -> RSULFCausalLM:
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        wte = model.transformer.wte
        token_embedding = nn.Embedding(wte.weight.size(0), wte.weight.size(1))
        token_embedding.weight.data = wte.weight.detach().clone().cpu()
        token_embedding.weight.requires_grad = False
        pos_embedding = None
        if hasattr(model.transformer, "wpe"):
            wpe = model.transformer.wpe
            pos_embedding = nn.Embedding(wpe.weight.size(0), wpe.weight.size(1))
            pos_embedding.weight.data = wpe.weight.detach().clone().cpu()
            pos_embedding.weight.requires_grad = False
        final_norm = None
        if hasattr(model.transformer, "ln_f"):
            ln_f = model.transformer.ln_f
            final_norm = nn.LayerNorm(ln_f.weight.numel(), elementwise_affine=True)
            final_norm.weight.data = ln_f.weight.detach().clone().cpu()
            final_norm.bias.data = ln_f.bias.detach().clone().cpu()
            final_norm.weight.requires_grad = False
            final_norm.bias.requires_grad = False
        vocab = token_embedding.weight.size(0)
        d_model = token_embedding.weight.size(1)
        lm_head = nn.Linear(d_model, vocab, bias=False)
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            lm_head.weight.data = model.lm_head.weight.detach().clone().cpu()
        else:
            lm_head.weight.data = token_embedding.weight.detach().clone().cpu()
        lm_head.weight.requires_grad = False
        return RSULFCausalLM(rsulf, token_embedding, lm_head, final_norm=final_norm, pos_embedding=pos_embedding)

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        emb = model.model.embed_tokens
        token_embedding = nn.Embedding(emb.weight.size(0), emb.weight.size(1))
        token_embedding.weight.data = emb.weight.detach().clone().cpu()
        token_embedding.weight.requires_grad = False
        final_norm = None
        if hasattr(model.model, "norm"):
            nrm = model.model.norm
            d_model = token_embedding.weight.size(1)
            ln = nn.LayerNorm(d_model, elementwise_affine=True)
            ln.weight.data = nrm.weight.detach().clone().cpu()
            ln.bias.data.zero_()
            ln.weight.requires_grad = False
            ln.bias.requires_grad = False
            final_norm = ln
        vocab = token_embedding.weight.size(0)
        d_model = token_embedding.weight.size(1)
        lm_head = nn.Linear(d_model, vocab, bias=False)
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            lm_head.weight.data = model.lm_head.weight.detach().clone().cpu()
        else:
            lm_head.weight.data = token_embedding.weight.detach().clone().cpu()
        lm_head.weight.requires_grad = False
        return RSULFCausalLM(rsulf, token_embedding, lm_head, final_norm=final_norm, pos_embedding=None)

    raise ValueError("Unsupported model structure for RSULF causal LM")


def convert_transformer_to_rsulf(
    model: nn.Module,
    d_model: int = 4096,
    r: int = 1024,
    eta: float = 0.01,
    alpha: float = 0.02,
    beta: float = 0.01,
    gamma: float = 0.99,
    seq_len: int = 128,
    window: int = 8,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 4,
    verbose: bool = False,
    exact: bool = False,
) -> RSULFModel:
    converter = RSULFTransformerConverter(
        d_model=d_model,
        r=r,
        eta=eta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seq_len=seq_len,
        window=window,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        verbose=verbose,
        exact=exact,
    )
    return converter.convert_model(model)

class FFNPotential(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.d_model = d_model
        self.P = nn.Parameter(torch.zeros(d_model, d_model))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x_flat = x.view(-1, d)
        quad = 0.5 * (x_flat @ self.P * x_flat).sum(dim=-1, keepdim=True)
        neu = self.net(x_flat)
        phi = quad + neu
        return phi.view(b, l)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            phi = self.forward(x_in).sum()
            grad = torch.autograd.grad(phi, x_in, create_graph=True)[0]
        return grad


class LowRankFFN(nn.Module):
    def __init__(self, mlp: nn.Module, rank: int):
        super().__init__()
        if hasattr(mlp, 'c_fc'):
            w1 = mlp.c_fc.weight.data
            b1 = mlp.c_fc.bias.data if mlp.c_fc.bias is not None else None
            w2 = mlp.c_proj.weight.data
            b2 = mlp.c_proj.bias.data if mlp.c_proj.bias is not None else None
        else:
            raise ValueError("Unsupported MLP structure")
        
        d_model, ffn_dim = w1.shape
        rank = min(rank, d_model, ffn_dim)
        u1, s1, v1 = torch.linalg.svd(w1, full_matrices=False)
        u1_r = u1[:, :rank]
        s1_r = s1[:rank]
        v1_r = v1[:rank, :]
        self.w1_a = nn.Parameter(u1_r * s1_r.unsqueeze(0))
        self.w1_b = nn.Parameter(v1_r)
        self.b1 = nn.Parameter(b1) if b1 is not None else None
        u2, s2, v2 = torch.linalg.svd(w2, full_matrices=False)
        u2_r = u2[:, :rank]
        s2_r = s2[:rank]
        v2_r = v2[:rank, :]
        self.w2_a = nn.Parameter(u2_r * s2_r.unsqueeze(0))
        self.w2_b = nn.Parameter(v2_r)
        self.b2 = nn.Parameter(b2) if b2 is not None else None
        
        self.act = nn.GELU(approximate='tanh')
        self.rank = rank
        self.d_model = d_model
        self.ffn_dim = ffn_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.w1_a @ self.w1_b
        if self.b1 is not None:
            h = h + self.b1
        h = self.act(h)
        out = h @ self.w2_a @ self.w2_b
        if self.b2 is not None:
            out = out + self.b2
        return out

    def param_count(self):
        original = self.ffn_dim * self.d_model * 2
        compressed = 2 * (self.rank * self.d_model + self.rank * self.ffn_dim)
        return compressed, original


class StructuralRSULFLayer(nn.Module):
    def __init__(self, block: nn.Module, d_model: int, rank: int):
        super().__init__()
        self.ln_1 = copy.deepcopy(block.ln_1)
        self.ln_2 = copy.deepcopy(block.ln_2)
        self.attn = copy.deepcopy(block.attn)
        self.mlp = LowRankFFN(block.mlp, rank)
        self.potential = FFNPotential(d_model, hidden_dim=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.ln_1(x)
        attn_outputs = self.attn(u)
        attn_out = attn_outputs[0] if isinstance(attn_outputs, (tuple, list)) else attn_outputs
        y = x + attn_out
        w = self.ln_2(y)
        ffn_out = self.mlp(w)
        return y + ffn_out


class StructuralRSULFModel(nn.Module):
    def __init__(
        self,
        blocks: List[nn.Module],
        d_model: int,
        rank: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if rank is None:
            if hidden_dim is not None:
                rank = hidden_dim
            else:
                rank = d_model
        self.layers = nn.ModuleList(
            [StructuralRSULFLayer(block, d_model, rank) for block in blocks]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def wrappers(self) -> nn.ModuleList:
        return self.layers
```
---
## File: `reality_stone/python/reality_stone/optim/__init__.py`

```python
"""Riemannian optimizers for hyperbolic spaces."""

from .riemannian_adam import PoincareRiemannianAdam

__all__ = ["PoincareRiemannianAdam"]
```
---
## File: `reality_stone/python/reality_stone/optim/riemannian_adam.py`

```python
"""Riemannian Adam optimizer for Poincare ball manifold."""

import torch
from torch.optim import Optimizer
import numpy as np
from typing import List, Optional, Callable

try:
    import reality_stone._rust as _rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class PoincareRiemannianAdam(Optimizer):
    def __init__(
        self,
        params,
        c: float,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        max_norm_eps: float = 1e-7,  # Relaxed boundary constraint for f32
    ):
        if not _HAS_RUST:
            raise RuntimeError(
                "Rust extension not available. "
                "Please build with: uv run maturin develop --features cuda"
            )
        if c <= 0:
            raise ValueError(f"Curvature c must be positive, got {c}")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if max_norm_eps < 0.0:
            raise ValueError(f"Invalid max_norm_eps value: {max_norm_eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, c=c, max_norm_eps=max_norm_eps)
        super().__init__(params, defaults)
        self._step = 0
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            c = group["c"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.detach()
                state = self.state[p]
                
                # Initialize state on first step
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                
                m = state["m"]
                v = state["v"]
                
                # Convert to numpy for Rust call
                x_np = p.detach().cpu().numpy().astype(np.float32)
                g_np = grad.cpu().numpy().astype(np.float32)
                m_np = m.cpu().numpy().astype(np.float32)
                v_np = v.cpu().numpy().astype(np.float32)
                
                # Handle 1D tensors (e.g. bias) by reshaping to 2D
                is_1d = x_np.ndim == 1
                if is_1d:
                    x_np = x_np.reshape(1, -1)
                    g_np = g_np.reshape(1, -1)
                    m_np = m_np.reshape(1, -1)
                    v_np = v_np.reshape(1, -1)
                
                # Call Rust core implementation
                max_norm_eps = group.get("max_norm_eps", 1e-7)
                x_new_np, m_new_np, v_new_np = _rust.poincare.poincare_riemannian_adam_step_cpu(  # type: ignore[attr-defined]
                    x_np,
                    g_np,
                    m_np,
                    v_np,
                    self._step,
                    float(c),
                    float(lr),
                    float(beta1),
                    float(beta2),
                    float(eps),
                    float(max_norm_eps),
                )
                
                # Restore shape if 1D
                if is_1d:
                    x_new_np = x_new_np.reshape(-1)
                    m_new_np = m_new_np.reshape(-1)
                    v_new_np = v_new_np.reshape(-1)
                
                # Update parameter and state
                p.copy_(torch.from_numpy(x_new_np).to(p.device))
                state["m"] = torch.from_numpy(m_new_np)
                state["v"] = torch.from_numpy(v_new_np)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
```
---
## File: `reality_stone/python/reality_stone/utils/__init__.py`

```python
"""Utility modules for reality_stone"""
from . import misc
from . import sampling
```
---
## File: `reality_stone/python/reality_stone/utils/misc.py`

```python
import torch
import numpy as np
import random

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_mnist_dataloaders(data_dir="./data", batch_size=256, test_batch_size=1000, download=True):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader

def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0
```
---
## File: `reality_stone/python/reality_stone/utils/pre_segmenter.py`

```python
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
try:
    from transformers import AutoTokenizer
except Exception: 
    AutoTokenizer = None 


@dataclass
class TreeNode:
    id: int
    type: str
    parent: Optional[int]
    text: str


@dataclass
class DocumentTree:
    nodes: List[TreeNode]
    root_id: int

    def children(self, node_id: int) -> List[int]:
        return [n.id for n in self.nodes if n.parent == node_id]


class LevelSegmenter:
    def __init__(self, level: str, parent: "PreSegmenter"):
        self.level = level
        self._segment_sentences = parent._segment_sentences
        self._tokenize_sentences = parent._tokenize_sentences
    
    def segment(self, text: str) -> List[str]:
        if self.level == "document":
            return [text]
        if self.level == "section":
            return [text]
        if self.level == "subsection":
            blocks = re.split(r"\n\n+", text)
            # Heuristic: drop leading section title block if it has no newline or sentence punctuation
            if blocks and ("\n" not in blocks[0] and "." not in blocks[0] and "!" not in blocks[0] and "?" not in blocks[0]):
                blocks = blocks[1:]
            return blocks
        if self.level == "paragraph":
            return re.split(r"\n\n+", text)
        if self.level == "sentence":
            return self._segment_sentences(text)
        if self.level == "phrase":
            return re.split(r"[.,;]+", text)
        if self.level == "token":
            return self._tokenize_sentences([text])[1][0]
        return [text]


class PreSegmenter:
    def __init__(
        self,
        max_length: int = 128,
        k_neighbors: int = 3,
        tokenizer_name: str = "klue/bert-base",
    ):
        self.max_length = max_length
        self.k_neighbors = k_neighbors

        self.sentence_endings = re.compile(r"([.!?])\s+")

        self.tokenizer = None
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                print(f"Warning: failed to load tokenizer '{tokenizer_name}': {e}")
                self.tokenizer = None
    
    def recursive_segment(self, text: str, levels: List[str] = ['document', 'paragraph', 'sentence', 'token']) -> DocumentTree:
        nodes = []
        node_id = 0
        parent_map = {}  # id -> children list
        
        def add_node(level: str, text: str, parent: Optional[int]) -> int:
            nonlocal node_id
            nid = node_id
            nodes.append(TreeNode(id=nid, type=level, parent=parent, text=text))
            if parent is not None:
                if parent not in parent_map:
                    parent_map[parent] = []
                parent_map[parent].append(nid)
            node_id += 1
            return nid
        
        # Start with root
        root_id = add_node('document', text, None)
        
        # Recursive build
        def build_level(current_id: int, current_text: str, level_idx: int):
            if level_idx >= len(levels) - 1:
                return
            next_level = levels[level_idx + 1]
            segmenter = LevelSegmenter(next_level, self)
            children_texts = segmenter.segment(current_text)
            for child_text in children_texts:
                if not child_text.strip():
                    continue
                child_id = add_node(next_level, child_text, current_id)
                build_level(child_id, child_text, level_idx + 1)
        
        build_level(root_id, text, 0)
        
        # Handle token level separately for all leaf nodes (sentences/phrases)
        for node in nodes:
            if node.type in ['sentence', 'phrase']:  # Assuming leaves before tokens
                tokens = self._tokenize_sentences([node.text])[1][0]
                for tok in tokens:
                    add_node('token', tok, node.id)
        
        tree = DocumentTree(nodes=nodes, root_id=root_id)
        return tree

    def __call__(self, paragraph: str) -> Dict:
        # Update to use recursive_segment
        tree = self.recursive_segment(paragraph)
        
        # Extract existing fields from tree
        sentences = [n.text for n in tree.nodes if n.type == 'sentence']
        tokens, token_strings = self._tokenize_sentences(sentences)
        replacement_mask = self._generate_replacement_mask(token_strings, sentences)
        topo_idx = self._build_topology(len(sentences), k=self.k_neighbors)
        # tree = self._build_document_tree(paragraph, sentences, token_strings) # This line is no longer needed
        
        metadata = {
            "num_sentences": len(sentences),
            "sentence_lengths": [len(s.split()) for s in sentences],
            "total_tokens": tokens.shape[1]
        }
        
        return {
            "sentences": sentences,
            "tokens": tokens,
            "replacement_mask": replacement_mask,
            "topo_idx": topo_idx,
            "tree": tree,
            "metadata": metadata,
        }
    
    def _segment_sentences(self, paragraph: str) -> List[str]:
        """
        문장 분해
        
        docs 명세: 
        - 한국어 kss 또는 nltk.sent_tokenize 사용
        - 너무 짧은 문장 병합
        """
        # 간단한 정규식 기반 문장 분리
        sentences = []
        current = []
        
        for char in paragraph:
            current.append(char)
            if char in '.!?' and len(''.join(current).strip()) > 5:
                sent = ''.join(current).strip()
                if sent:
                    sentences.append(sent)
                current = []
        
        # 남은 문자열 처리
        if current:
            sent = ''.join(current).strip()
            if sent:
                sentences.append(sent)
        
        # 후처리: 너무 짧은 문장 병합
        merged = []
        buffer = ""
        for sent in sentences:
            if len(sent) < 10 and buffer:
                buffer += " " + sent
            else:
                if buffer:
                    merged.append(buffer)
                buffer = sent
        if buffer:
            merged.append(buffer)
        
        return merged if merged else sentences
    
    def _tokenize_sentences(self, sentences: List[str]) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        문장 토큰화

        우선적으로 HF 토크나이저(예: klue/bert-base)를 사용하고,
        transformers가 없거나 로딩에 실패한 환경에서는 기존 문자 단위 토큰화로 fallback 한다.

        Returns:
            tokens: [num_sents, max_seq_len] 토큰 ID 텐서
            token_strings: [num_sents][seq_len] 토큰 문자열 리스트
        """
        all_tokens: List[List[int]] = []
        all_token_strings: List[List[str]] = []

        if self.tokenizer is not None:
            vocab_pad_id = self.tokenizer.pad_token_id or 0
            vocab_unk_id = getattr(self.tokenizer, "unk_token_id", None)
            if vocab_unk_id is not None and vocab_unk_id != vocab_pad_id:
                fallback_id = vocab_unk_id
            else:
                fallback_id = 1 if vocab_pad_id == 0 else 0
            for sent in sentences:
                encoded = self.tokenizer.encode(
                    sent,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    truncation=True,
                )
                if len(encoded) == 0:
                    token_ids = [fallback_id]
                else:
                    token_ids = encoded
                token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
                all_tokens.append(token_ids)
                all_token_strings.append(token_strs)
        else:
            # 문자 단위 토큰화 (이전 구현과 동일한 fallback 경로)
            for sent in sentences:
                chars = list(sent)
                token_ids = [ord(c) for c in chars]
                all_tokens.append(token_ids)
                all_token_strings.append(chars)
            vocab_pad_id = 0

        max_len = max(1, min(max((len(t) for t in all_tokens), default=0), self.max_length))

        padded_tokens: List[List[int]] = []
        for token_ids in all_tokens:
            padded = token_ids[:max_len] + [vocab_pad_id] * (max_len - len(token_ids))
            padded_tokens.append(padded)

        if len(padded_tokens) == 0:
            return torch.zeros((0, 0), dtype=torch.long), []

        return torch.tensor(padded_tokens, dtype=torch.long), all_token_strings
    
    def _generate_replacement_mask(
        self,
        token_strings: List[List[str]],
        sentences: List[str]
    ) -> torch.Tensor:
        """
        교체 가능 토큰 마스크 생성
        
        docs 명세:
        - 고정 토큰: 고유명사, 숫자, 특수 기호
        - 교체 가능: 일반 명사, 동사, 형용사
        
        Returns:
            mask: [num_sents, seq_len] 0=고정, 1=교체 가능
        """
        masks = []
        
        for tokens in token_strings:
            mask = []
            for token in tokens:
                # 간단한 규칙 기반 판정
                if self._is_replaceable(token):
                    mask.append(1)
                else:
                    mask.append(0)
            masks.append(mask)
        
        # 패딩 길이는 토큰화 길이와 동일하게 self.max_length로 클램프
        max_len = min(max(len(m) for m in masks) if masks else 0, self.max_length)
        padded_masks = []
        for mask in masks:
            padded = mask[:max_len] + [0] * (max_len - len(mask))
            padded_masks.append(padded)
        
        return torch.tensor(padded_masks, dtype=torch.long)
    
    def _is_replaceable(self, token: str) -> bool:
        """
        토큰 교체 가능 여부 판정
        
        개선: 논문 Section 6.2의 lexical space 확장
        - 한글 명사/동사 (2자 이상) 허용
        - 영어 단어 (3자 이상) 허용
        - 특수 토큰, 문장부호, 단독 조사만 제외
        """
        if not token or not token.strip():
            return False
        
        # 문장부호 제외
        if token in ".,!?;:()[]{}\"'":
            return False
        
        # BERT 특수 토큰 제외
        if token.startswith('[') and token.endswith(']'):
            return False
        if token.startswith('##'):
            return True
        
        # 한글 단어 (2자 이상) 허용
        if len(token) >= 2:
            has_hangul = any('가' <= c <= '힣' for c in token)
            if has_hangul:
                return True
        
        # 영어 단어 (3자 이상) 허용
        if token.isalpha() and len(token) >= 3:
            return True
        
        # 영문+숫자 혼합 (4자 이상) 허용
        if token.isalnum() and len(token) >= 4:
            return True
        
        return False
    
    def _build_topology(self, num_sentences: int, k: int = 3) -> torch.Tensor:
        """
        시간 순서 기반 topology 생성
        
        docs 명세:
        - 시간 순서: 이전/다음 문장을 이웃으로
        - k개 채우기
        
        Returns:
            topo_idx: [num_sentences, k] 이웃 인덱스
        """
        topo = []
        for i in range(num_sentences):
            neighbors = []
            
            # 이전 문장
            if i > 0:
                neighbors.append(i - 1)
            
            # 다음 문장
            if i < num_sentences - 1:
                neighbors.append(i + 1)
            
            # k개 채우기 (자기 자신으로)
            while len(neighbors) < k:
                neighbors.append(i)
            
            topo.append(neighbors[:k])
        
        return torch.tensor(topo, dtype=torch.long)

    def _build_document_tree(
        self,
        paragraph: str,
        sentences: List[str],
        token_strings: List[List[str]],
    ) -> DocumentTree:
        """
        문단을 일반 트리(document → sentence → token)로 표현.
        
        - docs 2장, 10장에서 정의한 트리 T=(V,E) 의 최소 구현:
          * type(v) ∈ {document, sentence, token}
        - 현재는 3레벨 구조이지만, 타입/노드 정의만 추가하면
          section/subsection/phrase 등으로 확장 가능.
        """
        nodes: List[TreeNode] = []
        node_id = 0

        # 루트 노드 (document)
        root_id = node_id
        nodes.append(
            TreeNode(
                id=root_id,
                type="document",
                parent=None,
                text=paragraph,
            )
        )
        node_id += 1

        # 문장 노드들
        sentence_node_ids: List[int] = []
        for sent in sentences:
            sid = node_id
            nodes.append(
                TreeNode(
                    id=sid,
                    type="sentence",
                    parent=root_id,
                    text=sent,
                )
            )
            sentence_node_ids.append(sid)
            node_id += 1

        # 토큰 노드들 (패딩 제외)
        for s_idx, sent_tokens in enumerate(token_strings):
            parent_sid = sentence_node_ids[s_idx]
            for tok in sent_tokens:
                # HF 토크나이저가 없는 경우 문자 단위 토큰도 그대로 사용
                if tok is None:
                    continue
                # 패딩 토큰(예: [PAD])은 텍스트 의미가 없으므로 노드로 만들지 않음
                if isinstance(tok, str) and tok.strip() == "":
                    continue
                tid = node_id
                nodes.append(
                    TreeNode(
                        id=tid,
                        type="token",
                        parent=parent_sid,
                        text=str(tok),
                    )
                )
                node_id += 1

        return DocumentTree(nodes=nodes, root_id=root_id)
```
---
## File: `reality_stone/python/reality_stone/utils/sampling.py`

```python
import torch

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty is None or float(penalty) == 1.0:
        return logits
    out = logits.clone()
    bsz = generated_ids.size(0)
    for bi in range(bsz):
        seen = torch.unique(generated_ids[bi]).long()
        vals = out[bi, seen]
        out[bi, seen] = torch.where(vals < 0, vals * float(penalty), vals / float(penalty))
    return out

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    out = logits
    if top_k is not None and int(top_k) > 0 and int(top_k) < out.size(-1):
        k = int(top_k)
        topk_vals, topk_idx = torch.topk(out, k, dim=-1)
        masked = torch.full_like(out, float("-inf"))
        masked.scatter_(1, topk_idx, topk_vals)
        out = masked
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = cum > float(top_p)
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        out = torch.full_like(out, float("-inf"))
        out.scatter_(1, sorted_idx, sorted_logits)
    return out


def sample_next_token(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    scores = logits
    scores = apply_repetition_penalty(scores, generated_ids, repetition_penalty)
    if temperature is not None and float(temperature) > 0 and float(temperature) != 1.0:
        scores = scores / float(max(temperature, 1e-8))
    scores = top_k_top_p_filter(scores, top_k=top_k, top_p=top_p)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```
---
## File: `reality_stone/python/reality_stone/utils/text_corpus.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class CorpusDoc:
    path: str
    text: str


def iter_text_files(
    root: str | Path,
    exts: Sequence[str] = (".md", ".txt"),
    max_bytes: int = 2_000_000,
) -> Iterator[Path]:
    root_p = Path(root)
    if root_p.is_file():
        if root_p.suffix.lower() in set(e.lower() for e in exts):
            yield root_p
        return
    for p in root_p.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in set(e.lower() for e in exts):
            continue
        try:
            if p.stat().st_size <= max_bytes:
                yield p
        except OSError:
            continue


def read_text_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return None
    # naive encoding fallback chain
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def load_corpus(
    roots: Sequence[str | Path],
    exts: Sequence[str] = (".md", ".txt"),
    max_docs: int = 2000,
    max_bytes_per_doc: int = 2_000_000,
) -> List[CorpusDoc]:
    out: List[CorpusDoc] = []
    seen: set[str] = set()
    for r in roots:
        for p in iter_text_files(r, exts=exts, max_bytes=max_bytes_per_doc):
            ps = str(p)
            if ps in seen:
                continue
            seen.add(ps)
            txt = read_text_file(p)
            if not txt:
                continue
            out.append(CorpusDoc(path=ps, text=txt))
            if len(out) >= max_docs:
                return out
    return out


def chunk_text(
    text: str,
    chunk_chars: int = 8000,
    overlap_chars: int = 1000,
) -> Iterator[str]:
    if chunk_chars <= 0:
        yield text
        return
    step = max(1, int(chunk_chars) - int(overlap_chars))
    n = len(text)
    i = 0
    while i < n:
        yield text[i : i + chunk_chars]
        i += step
```
---
## File: `reality_stone/README.md`

```markdown
# Reality Stone

Reality Stone is the vendored geometry backend used by Clarus Equation.

This copy is kept lean for integration: runtime code, Rust bindings, Python
fallbacks, and tests are retained; upstream repository metadata, build outputs,
experiments, and generated documentation are intentionally omitted.

## Layout

```text
reality_stone/
  src/                    Rust core and PyO3 bindings
  python/reality_stone/   Python API and fallback implementations
  python/reality_stone/clarus/
                          Clarus runtime and CE modules
  examples/               Single unified Clarus/Reality Stone demo
  tests/                  Rust and Python regression tests
  Cargo.toml              Rust crate metadata
  pyproject.toml          Python/maturin package metadata
```

## Python Usage

```python
import reality_stone as rs
from reality_stone.clarus.runtime import BrainRuntime

status = (rs.__version__, rs._has_rust_ext, rs._has_cuda)
```

When the compiled Rust extension is unavailable, `python/reality_stone/_rust.py`
and `python/reality_stone/_fallback.py` provide compatibility paths so Clarus can
still import and run CPU fallback flows.

## Validation

From the repository root:

```powershell
$env:PYTHONPATH = "reality_stone/python"
.\.venv\Scripts\python.exe -m pytest -q reality_stone\tests\layer reality_stone\tests\test_unified_riemannian.py reality_stone\tests\llm\test_metric_attention.py reality_stone\tests\llm\test_metric_router.py reality_stone\tests\api\test_pipeline_api.py
cargo test --manifest-path reality_stone\Cargo.toml --no-default-features
.\.venv\Scripts\python.exe -B reality_stone\examples\unified_clarus_demo.py
```

## Native Build Note

This nested package metadata builds the optional Reality Stone extension as
`reality_stone._rust`. The repository-root `pyproject.toml` builds the optional
Clarus extension as `reality_stone.clarus._rust` for the unified checkout.
Both paths have Python fallbacks, so tests and the unified demo do not require a
native build.
```
---
## File: `reality_stone/src/bindings/bellman.rs`

```rust
use crate::layers::bellman::{
    compute_diagonal_geodesic_backward, compute_diagonal_geodesic_update,
};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Applies the Bellman-Lagrangian Geodesic flow to the input features.
///
/// This layer deforms the input state based on a learnable metric (here approximated by Sigmoid)
/// to follow the geodesic path of the induced manifold.
///
/// Args:
///     x: Input tensor (Batch x Dim)
///     dt: Time step for the flow (default=0.1). Controls how far along the geodesic to move.
#[pyfunction]
#[pyo3(name = "bellman_geodesic_forward")]
pub fn bellman_geodesic_forward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    dt: Option<f64>,
) -> &'py PyArray2<f64> {
    let input = x.as_array();
    let step = dt.unwrap_or(0.1);

    let output = compute_diagonal_geodesic_update(&input, step);

    output.into_pyarray(py)
}

/// Backward pass for the Bellman-Lagrangian Geodesic flow.
#[pyfunction]
#[pyo3(name = "bellman_geodesic_backward")]
pub fn bellman_geodesic_backward<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f64>,
    input: PyReadonlyArray2<f64>,
    dt: Option<f64>,
) -> &'py PyArray2<f64> {
    let grad = grad_output.as_array();
    let inp = input.as_array();
    let step = dt.unwrap_or(0.1);

    let grad_input = compute_diagonal_geodesic_backward(&grad, &inp, step);

    grad_input.into_pyarray(py)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bellman_geodesic_forward, m)?)?;
    m.add_function(wrap_pyfunction!(bellman_geodesic_backward, m)?)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/diffusion.rs`

```rust
use crate::layers::diffusion::RiemannianDiffusion;
use pyo3::prelude::*;

// CUDA FFI 선언
#[cfg(feature = "cuda")]
extern "C" {
    fn riemannian_diffusion_step_cuda(
        h: *const f32,
        flow: *const f32,
        output: *mut f32,
        alpha: f32,
        dt: f32,
        n: i32,
        d: i32,
        stream: *mut std::ffi::c_void,
    );
}

#[pyclass]
pub struct PyRiemannianDiffusion {
    inner: RiemannianDiffusion,
}

#[pymethods]
impl PyRiemannianDiffusion {
    #[new]
    pub fn new(dim: usize, alpha: f32, dt: f32) -> Self {
        Self {
            inner: RiemannianDiffusion::new(dim, alpha, dt),
        }
    }

    /// CUDA 전용 Step 함수 (Zero-copy)
    /// PyTorch Tensor의 data_ptr을 직접 받습니다.
    #[cfg(feature = "cuda")]
    pub fn step_cuda(
        &self,
        h_ptr: u64,
        flow_ptr: u64,
        out_ptr: u64,
        n: i32,
        d: i32,
    ) -> PyResult<()> {
        unsafe {
            riemannian_diffusion_step_cuda(
                h_ptr as *const f32,
                flow_ptr as *const f32,
                out_ptr as *mut f32,
                self.inner.alpha,
                self.inner.dt,
                n,
                d,
                std::ptr::null_mut(), // Default stream
            );
        }
        Ok(())
    }

    // CPU Fallback (기존 코드 유지)
    pub fn step_cpu<'py>(
        &self,
        py: Python<'py>,
        h: numpy::PyReadonlyArray2<f32>,
        flow_field: numpy::PyReadonlyArray2<f32>,
    ) -> &'py numpy::PyArray2<f32> {
        use numpy::IntoPyArray;
        let h_arr = h.as_array();
        let flow_arr = flow_field.as_array();
        let result = self.inner.step(&h_arr, &flow_arr);
        result.into_pyarray(py)
    }
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRiemannianDiffusion>()?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/extraction.rs`

```rust
use crate::ops::extraction;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "extract_metric_cuda")]
pub fn extract_metric_cuda_py<'py>(
    py: Python<'py>,
    w: PyReadonlyArray2<f32>,
    calibration_data: PyReadonlyArray2<f32>,
    target_dim: usize,
    num_steps: usize,
    curvature: f32,
    lr: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, &'py PyArray2<f32>) {
    let w_view = w.as_array();
    let calib_view = calibration_data.as_array();

    let (u, g, v) = py.allow_threads(move || {
        extraction::extract_metric_cuda(w_view, calib_view, target_dim, num_steps, curvature, lr)
    });

    (u.into_pyarray(py), g.into_pyarray(py), v.into_pyarray(py))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_metric_cuda_py, m)?)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/geodesic_attention.rs`

```rust
#[cfg(feature = "cuda")]
use ndarray::Array4;
#[cfg(feature = "cuda")]
use numpy::IntoPyArray;
use numpy::{PyArray4, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;

#[cfg(feature = "cuda")]
extern "C" {
    fn geodesic_topk_attention_cuda(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        idx: *const i64,
        l: *const f32,
        c: f32,
        tau: f32,
        b: i32,
        h: i32,
        t: i32,
        s: i32,
        k_topk: i32,
        d_h: i32,
        d_v: i32,
        out: *mut f32,
    );

    // Low-level CUDA entry point; actual C symbol name is `batched_cholesky_cuda`.
    #[link_name = "batched_cholesky_cuda"]
    fn batched_cholesky_cuda_ffi(A: *const f32, L: *mut f32, batch_count: i32, d: i32);
}

/// Fused Geodesic Top-k Attention (CUDA)
///
/// 한 커널에서 SPD metric 적용, geodesic distance 계산, softmax, aggregation을 모두 수행
///
/// # Arguments
/// * `q` - Query tensor [B, H, T, d_h]
/// * `k` - Key tensor [B, H, S, d_h]
/// * `v` - Value tensor [B, H, S, d_v]
/// * `idx` - Top-k indices [B, T, K]
/// * `l_factor` - SPD Cholesky factor [d_h, d_h]
/// * `c` - Curvature (default: 1.0)
/// * `tau` - Temperature (default: 1.0)
///
/// # Returns
/// * Output tensor [B, H, T, d_v]
#[pyfunction]
#[pyo3(signature = (q, k, v, idx, l_factor, c=1.0, tau=1.0))]
pub fn geodesic_topk_attention(
    py: Python,
    q: PyReadonlyArray4<f32>,
    k: PyReadonlyArray4<f32>,
    v: PyReadonlyArray4<f32>,
    idx: PyReadonlyArray3<i64>,
    l_factor: PyReadonlyArray2<f32>,
    c: f32,
    tau: f32,
) -> PyResult<Py<PyArray4<f32>>> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (&py, &q, &k, &v, &idx, &l_factor, c, tau);
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        // Get dimensions
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();
        let idx_shape = idx.shape();

        let b = q_shape[0] as i32;
        let h = q_shape[1] as i32;
        let t = q_shape[2] as i32;
        let d_h = q_shape[3] as i32;

        let s = k_shape[2] as i32;
        let d_v = v_shape[3] as i32;
        let k_topk = idx_shape[2] as i32;

        // Validate shapes
        if k_shape[0] != b as usize || k_shape[1] != h as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "K shape mismatch: expected [{}, {}, ?, {}], got {:?}",
                b, h, d_h, k_shape
            )));
        }
        if v_shape[0] != b as usize || v_shape[1] != h as usize || v_shape[2] != s as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "V shape mismatch: expected [{}, {}, {}, {}], got {:?}",
                b, h, s, d_v, v_shape
            )));
        }
        if idx_shape[0] != b as usize || idx_shape[1] != t as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "idx shape mismatch: expected [{}, {}, {}], got {:?}",
                b, t, k_topk, idx_shape
            )));
        }

        // Get raw pointers
        let q_ptr = q.as_slice()?.as_ptr();
        let k_ptr = k.as_slice()?.as_ptr();
        let v_ptr = v.as_slice()?.as_ptr();
        let idx_ptr = idx.as_slice()?.as_ptr();
        let l_ptr = l_factor.as_slice()?.as_ptr();

        // Allocate output buffer (flattened)
        let out_size = (b * h * t * d_v) as usize;
        let mut out_vec = vec![0.0f32; out_size];

        // Call CUDA kernel
        unsafe {
            geodesic_topk_attention_cuda(
                q_ptr,
                k_ptr,
                v_ptr,
                idx_ptr,
                l_ptr,
                c,
                tau,
                b,
                h,
                t,
                s,
                k_topk,
                d_h,
                d_v,
                out_vec.as_mut_ptr(),
            );
        }

        // Convert to numpy Array4 and then to PyArray4
        let out_shape = (b as usize, h as usize, t as usize, d_v as usize);
        let out_array = Array4::from_shape_vec(out_shape, out_vec).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to reshape geodesic_topk_attention output: {e}"
            ))
        })?;
        Ok(out_array.into_pyarray(py).to_owned())
    }
}

/// Batched SPD Cholesky Decomposition (CUDA)
///
/// # Arguments
/// * `g` - SPD matrices [B, T, d, d]
///
/// # Returns
/// * Cholesky factors [B, T, d, d]
#[pyfunction]
pub fn batched_cholesky_cuda(
    _py: Python,
    _g: PyReadonlyArray4<f32>,
) -> PyResult<Py<PyArray4<f32>>> {
    #[cfg(not(feature = "cuda"))]
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        let g_shape = _g.shape();
        if g_shape[2] != g_shape[3] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input must be square matrices, got shape {:?}",
                g_shape
            )));
        }

        let batch_count = (g_shape[0] * g_shape[1]) as i32;
        let d = g_shape[2] as i32;

        let g_ptr = _g.as_slice()?.as_ptr();

        // Allocate output L
        let l_size = (batch_count * d * d) as usize;
        let mut l_vec = vec![0.0f32; l_size];

        unsafe {
            batched_cholesky_cuda_ffi(g_ptr, l_vec.as_mut_ptr(), batch_count, d);
        }

        let out_shape = (g_shape[0], g_shape[1], g_shape[2], g_shape[3]);
        let out_array = Array4::from_shape_vec(out_shape, l_vec).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to reshape cholesky output: {e}"
            ))
        })?;

        Ok(out_array.into_pyarray(_py).to_owned())
    }
}

pub fn register(m: &PyModule) -> PyResult<()> {
    let sub = PyModule::new(m.py(), "geodesic")?;
    sub.add_function(wrap_pyfunction!(geodesic_topk_attention, sub)?)?;
    sub.add_function(wrap_pyfunction!(batched_cholesky_cuda, sub)?)?;
    m.add_submodule(sub)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/hyper_metric.rs`

```rust
use crate::layers::hyper_metric::{HyperMetric, TinyMLP};
use crate::layers::symplectic::{SymplecticLayer, SymplecticState};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct PyHyperMetric {
    inner: HyperMetric,
}

#[pymethods]
impl PyHyperMetric {
    #[new]
    #[pyo3(signature = (u_global, v_global, w1, b1, w2, b2))]
    pub fn new(
        u_global: PyReadonlyArray2<f32>,
        v_global: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        b1: PyReadonlyArray1<f32>,
        w2: PyReadonlyArray2<f32>,
        b2: PyReadonlyArray1<f32>,
    ) -> Self {
        let mlp = TinyMLP::from_weights(
            w1.as_array().to_owned(),
            b1.as_array().to_owned(),
            w2.as_array().to_owned(),
            b2.as_array().to_owned(),
        );

        let inner = HyperMetric::from_components(
            u_global.as_array().to_owned(),
            v_global.as_array().to_owned(),
            mlp,
        );

        Self { inner }
    }

    pub fn generate_core<'py>(
        &self,
        py: Python<'py>,
        layer_emb: PyReadonlyArray1<f32>,
    ) -> &'py PyArray2<f32> {
        let core = self.inner.generate_core(&layer_emb.as_array().to_owned());
        core.into_pyarray(py)
    }

    pub fn project_forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        layer_emb: PyReadonlyArray1<f32>,
    ) -> &'py PyArray2<f32> {
        let out = self
            .inner
            .project_forward(&x.as_array().to_owned(), &layer_emb.as_array().to_owned());
        out.into_pyarray(py)
    }
}

#[pyclass]
pub struct PySymplecticLayer {
    inner: SymplecticLayer,
}

#[pymethods]
impl PySymplecticLayer {
    #[new]
    pub fn new(
        layer_idx: usize,
        layer_emb: PyReadonlyArray1<f32>,
        hyper_metric: &PyHyperMetric,
        dt: f32,
    ) -> Self {
        let inner = SymplecticLayer::new(
            layer_idx,
            layer_emb.as_array().to_owned(),
            hyper_metric.inner.clone(),
            dt,
        );
        Self { inner }
    }

    pub fn step<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<f32>,
        p: PyReadonlyArray2<f32>,
        x_input: PyReadonlyArray2<f32>,
    ) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
        let mut state = SymplecticState {
            q: q.as_array().to_owned(),
            p: p.as_array().to_owned(),
        };

        let _ = self.inner.step(&mut state, &x_input.as_array().to_owned());

        (state.q.into_pyarray(py), state.p.into_pyarray(py))
    }
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHyperMetric>()?;
    m.add_class::<PySymplecticLayer>()?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/klein.rs`

```rust
use crate::layers::klein;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn klein_add<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = klein::klein_add(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_scalar<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    r: f32,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let result = klein::klein_scalar(&u_arr, c, r);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_distance<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray1<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = klein::klein_distance(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_to_poincare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = klein::klein_to_poincare(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_to_lorentz<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = klein::klein_to_lorentz(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_layer_forward<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = klein::klein_layer_forward(&u_arr, &v_arr, c, t);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_ball_layer_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let (grad_u, grad_v) = klein::klein_layer_backward(&grad_output_arr, &u_arr, &v_arr, c, t);
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py))
}

#[pyfunction]
fn from_poincare_dynamic_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, f32) {
    let x_view = x.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let c = dynamic_c.compute_c();
    let result = klein::from_poincare(&x_view, c);
    (result.into_pyarray(py), c)
}

#[pyfunction]
fn from_poincare_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<'py, f32>,
    x: PyReadonlyArray2<'py, f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (Py<PyArray2<f32>>, f32) {
    let grad_output_view = grad_output.as_array();
    let x_view = x.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let c = dynamic_c.compute_c();

    let grad_x = Array2::zeros((x.shape()[0], x.shape()[1]));

    let grad_c_tensor = klein::from_poincare_grad_c(&x_view, c);
    let grad_c = (&grad_output_view * &grad_c_tensor).sum();
    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;

    (grad_x.into_pyarray(py).to_owned(), grad_kappa)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn klein_distance_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    klein::cuda::klein_distance_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn klein_layer_forward_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    klein::cuda::klein_layer_forward_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn klein_ball_layer_backward_cuda(
    grad_output: usize,
    u: usize,
    v: usize,
    grad_u: usize,
    grad_v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    klein::cuda::klein_layer_backward_cuda(
        grad_output as *const f32,
        u as *const f32,
        v as *const f32,
        grad_u as *mut f32,
        grad_v as *mut f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(klein_add, m)?)?;
    m.add_function(wrap_pyfunction!(klein_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(klein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(klein_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(klein_to_lorentz, m)?)?;
    m.add_function(wrap_pyfunction!(klein_layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(klein_ball_layer_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_poincare_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_poincare_dynamic_backward_cpu, m)?)?;

    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(klein_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(klein_layer_forward_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(klein_ball_layer_backward_cuda, m)?)?;
    }

    Ok(())
}
```
---
## File: `reality_stone/src/bindings/lorentz.rs`

```rust
use crate::layers::lorentz;
use crate::ops::{DynamicCurvature, LayerWiseDynamicCurvature};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn lorentz_add<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> Bound<'py, PyArray2<f32>> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_add(&u_arr, &v_arr, c);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_scalar<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    r: f32,
    c: f32,
) -> Bound<'py, PyArray2<f32>> {
    let u_arr = u.as_array();
    let result = lorentz::lorentz_scalar(&u_arr, c, r);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_distance<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> Bound<'py, PyArray1<f32>> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_distance(&u_arr, &v_arr, c);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_inner<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray1<f32>> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_inner(&u_arr, &v_arr);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_to_poincare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> Bound<'py, PyArray2<f32>> {
    let x_arr = x.as_array();
    let result = lorentz::lorentz_to_poincare(&x_arr, c);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_to_klein<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> Bound<'py, PyArray2<f32>> {
    let x_arr = x.as_array();
    let result = lorentz::lorentz_to_klein(&x_arr, c);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_layer_forward<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> Bound<'py, PyArray2<f32>> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_layer_forward(&u_arr, &v_arr, c, t);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_ball_layer_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let (grad_u, grad_v) = lorentz::lorentz_layer_backward(&grad_output_arr, &u_arr, &v_arr, c, t);
    (grad_u.into_pyarray_bound(py), grad_v.into_pyarray_bound(py))
}

#[pyfunction]
pub fn lorentz_layer_dynamic_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = DynamicCurvature::new(kappa, c_min, c_max);
    let (out, c) = lorentz::lorentz_layer_dynamic(&u_arr, &v_arr, &dynamic_c, t);
    (out.into_pyarray_bound(py), c)
}

#[pyfunction]
pub fn lorentz_layer_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = DynamicCurvature::new(kappa, c_min, c_max);
    let (gu, gv, gk) =
        lorentz::lorentz_layer_dynamic_backward(&grad_output_arr, &u_arr, &v_arr, &dynamic_c, t);
    (gu.into_pyarray_bound(py), gv.into_pyarray_bound(py), gk)
}

#[pyfunction]
pub fn lorentz_layer_layerwise_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let lw = LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (out, c) = lorentz::lorentz_layer_layerwise(&u_arr, &v_arr, &lw, layer_idx, t);
    (out.into_pyarray_bound(py), c)
}

#[pyfunction]
pub fn lorentz_layer_layerwise_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let lw = LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (gu, gv, gk) = lorentz::lorentz_layer_layerwise_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &lw,
        layer_idx,
        t,
    );
    (gu.into_pyarray_bound(py), gv.into_pyarray_bound(py), gk)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn lorentz_distance_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    lorentz::cuda::lorentz_distance_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn lorentz_layer_forward_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    lorentz::cuda::lorentz_layer_forward_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn lorentz_ball_layer_backward_cuda(
    grad_output: usize,
    u: usize,
    v: usize,
    grad_u: usize,
    grad_v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    lorentz::cuda::lorentz_layer_backward_cuda(
        grad_output as *const f32,
        u as *const f32,
        v as *const f32,
        grad_u as *mut f32,
        grad_v as *mut f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

#[pyfunction]
fn from_poincare_dynamic_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (Bound<'py, PyArray2<f32>>, f32) {
    let x_view = x.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let c = dynamic_c.compute_c();
    let result = lorentz::from_poincare(&x_view, c);
    (result.into_pyarray_bound(py), c)
}

#[pyfunction]
fn from_poincare_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<'py, f32>,
    x: PyReadonlyArray2<'py, f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (Py<PyArray2<f32>>, f32) {
    let grad_output_view = grad_output.as_array();
    let x_view = x.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let c = dynamic_c.compute_c();

    let grad_x = Array2::zeros((x.shape()[0], x.shape()[1]));

    let grad_c_tensor = lorentz::from_poincare_grad_c(&x_view, c);
    let grad_c = (&grad_output_view * &grad_c_tensor).sum();
    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;

    (grad_x.into_pyarray_bound(py).unbind(), grad_kappa)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lorentz_add, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_distance, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_inner, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_klein, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_ball_layer_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_dynamic_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_layerwise_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_layerwise_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_poincare_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_poincare_dynamic_backward_cpu, m)?)?;

    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(lorentz_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(lorentz_layer_forward_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(lorentz_ball_layer_backward_cuda, m)?)?;
    }

    Ok(())
}
```
---
## File: `reality_stone/src/bindings/macros.rs`

```rust
// src/bindings/macros.rs

/// PyFunction 바인딩 생성을 위한 매크로
///
/// 사용법:
/// `create_binding!(파이썬_함수명, Rust_함수_경로, [인자1, 인자2, ...], 반환_타입);`
///
/// 예시:
/// `create_binding!(poincare_add, crate::layers::poincare::poincare_add, [u, v, c], PyArray2);`

#[macro_export]
macro_rules! create_binding {
    // (u, v, c) -> Array2<f32>
    ($py_fn_name:ident, $rust_fn:path, [u, v, c], PyArray2) => {
        #[pyo3::prelude::pyfunction]
        pub fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            u: numpy::PyReadonlyArray2<f32>,
            v: numpy::PyReadonlyArray2<f32>,
            c: f32,
        ) -> &'py numpy::PyArray2<f32> {
            let u_arr = u.as_array();
            let v_arr = v.as_array();
            let result = $rust_fn(&u_arr, &v_arr, c);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (u, r, c) -> Array2<f32>
    ($py_fn_name:ident, $rust_fn:path, [u, r, c], PyArray2) => {
        #[pyo3::prelude::pyfunction]
        pub fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            u: numpy::PyReadonlyArray2<f32>,
            r: f32,
            c: f32,
        ) -> &'py numpy::PyArray2<f32> {
            let u_arr = u.as_array();
            let result = $rust_fn(&u_arr, r, c);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (u, v, c) -> Array1<f32>
    ($py_fn_name:ident, $rust_fn:path, [u, v, c], PyArray1) => {
        #[pyo3::prelude::pyfunction]
        pub fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            u: numpy::PyReadonlyArray2<f32>,
            v: numpy::PyReadonlyArray2<f32>,
            c: f32,
        ) -> &'py numpy::PyArray1<f32> {
            let u_arr = u.as_array();
            let v_arr = v.as_array();
            let result = $rust_fn(&u_arr, &v_arr, c);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (u, v, c, eps) -> Array1<f32>
    ($py_fn_name:ident, $rust_fn:path, [u, v, c, eps], PyArray1) => {
        #[pyo3::prelude::pyfunction]
        pub fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            u: numpy::PyReadonlyArray2<f32>,
            v: numpy::PyReadonlyArray2<f32>,
            c: f32,
            eps: f32,
        ) -> &'py numpy::PyArray1<f32> {
            let u_arr = u.as_array();
            let v_arr = v.as_array();
            let result = $rust_fn(&u_arr, &v_arr, c, eps);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (x, c) -> Array2<f32>
    ($py_fn_name:ident, $rust_fn:path, [x, c], PyArray2) => {
        #[pyo3::prelude::pyfunction]
        pub fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            x: numpy::PyReadonlyArray2<f32>,
            c: f32,
        ) -> &'py numpy::PyArray2<f32> {
            let x_arr = x.as_array();
            let result = $rust_fn(&x_arr, c);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (x, v, c, eps) -> Array2<f32>
    ($py_fn_name:ident, $rust_fn:path, [x, v, c, eps], PyArray2) => {
        #[pyo3::prelude::pyfunction]
        pub fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            x: numpy::PyReadonlyArray2<f32>,
            v: numpy::PyReadonlyArray2<f32>,
            c: f32,
            eps: f32,
        ) -> &'py numpy::PyArray2<f32> {
            let x_arr = x.as_array();
            let v_arr = v.as_array();
            let result = $rust_fn(&x_arr, &v_arr, c, eps);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (x, y, c, eps) -> Array2<f32>
    ($py_fn_name:ident, $rust_fn:path, [x, y, c, eps], PyArray2) => {
        #[pyo3::prelude::pyfunction]
        pub fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            x: numpy::PyReadonlyArray2<f32>,
            y: numpy::PyReadonlyArray2<f32>,
            c: f32,
            eps: f32,
        ) -> &'py numpy::PyArray2<f32> {
            let x_arr = x.as_array();
            let y_arr = y.as_array();
            let result = $rust_fn(&x_arr, &y_arr, c, eps);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };
}
```
---
## File: `reality_stone/src/bindings/memory.rs`

```rust
use crate::layers::memory::GeodesicMemory;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass]
pub struct PyGeodesicMemory {
    inner: GeodesicMemory,
}

#[pymethods]
impl PyGeodesicMemory {
    #[new]
    pub fn new(d_model: usize, threshold: f32) -> Self {
        Self {
            inner: GeodesicMemory::new(d_model, threshold),
        }
    }

    pub fn push(&mut self, t: usize, x: PyReadonlyArray1<f32>) -> bool {
        self.inner.push(t, x.as_array())
    }

    pub fn query<'py>(&self, py: Python<'py>, t: f32) -> &'py PyArray1<f32> {
        let result = self.inner.query(t);
        result.into_pyarray(py)
    }

    pub fn get_stats(&self) -> (usize, usize, f32) {
        self.inner.get_compression_stats()
    }

    pub fn reset(&mut self) {
        let d = self.inner.d_model;
        let th = self.inner.threshold;
        self.inner = GeodesicMemory::new(d, th);
    }
}
```
---
## File: `reality_stone/src/bindings/metrikey.rs`

```rust
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyfunction]
pub fn householder_chain_apply_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: PyReadonlyArray1<f32>,
) -> Py<PyArray1<f32>> {
    Python::with_gil(|py| {
        let x = x.as_array().to_owned();
        let y = crate::ops::householder_chain_apply_from_key(key, dim, num, &x);
        PyArray1::from_vec(py, y.to_vec()).to_owned()
    })
}

#[pyfunction]
pub fn householder_chain_apply_transpose_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: PyReadonlyArray1<f32>,
) -> Py<PyArray1<f32>> {
    Python::with_gil(|py| {
        let x = x.as_array().to_owned();
        let y = crate::ops::householder_chain_apply_transpose_from_key(key, dim, num, &x);
        PyArray1::from_vec(py, y.to_vec()).to_owned()
    })
}

#[pyfunction]
pub fn givens_chain_apply_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: PyReadonlyArray1<f32>,
) -> Py<PyArray1<f32>> {
    Python::with_gil(|py| {
        let x = x.as_array().to_owned();
        let y = crate::ops::givens_chain_apply_from_key(key, dim, num, &x);
        PyArray1::from_vec(py, y.to_vec()).to_owned()
    })
}

#[pyfunction]
pub fn lowrank_plus_diag_apply_from_key(
    key_u: &str,
    key_v: &str,
    s_diag: PyReadonlyArray1<f32>,
    rank: usize,
    x: PyReadonlyArray1<f32>,
) -> Py<PyArray1<f32>> {
    Python::with_gil(|py| {
        let s = s_diag.as_array().to_owned();
        let x = x.as_array().to_owned();
        let y = crate::ops::lowrank_plus_diag_apply_from_key(key_u, key_v, &s, rank, &x);
        PyArray1::from_vec(py, y.to_vec()).to_owned()
    })
}
#[pyfunction]
pub fn rotate_metric_factor_block<'py>(
    py: Python<'py>,
    key: &str,
    l: PyReadonlyArray2<f32>,
    global_dim: usize,
) -> &'py PyArray2<f32> {
    let l = l.as_array().to_owned();
    let out = crate::ops::rotate_metric_factor_block(key, &l, global_dim);
    PyArray2::from_owned_array(py, out)
}

#[pyfunction]
pub fn spd_metric_from_key_weighted<'py>(
    py: Python<'py>,
    key: &str,
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
    mass: f32,
) -> &'py PyArray2<f32> {
    let g = crate::ops::spd_metric_from_key_weighted(key, dim, min_lambda, max_lambda, mass);
    PyArray2::from_owned_array(py, g)
}

#[pyfunction]
pub fn compose_layers_gravity<'py>(
    py: Python<'py>,
    keys: Vec<String>,
    masses: Vec<f32>,
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> &'py PyArray2<f32> {
    let t = crate::ops::compose_layers_gravity(&keys, &masses, dim, min_lambda, max_lambda);
    PyArray2::from_owned_array(py, t)
}

// f64 high-precision variants
#[pyfunction]
pub fn compose_layers_gravity_f64<'py>(
    py: Python<'py>,
    keys: Vec<String>,
    masses: Vec<f64>,
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
) -> &'py PyArray2<f64> {
    let t = crate::ops::compose_layers_gravity_f64(&keys, &masses, dim, min_lambda, max_lambda);
    PyArray2::from_owned_array(py, t)
}

#[pyfunction]
pub fn apply_linear_f64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f64>,
    vecs: PyReadonlyArray2<f64>,
) -> &'py PyArray2<f64> {
    let matrix = matrix.as_array().to_owned();
    let vecs = vecs.as_array().to_owned();
    let out = crate::ops::apply_linear_f64(&matrix, &vecs);
    PyArray2::from_owned_array(py, out)
}

#[pyfunction]
pub fn effective_metric_from_transform_f64<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
) -> &'py PyArray2<f64> {
    let t_arr = t.as_array().to_owned();
    let g = crate::ops::effective_metric_from_transform_f64(&t_arr);
    PyArray2::from_owned_array(py, g)
}

#[pyfunction]
pub fn metric_factor_cholesky_f64<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, f64>,
) -> &'py PyArray2<f64> {
    let g_arr = g.as_array().to_owned();
    let u = crate::ops::metric_factor_cholesky_f64(&g_arr);
    PyArray2::from_owned_array(py, u)
}

#[pyfunction]
pub fn compose_layers_gravity_compact_f64<'py>(
    py: Python<'py>,
    master_key: &str,
    num_layers: usize,
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
    mass_base: f64,
    mass_step: f64,
) -> &'py PyArray2<f64> {
    let t = crate::ops::compose_layers_gravity_compact_f64(
        master_key, num_layers, dim, min_lambda, max_lambda, mass_base, mass_step,
    );
    PyArray2::from_owned_array(py, t)
}

// Collapsed transform wrapper (F32) for order-preserving layer composition
#[pyclass]
pub struct CollapsedTransformF32 {
    t: Array2<f32>,
    dim: usize,
}

// High-precision collapsed transform (F64) for strict verification/inference
#[pyclass]
pub struct CollapsedTransformF64 {
    t: ndarray::Array2<f64>,
    dim: usize,
}

#[pymethods]
impl CollapsedTransformF64 {
    #[new]
    fn new(t: PyReadonlyArray2<f64>) -> Self {
        let t_arr = t.as_array().to_owned();
        let dim = t_arr.dim().1;
        Self { t: t_arr, dim }
    }

    #[staticmethod]
    fn from_keys(
        keys: Vec<String>,
        masses: Vec<f64>,
        dim: usize,
        min_lambda: f64,
        max_lambda: f64,
    ) -> Self {
        let t = crate::ops::compose_layers_gravity_f64(&keys, &masses, dim, min_lambda, max_lambda);
        Self { t, dim }
    }

    #[staticmethod]
    fn from_master_key_compact(
        master_key: &str,
        num_layers: usize,
        dim: usize,
        min_lambda: f64,
        max_lambda: f64,
        mass_base: f64,
        mass_step: f64,
    ) -> Self {
        let t = crate::ops::compose_layers_gravity_compact_f64(
            master_key, num_layers, dim, min_lambda, max_lambda, mass_base, mass_step,
        );
        Self { t, dim }
    }

    fn apply<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> &'py PyArray2<f64> {
        let x_arr = x.as_array().to_owned();
        let out = crate::ops::apply_linear_f64(&self.t, &x_arr);
        PyArray2::from_owned_array(py, out)
    }

    fn matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        PyArray2::from_owned_array(py, self.t.clone())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }
}

#[pymethods]
impl CollapsedTransformF32 {
    #[new]
    fn new(t: PyReadonlyArray2<f32>) -> Self {
        let t_arr = t.as_array().to_owned();
        let dim = t_arr.dim().1;
        Self { t: t_arr, dim }
    }

    #[staticmethod]
    fn from_keys(
        keys: Vec<String>,
        masses: Vec<f32>,
        dim: usize,
        min_lambda: f32,
        max_lambda: f32,
    ) -> Self {
        let t = crate::ops::compose_layers_gravity(&keys, &masses, dim, min_lambda, max_lambda);
        Self { t, dim }
    }

    #[staticmethod]
    fn from_master_key_compact(
        master_key: &str,
        num_layers: usize,
        dim: usize,
        min_lambda: f64,
        max_lambda: f64,
        mass_base: f64,
        mass_step: f64,
    ) -> Self {
        // Use high-precision f64 path for composition, then cast to f32 for runtime apply
        let t64 = crate::ops::compose_layers_gravity_compact_f64(
            master_key, num_layers, dim, min_lambda, max_lambda, mass_base, mass_step,
        );
        let t = t64.mapv(|v| v as f32);
        Self { t, dim }
    }

    /// Apply the collapsed transform to a batch of row-vectors X: (batch, dim)
    fn apply<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>) -> &'py PyArray2<f32> {
        let x_arr = x.as_array().to_owned();
        let out = crate::ops::apply_linear(&self.t, &x_arr);
        PyArray2::from_owned_array(py, out)
    }

    /// Return the transform matrix (dim, dim)
    fn matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        PyArray2::from_owned_array(py, self.t.clone())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }
}

#[pyfunction]
pub fn spd_metric_from_key<'py>(
    py: Python<'py>,
    key: &str,
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> &'py PyArray2<f32> {
    let g = crate::ops::spd_metric_from_key(key, dim, min_lambda, max_lambda);
    PyArray2::from_owned_array(py, g)
}

#[pyfunction]
pub fn metric_factor_cholesky<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<f32>,
) -> &'py PyArray2<f32> {
    let g = g.as_array().to_owned();
    let l = crate::ops::metric_factor_cholesky(&g);
    PyArray2::from_owned_array(py, l)
}

#[pyfunction]
pub fn mahalanobis_distance_sq_g(
    x: PyReadonlyArray1<f32>,
    y: PyReadonlyArray1<f32>,
    g: PyReadonlyArray2<f32>,
) -> f32 {
    let x = x.as_array().to_owned();
    let y = y.as_array().to_owned();
    let g = g.as_array().to_owned();
    crate::ops::mahalanobis_distance_sq_g(&x, &y, &g)
}

#[pyfunction]
pub fn mahalanobis_distance_sq_l(
    x: PyReadonlyArray1<f32>,
    y: PyReadonlyArray1<f32>,
    l: PyReadonlyArray2<f32>,
) -> f32 {
    let x = x.as_array().to_owned();
    let y = y.as_array().to_owned();
    let l = l.as_array().to_owned();
    crate::ops::mahalanobis_distance_sq_l(&x, &y, &l)
}

#[pyfunction]
pub fn block_orthogonal_from_key<'py>(
    py: Python<'py>,
    key: &str,
    global_dim: usize,
    dept_dim: usize,
) -> &'py PyArray2<f32> {
    let q = crate::ops::block_orthogonal_from_key(key, global_dim, dept_dim);
    PyArray2::from_owned_array(py, q)
}

#[pyfunction]
pub fn spd_block_metric_from_key<'py>(
    py: Python<'py>,
    key: &str,
    global_dim: usize,
    dept_dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> &'py PyArray2<f32> {
    let g =
        crate::ops::spd_block_metric_from_key(key, global_dim, dept_dim, min_lambda, max_lambda);
    PyArray2::from_owned_array(py, g)
}

#[pyfunction]
pub fn compose_layers_order_preserving<'py>(
    py: Python<'py>,
    layers: Vec<PyReadonlyArray2<f32>>,
) -> &'py PyArray2<f32> {
    let mut rust_layers = Vec::with_capacity(layers.len());
    for a in layers.into_iter() {
        rust_layers.push(a.as_array().to_owned());
    }
    let t = crate::ops::compose_layers_order_preserving(&rust_layers);
    PyArray2::from_owned_array(py, t)
}

#[pyfunction]
pub fn apply_linear<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f32>,
    vecs: PyReadonlyArray2<f32>,
) -> &'py PyArray2<f32> {
    let matrix = matrix.as_array().to_owned();
    let vecs = vecs.as_array().to_owned();
    let out = crate::ops::apply_linear(&matrix, &vecs);
    PyArray2::from_owned_array(py, out)
}

pub fn init_module(_py: Python, m: &PyModule) -> PyResult<()> {
    let sub = PyModule::new(_py, "metrikey")?;
    sub.add_function(wrap_pyfunction!(spd_metric_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(metric_factor_cholesky, sub)?)?;
    sub.add_function(wrap_pyfunction!(mahalanobis_distance_sq_g, sub)?)?;
    sub.add_function(wrap_pyfunction!(mahalanobis_distance_sq_l, sub)?)?;
    sub.add_function(wrap_pyfunction!(block_orthogonal_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(spd_block_metric_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(spd_metric_from_key_weighted, sub)?)?;
    sub.add_function(wrap_pyfunction!(compose_layers_order_preserving, sub)?)?;
    sub.add_function(wrap_pyfunction!(compose_layers_gravity, sub)?)?;
    sub.add_function(wrap_pyfunction!(compose_layers_gravity_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(apply_linear, sub)?)?;
    sub.add_function(wrap_pyfunction!(apply_linear_f64, sub)?)?;
    // Exact ops exposure
    #[pyfunction]
    fn layer_norm_forward_exact_f32_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        gamma: PyReadonlyArray1<f32>,
        beta: PyReadonlyArray1<f32>,
        eps: f32,
    ) -> (&'py PyArray2<f32>, &'py PyArray1<f32>, &'py PyArray1<f32>) {
        let x = x.as_array().to_owned();
        let gamma = gamma.as_array().to_owned();
        let beta = beta.as_array().to_owned();
        let (y, mu, rstd) = crate::ops::layer_norm_forward_exact_f32(&x, &gamma, &beta, eps);
        (
            PyArray2::from_owned_array(py, y),
            PyArray1::from_owned_array(py, mu),
            PyArray1::from_owned_array(py, rstd),
        )
    }
    #[pyfunction]
    fn gelu_new_f32_py<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>) -> &'py PyArray2<f32> {
        let x = x.as_array().to_owned();
        let y = crate::ops::gelu_new_f32(&x);
        PyArray2::from_owned_array(py, y)
    }
    #[pyfunction]
    fn softmax_lastdim_f32_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        let x = x.as_array().to_owned();
        let y = crate::ops::softmax_lastdim_f32(&x);
        PyArray2::from_owned_array(py, y)
    }

    sub.add_function(wrap_pyfunction!(layer_norm_forward_exact_f32_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(gelu_new_f32_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(softmax_lastdim_f32_py, sub)?)?;
    // f64 exact ops
    #[pyfunction]
    fn layer_norm_forward_exact_f64_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        gamma: PyReadonlyArray1<f64>,
        beta: PyReadonlyArray1<f64>,
        eps: f64,
    ) -> (&'py PyArray2<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>) {
        let x = x.as_array().to_owned();
        let gamma = gamma.as_array().to_owned();
        let beta = beta.as_array().to_owned();
        let (y, mu, rstd) = crate::ops::layer_norm_forward_exact_f64(&x, &gamma, &beta, eps);
        (
            PyArray2::from_owned_array(py, y),
            PyArray1::from_owned_array(py, mu),
            PyArray1::from_owned_array(py, rstd),
        )
    }
    #[pyfunction]
    fn gelu_new_f64_py<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
        let x = x.as_array().to_owned();
        let y = crate::ops::gelu_new_f64(&x);
        PyArray2::from_owned_array(py, y)
    }
    #[pyfunction]
    fn softmax_lastdim_f64_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let x = x.as_array().to_owned();
        let y = crate::ops::softmax_lastdim_f64(&x);
        PyArray2::from_owned_array(py, y)
    }
    sub.add_function(wrap_pyfunction!(layer_norm_forward_exact_f64_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(gelu_new_f64_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(softmax_lastdim_f64_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(effective_metric_from_transform_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(metric_factor_cholesky_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(compose_layers_gravity_compact_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(rotate_metric_factor_block, sub)?)?;
    // Implicit transforms
    sub.add_function(wrap_pyfunction!(householder_chain_apply_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        householder_chain_apply_transpose_from_key,
        sub
    )?)?;
    sub.add_function(wrap_pyfunction!(givens_chain_apply_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(lowrank_plus_diag_apply_from_key, sub)?)?;
    // Classes
    sub.add_class::<CollapsedTransformF32>()?;
    sub.add_class::<CollapsedTransformF64>()?;
    sub.add_class::<CollapsedRunnerF32>()?;
    sub.add_class::<CollapsedRunnerF64>()?;

    m.add_submodule(sub)?;
    Ok(())
}

// High-speed inference runner (CPU, f32): holds T_total, embedding matrix, and lm_head
#[pyclass]
pub struct CollapsedRunnerF32 {
    t: Array2<f32>,                     // (d, d)
    embed: Array2<f32>,                 // (vocab, d)
    lm_w: Array2<f32>,                  // (vocab, d)
    lm_b: Option<ndarray::Array1<f32>>, // (vocab)
}

// High-precision inference runner (CPU, f64)
#[pyclass]
pub struct CollapsedRunnerF64 {
    t: ndarray::Array2<f64>,            // (d, d)
    embed: ndarray::Array2<f64>,        // (vocab, d)
    lm_w: ndarray::Array2<f64>,         // (vocab, d)
    lm_b: Option<ndarray::Array1<f64>>, // (vocab)
}

#[pymethods]
impl CollapsedRunnerF64 {
    #[new]
    fn new(
        t: PyReadonlyArray2<f64>,
        embed: PyReadonlyArray2<f64>,
        lm_w: PyReadonlyArray2<f64>,
        lm_b: Option<PyReadonlyArray1<f64>>,
    ) -> Self {
        let t_arr = t.as_array().to_owned();
        let embed_arr = embed.as_array().to_owned();
        let lm_w_arr = lm_w.as_array().to_owned();
        let lm_b_arr = lm_b.map(|b| b.as_array().to_owned());
        Self {
            t: t_arr,
            embed: embed_arr,
            lm_w: lm_w_arr,
            lm_b: lm_b_arr,
        }
    }

    /// step: ids (batch,) -> logits (batch, vocab)
    fn step<'py>(&self, py: Python<'py>, ids: PyReadonlyArray1<'py, i64>) -> &'py PyArray2<f64> {
        use ndarray::Array2 as A2;
        let ids_arr = ids.as_array();
        let batch = ids_arr.len();
        let d = self.t.dim().1;
        // gather
        let mut x = A2::<f64>::zeros((batch, d));
        for (i, &tok) in ids_arr.iter().enumerate() {
            let idx = if tok < 0 { 0usize } else { tok as usize };
            let row = self.embed.row(idx);
            x.row_mut(i).assign(&row);
        }
        // x * T^T
        let xt = x.dot(&self.t.t());
        // logits = xt · W^T
        let mut logits = xt.dot(&self.lm_w.t());
        if let Some(bias) = &self.lm_b {
            for mut row in logits.rows_mut() {
                row += &bias.view();
            }
        }
        PyArray2::from_owned_array(py, logits)
    }
}

#[pymethods]
impl CollapsedRunnerF32 {
    #[new]
    fn new(
        t: PyReadonlyArray2<f32>,
        embed: PyReadonlyArray2<f32>,
        lm_w: PyReadonlyArray2<f32>,
        lm_b: Option<PyReadonlyArray1<f32>>,
    ) -> Self {
        let t_arr = t.as_array().to_owned();
        let embed_arr = embed.as_array().to_owned();
        let lm_w_arr = lm_w.as_array().to_owned();
        let lm_b_arr = lm_b.map(|b| b.as_array().to_owned());
        Self {
            t: t_arr,
            embed: embed_arr,
            lm_w: lm_w_arr,
            lm_b: lm_b_arr,
        }
    }

    /// step: ids (batch,) -> logits (batch, vocab)
    fn step<'py>(&self, py: Python<'py>, ids: PyReadonlyArray1<'py, i64>) -> &'py PyArray2<f32> {
        use ndarray::Array2;
        let ids_arr = ids.as_array();
        let batch = ids_arr.len();
        let d = self.t.dim().1;

        // Gather embeddings
        let mut x = Array2::<f32>::zeros((batch, d));
        for (i, &tok) in ids_arr.iter().enumerate() {
            let idx = if tok < 0 { 0usize } else { tok as usize };
            let row = self.embed.row(idx);
            x.row_mut(i).assign(&row);
        }
        // Apply T_total: x' = x * T^T  => (batch,d) = (batch,d) · (d,d)
        let xt = x.dot(&self.t.t());
        // Logits = x' · W^T  with W: (vocab,d) → W^T: (d,vocab)
        let mut logits = xt.dot(&self.lm_w.t());
        if let Some(bias) = &self.lm_b {
            // add bias per vocab
            for mut row in logits.rows_mut() {
                row += &bias.view();
            }
        }
        PyArray2::from_owned_array(py, logits)
    }
}
```
---
## File: `reality_stone/src/bindings/mobius.rs`

```rust
use crate::ops::mobius;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn mobius_add_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = mobius::mobius_add(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn mobius_add_cuda(
    _py: Python,
    u_ptr: usize,
    v_ptr: usize,
    out_ptr: usize,
    batch_size: i64,
    dim: i64,
    c: f32,
) -> PyResult<()> {
    let u_ptr_f32 = u_ptr as *const f32;
    let v_ptr_f32 = v_ptr as *const f32;
    let out_ptr_f32 = out_ptr as *mut f32;
    mobius::cuda::mobius_add_cuda(out_ptr_f32, u_ptr_f32, v_ptr_f32, c, batch_size, dim);
    Ok(())
}

#[pyfunction]
pub fn mobius_scalar_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    r: f32,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let result = mobius::mobius_scalar(&u_arr, c, r);
    result.into_pyarray(py)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn mobius_scalar_cuda(
    _py: Python,
    u_ptr: usize,
    out_ptr: usize,
    batch_size: i64,
    dim: i64,
    r: f32,
    c: f32,
) -> PyResult<()> {
    let u_ptr_f32 = u_ptr as *const f32;
    let out_ptr_f32 = out_ptr as *mut f32;
    mobius::cuda::mobius_scalar_cuda(out_ptr_f32, u_ptr_f32, c, r, batch_size, dim);
    Ok(())
}

// 동적 곡률 Mobius 덧셈
#[pyfunction]
pub fn mobius_add_dynamic_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let (result, c) = mobius::mobius_add_dynamic(&u_arr, &v_arr, &dynamic_c);
    (result.into_pyarray(py), c)
}

// 동적 곡률 Mobius 덧셈의 backward pass
#[pyfunction]
pub fn mobius_add_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let (grad_u, grad_v, grad_kappa) =
        mobius::mobius_add_dynamic_backward(&grad_output_arr, &u_arr, &v_arr, &dynamic_c);
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py), grad_kappa)
}

#[pyfunction]
pub fn mobius_add_layerwise_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappas: Vec<f32>,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let layer_curvatures = crate::ops::LayerWiseDynamicCurvature::from_kappas(kappas, c_min, c_max);
    let (result, c) = mobius::mobius_add_layerwise(&u_arr, &v_arr, &layer_curvatures, layer_idx);
    (result.into_pyarray(py), c)
}

#[pyfunction]
pub fn mobius_add_layerwise_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappas: Vec<f32>,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let layer_curvatures = crate::ops::LayerWiseDynamicCurvature::from_kappas(kappas, c_min, c_max);
    let (grad_u, grad_v, grad_kappa) = mobius::mobius_add_layerwise_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &layer_curvatures,
        layer_idx,
    );
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py), grad_kappa)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mobius_add_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_scalar_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_dynamic_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_layerwise_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_layerwise_backward_cpu, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(mobius_add_cuda, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(mobius_scalar_cuda, m)?)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/mod.rs`

```rust
pub mod hyper_metric;
pub mod memory;
pub mod rsulf;
pub mod spline;
pub mod spline_cache;

pub mod bellman;
pub mod diffusion;
pub mod extraction;
pub mod geodesic_attention;
pub mod klein;
pub mod lorentz;
pub mod macros;
pub mod metrikey;
pub mod mobius;
pub mod poincare;
pub mod riemann;
pub mod suppression;
pub mod unified_riemannian;

use pyo3::prelude::*;

#[pymodule]
pub fn _rust(py: Python, m: &PyModule) -> PyResult<()> {
    mobius::register(m)?;
    poincare::register(m)?;
    lorentz::register(m)?;
    klein::register(m)?;
    riemann::register(m)?;
    metrikey::init_module(py, m)?;
    bellman::register(m)?;
    extraction::register(m)?;
    suppression::register(m)?;
    rsulf::register(m)?;
    spline::register_spline_module(py, m)?;
    m.add_class::<spline_cache::PySplineCache>()?;
    m.add_class::<memory::PyGeodesicMemory>()?;
    diffusion::register(m)?;
    hyper_metric::register(m)?;
    geodesic_attention::register(m)?;
    unified_riemannian::register(m)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/poincare.rs`

```rust
use crate::{create_binding, layers::poincare, ops::project};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// --- 매크로를 사용한 바인딩 생성 ---

create_binding!(
    poincare_distance_cpu,
    poincare::poincare_distance,
    [u, v, c, eps],
    PyArray1
);
create_binding!(
    poincare_to_lorentz_cpu,
    poincare::poincare_to_lorentz,
    [x, c],
    PyArray2
);
create_binding!(
    poincare_to_klein_cpu,
    poincare::poincare_to_klein,
    [x, c],
    PyArray2
);

// --- 매크로로 처리하기 복잡한 함수들 ---

#[pyfunction]
pub fn poincare_ball_layer_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    poincare::poincare_ball_layer(&u_arr, &v_arr, c, t).into_pyarray(py)
}

/// Exponential map on the Poincaré ball at point x with tangent vector v.
#[pyfunction]
pub fn poincare_exp_at_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    eps: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let v_arr = v.as_array();
    poincare::poincare_exp_at(&x_arr, &v_arr, c, eps).into_pyarray(py)
}

/// Logarithmic map on the Poincaré ball at point x for point y.
#[pyfunction]
pub fn poincare_log_at_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    c: f32,
    eps: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    poincare::poincare_log_at(&x_arr, &y_arr, c, eps).into_pyarray(py)
}

#[pyfunction]
pub fn poincare_ball_layer_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let (grad_u, grad_v) = poincare::poincare_ball_layer_backward(
        &grad_output.as_array(),
        &u.as_array(),
        &v.as_array(),
        c,
        t,
    );
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py))
}

// --- Dynamic / Layerwise bindings ---

#[pyfunction]
pub fn poincare_ball_layer_dynamic_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (&'py PyArray2<f32>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let (out, c) = poincare::poincare_ball_layer_dynamic(&u_arr, &v_arr, &dynamic_c, t);
    (out.into_pyarray(py), c)
}

#[pyfunction]
pub fn poincare_ball_layer_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let (gu, gv, gk) = poincare::poincare_ball_layer_dynamic_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &dynamic_c,
        t,
    );
    (gu.into_pyarray(py), gv.into_pyarray(py), gk)
}

#[pyfunction]
pub fn poincare_ball_layer_layerwise_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (&'py PyArray2<f32>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let lw = crate::ops::LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (out, c) = poincare::poincare_ball_layer_layerwise(&u_arr, &v_arr, &lw, layer_idx, t);
    (out.into_pyarray(py), c)
}

#[pyfunction]
pub fn poincare_ball_layer_layerwise_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let lw = crate::ops::LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (gu, gv, gk) = poincare::poincare_ball_layer_layerwise_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &lw,
        layer_idx,
        t,
    );
    (gu.into_pyarray(py), gv.into_pyarray(py), gk)
}

#[pyfunction]
pub fn mobius_add_vjp_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    c: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let (grad_x, grad_y) = crate::ops::mobius::mobius_add_vjp(
        &grad_output.as_array(),
        &x.as_array(),
        &y.as_array(),
        c,
    );
    (grad_x.into_pyarray(py), grad_y.into_pyarray(py))
}

#[pyfunction]
pub fn mobius_scalar_vjp_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    x: PyReadonlyArray2<f32>,
    c: f32,
    r: f32,
) -> &'py PyArray2<f32> {
    crate::ops::mobius::mobius_scalar_vjp(&grad_output.as_array(), &x.as_array(), c, r)
        .into_pyarray(py)
}

#[pyfunction]
pub fn project_to_ball_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    epsilon: f32,
) -> &'py PyArray2<f32> {
    project::project_to_ball(&x.as_array(), epsilon).into_pyarray(py)
}

#[pyfunction]
pub fn poincare_riemannian_adam_step_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    grad: PyReadonlyArray2<'py, f32>,
    m: PyReadonlyArray2<'py, f32>,
    v: PyReadonlyArray2<'py, f32>,
    step: u64,
    c: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    max_norm_eps: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, &'py PyArray2<f32>) {
    let x_arr = x.as_array();
    let grad_arr = grad.as_array();
    let mut m_arr = m.as_array().to_owned();
    let mut v_arr = v.as_array().to_owned();
    let x_new = poincare::poincare_riemannian_adam_step(
        &x_arr,
        &grad_arr,
        &mut m_arr,
        &mut v_arr,
        step,
        c,
        lr,
        beta1,
        beta2,
        eps,
        max_norm_eps,
    );
    (
        x_new.into_pyarray(py),
        m_arr.into_pyarray(py),
        v_arr.into_pyarray(py),
    )
}

// --- CUDA bindings ---

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn poincare_distance_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    boundary_eps: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    poincare::cuda::poincare_distance_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        boundary_eps,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn poincare_ball_layer_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    poincare::cuda::poincare_ball_layer_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn poincare_ball_layer_backward_cuda(
    grad_output: usize,
    u: usize,
    v: usize,
    grad_u: usize,
    grad_v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    poincare::cuda::poincare_ball_layer_backward_cuda(
        grad_output as *const f32,
        u as *const f32,
        v as *const f32,
        grad_u as *mut f32,
        grad_v as *mut f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

// --- 모듈 등록 ---

pub fn register(m: &PyModule) -> PyResult<()> {
    let sub = PyModule::new(m.py(), "poincare")?;
    sub.add_function(wrap_pyfunction!(poincare_distance_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_to_lorentz_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_to_klein_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_ball_layer_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_exp_at_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_log_at_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(mobius_add_vjp_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(mobius_scalar_vjp_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(project_to_ball_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_riemannian_adam_step_cpu, sub)?)?;

    // Dynamic / Layerwise
    sub.add_function(wrap_pyfunction!(poincare_ball_layer_dynamic_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        poincare_ball_layer_dynamic_backward_cpu,
        sub
    )?)?;
    sub.add_function(wrap_pyfunction!(poincare_ball_layer_layerwise_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        poincare_ball_layer_layerwise_backward_cpu,
        sub
    )?)?;

    #[cfg(feature = "cuda")]
    {
        sub.add_function(wrap_pyfunction!(poincare_distance_cuda, sub)?)?;
        sub.add_function(wrap_pyfunction!(poincare_ball_layer_cuda, sub)?)?;
        sub.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cuda, sub)?)?;
        m.add_function(wrap_pyfunction!(poincare_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(poincare_ball_layer_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cuda, m)?)?;
    }

    m.add_submodule(sub)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/riemann.rs`

```rust
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn riemann_lowrank_forward_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    p: PyReadonlyArray2<f32>,
    sigma: PyReadonlyArray2<f32>,
    q: PyReadonlyArray2<f32>,
    b_tan: PyReadonlyArray1<f32>,
    c: f32,
    epsilon: f32,
) -> &'py PyArray2<f32> {
    let out = crate::layers::riemann::riemann_lowrank_forward(
        &x.as_array(),
        &p.as_array(),
        &sigma.as_array(),
        &q.as_array(),
        &b_tan.as_array(),
        c,
        epsilon,
    );
    out.into_pyarray(py)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(riemann_lowrank_forward_cpu, m)?)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/rsulf.rs`

```rust
use crate::layers::decoder::RiemannianDecoder;
use crate::layers::human_decoder::{HumanStyleDecoder, StageWeights};
use crate::layers::rsulf::{
    adaptive_rank_svd, analyze_layer, block_lanczos_svd, create_causal_laplacian,
    fold_dimension_svd, fold_ffn_svd, nystrom_approximation, verify_fold_consistency,
    RSULFComponents, RSULFConfig, RSULFLayer,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(feature = "cuda")]
mod rsulf_cuda_ffi {
    use std::ffi::c_void;

    extern "C" {
        pub fn rsulf_forward_cuda(
            x: *const f32,
            v1: *const f32,
            s1: *const f32,
            u1: *const f32,
            v2: *const f32,
            s2: *const f32,
            u2: *const f32,
            g_inv: *const f32,
            v_mem: *const f32,
            eta: f32,
            alpha: f32,
            gamma_param: f32,
            batch: i32,
            d: i32,
            r: i32,
            ffn_dim: i32,
            x_out: *mut f32,
            v_out: *mut f32,
        );

        pub fn rsulf_batch_forward_cuda(
            x: *const f32,
            v1: *const f32,
            s1: *const f32,
            u1: *const f32,
            v2: *const f32,
            s2: *const f32,
            u2: *const f32,
            g_inv: *const f32,
            v_mem: *const f32,
            eta: f32,
            alpha: f32,
            gamma_param: f32,
            batch: i32,
            seq_len: i32,
            d: i32,
            r: i32,
            ffn_dim: i32,
            x_out: *mut f32,
            v_out: *mut f32,
        );

        pub fn rsulf_unified_forward_cuda(
            x: *const f32,
            v1: *const f32,
            s1: *const f32,
            u1: *const f32,
            v2: *const f32,
            s2: *const f32,
            u2: *const f32,
            g_inv: *const f32,
            laplacian: *const f32,
            v_mem: *const f32,
            eta: f32,
            alpha: f32,
            beta: f32,
            gamma_param: f32,
            curvature: f32,
            batch: i32,
            seq_len: i32,
            d: i32,
            r: i32,
            ffn_dim: i32,
            window: i32,
            x_out: *mut f32,
            v_out: *mut f32,
        );

        pub fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
        pub fn cudaFree(ptr: *mut c_void) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
    }
}

#[pyclass]
pub struct PyRSULFLayer {
    inner: RSULFLayer,
}

#[pymethods]
impl PyRSULFLayer {
    #[new]
    #[pyo3(signature = (wq, wk, w1, w2, d_model=4096, r=1024, eta=0.01, alpha=0.02, beta=0.01, gamma=0.99, seq_len=128, window=8))]
    pub fn new(
        wq: PyReadonlyArray2<f32>,
        wk: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        w2: PyReadonlyArray2<f32>,
        d_model: usize,
        r: usize,
        eta: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        seq_len: usize,
        window: usize,
    ) -> Self {
        let config = RSULFConfig {
            d_model,
            r,
            eta,
            alpha,
            beta,
            gamma,
            seq_len,
            window,
            calibration_samples: 1024,
        };

        let inner = RSULFLayer::from_transformer(
            wq.as_array(),
            wk.as_array(),
            w1.as_array(),
            w2.as_array(),
            config,
        );

        Self { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (wq, wk, w1, w2, g_diag, d_model=4096, r=1024, eta=0.01, alpha=0.02, beta=0.01, gamma=0.99, seq_len=128, window=8))]
    pub fn new_with_metric(
        wq: PyReadonlyArray2<f32>,
        wk: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        w2: PyReadonlyArray2<f32>,
        g_diag: PyReadonlyArray1<f32>,
        d_model: usize,
        r: usize,
        eta: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        seq_len: usize,
        window: usize,
    ) -> Self {
        let config = RSULFConfig {
            d_model,
            r,
            eta,
            alpha,
            beta,
            gamma,
            seq_len,
            window,
            calibration_samples: 1024,
        };

        let inner = RSULFLayer::from_transformer_with_metric(
            wq.as_array(),
            wk.as_array(),
            w1.as_array(),
            w2.as_array(),
            config,
            g_diag.as_array(),
        );

        Self { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (wq, wk, w1, w2, u_basis, basis_rank, d_model=4096, r=1024, eta=0.01, alpha=0.02, beta=0.01, gamma=0.99, seq_len=128, window=8))]
    pub fn new_with_basis(
        wq: PyReadonlyArray2<f32>,
        wk: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        w2: PyReadonlyArray2<f32>,
        u_basis: PyReadonlyArray2<f32>,
        basis_rank: usize,
        d_model: usize,
        r: usize,
        eta: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        seq_len: usize,
        window: usize,
    ) -> Self {
        let config = RSULFConfig {
            d_model,
            r,
            eta,
            alpha,
            beta,
            gamma,
            seq_len,
            window,
            calibration_samples: 1024,
        };

        let global_basis = crate::layers::rsulf::GlobalBasis {
            u: u_basis.as_array().to_owned(),
            rank: basis_rank,
        };

        let inner = RSULFLayer::from_transformer_with_basis(
            wq.as_array(),
            wk.as_array(),
            w1.as_array(),
            w2.as_array(),
            config,
            &global_basis,
        );

        Self { inner }
    }

    pub fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        v_mem: Option<PyReadonlyArray1<f32>>,
    ) -> (&'py PyArray2<f32>, &'py PyArray1<f32>) {
        let v_view = v_mem.as_ref().map(|v| v.as_array());
        let (output, v_new) = self.inner.forward(x.as_array(), v_view);
        (output.into_pyarray(py), v_new.into_pyarray(py))
    }

    pub fn param_count(&self) -> (usize, usize, f32) {
        self.inner.param_count()
    }

    #[getter]
    pub fn curvature(&self) -> f32 {
        self.inner.curvature
    }

    #[getter]
    pub fn d_model(&self) -> usize {
        self.inner.config.d_model
    }

    #[getter]
    pub fn r(&self) -> usize {
        self.inner.config.r
    }

    #[getter]
    pub fn eta(&self) -> f32 {
        self.inner.config.eta
    }

    #[getter]
    pub fn alpha(&self) -> f32 {
        self.inner.config.alpha
    }

    #[getter]
    pub fn beta(&self) -> f32 {
        self.inner.config.beta
    }

    #[getter]
    pub fn gamma(&self) -> f32 {
        self.inner.config.gamma
    }

    #[getter]
    pub fn g_inv<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.inner.g_inv.clone().into_pyarray(py)
    }

    #[getter]
    pub fn g_diag<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.inner.g_diag.clone().into_pyarray(py)
    }

    pub fn export_components<'py>(&self, py: Python<'py>) -> &'py PyDict {
        let comp = self.inner.export_components();
        let dict = PyDict::new(py);
        dict.set_item("d_model", comp.d_model).unwrap();
        dict.set_item("r", comp.r).unwrap();
        dict.set_item("eta", comp.eta).unwrap();
        dict.set_item("alpha", comp.alpha).unwrap();
        dict.set_item("beta", comp.beta).unwrap();
        dict.set_item("gamma", comp.gamma).unwrap();
        dict.set_item("seq_len", comp.seq_len).unwrap();
        dict.set_item("window", comp.window).unwrap();
        dict.set_item("g_diag", comp.g_diag.into_pyarray(py))
            .unwrap();
        dict.set_item("g_inv", comp.g_inv.into_pyarray(py)).unwrap();
        dict.set_item("g_sym", comp.g_sym.into_pyarray(py)).unwrap();
        dict.set_item("a_antisym", comp.a_antisym.into_pyarray(py))
            .unwrap();
        dict.set_item("u_metric", comp.u_metric.into_pyarray(py))
            .unwrap();
        dict.set_item("v_metric", comp.v_metric.into_pyarray(py))
            .unwrap();
        dict.set_item("g_core", comp.g_core.into_pyarray(py))
            .unwrap();
        dict.set_item("a_core", comp.a_core.into_pyarray(py))
            .unwrap();
        dict.set_item("curvature", comp.curvature).unwrap();
        dict.set_item("ffn_u1", comp.ffn_u1.into_pyarray(py))
            .unwrap();
        dict.set_item("ffn_s1", comp.ffn_s1.into_pyarray(py))
            .unwrap();
        dict.set_item("ffn_v1", comp.ffn_v1.into_pyarray(py))
            .unwrap();
        dict.set_item("ffn_u2", comp.ffn_u2.into_pyarray(py))
            .unwrap();
        dict.set_item("ffn_s2", comp.ffn_s2.into_pyarray(py))
            .unwrap();
        dict.set_item("ffn_v2", comp.ffn_v2.into_pyarray(py))
            .unwrap();
        dict
    }

    #[staticmethod]
    pub fn from_components(
        d_model: usize,
        r: usize,
        eta: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        seq_len: usize,
        window: usize,
        g_diag: PyReadonlyArray1<f32>,
        g_inv: PyReadonlyArray1<f32>,
        g_sym: PyReadonlyArray2<f32>,
        a_antisym: PyReadonlyArray2<f32>,
        u_metric: PyReadonlyArray2<f32>,
        v_metric: PyReadonlyArray2<f32>,
        g_core: PyReadonlyArray2<f32>,
        a_core: PyReadonlyArray2<f32>,
        curvature: f32,
        ffn_u1: PyReadonlyArray2<f32>,
        ffn_s1: PyReadonlyArray1<f32>,
        ffn_v1: PyReadonlyArray2<f32>,
        ffn_u2: PyReadonlyArray2<f32>,
        ffn_s2: PyReadonlyArray1<f32>,
        ffn_v2: PyReadonlyArray2<f32>,
    ) -> Self {
        let comp = RSULFComponents {
            d_model,
            r,
            eta,
            alpha,
            beta,
            gamma,
            seq_len,
            window,
            g_diag: g_diag.as_array().to_owned(),
            g_inv: g_inv.as_array().to_owned(),
            g_sym: g_sym.as_array().to_owned(),
            a_antisym: a_antisym.as_array().to_owned(),
            u_metric: u_metric.as_array().to_owned(),
            v_metric: v_metric.as_array().to_owned(),
            g_core: g_core.as_array().to_owned(),
            a_core: a_core.as_array().to_owned(),
            curvature,
            ffn_u1: ffn_u1.as_array().to_owned(),
            ffn_s1: ffn_s1.as_array().to_owned(),
            ffn_v1: ffn_v1.as_array().to_owned(),
            ffn_u2: ffn_u2.as_array().to_owned(),
            ffn_s2: ffn_s2.as_array().to_owned(),
            ffn_v2: ffn_v2.as_array().to_owned(),
        };
        let inner = RSULFLayer::from_components(comp);
        Self { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (wq, wk, w1, w2, d_model=4096, r=1024, eta=0.01, alpha=0.02, beta=0.01, gamma=0.99, seq_len=128, window=8, calibration_samples=1024))]
    pub fn new_fast(
        wq: PyReadonlyArray2<f32>,
        wk: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        w2: PyReadonlyArray2<f32>,
        d_model: usize,
        r: usize,
        eta: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        seq_len: usize,
        window: usize,
        calibration_samples: usize,
    ) -> Self {
        let config = RSULFConfig {
            d_model,
            r,
            eta,
            alpha,
            beta,
            gamma,
            seq_len,
            window,
            calibration_samples,
        };

        let inner = RSULFLayer::from_transformer_fast(
            wq.as_array(),
            wk.as_array(),
            w1.as_array(),
            w2.as_array(),
            config,
        );

        Self { inner }
    }
}

#[pyclass]
pub struct PyRiemannianDecoder {
    inner: RiemannianDecoder,
}

#[pymethods]
impl PyRiemannianDecoder {
    #[new]
    pub fn new(
        u_basis: PyReadonlyArray2<f32>,
        a: PyReadonlyArray2<f32>,
        bt: PyReadonlyArray2<f32>,
        bias: PyReadonlyArray1<f32>,
    ) -> Self {
        let u = u_basis.as_array().to_owned();
        let a_mat = a.as_array().to_owned();
        let bt_mat = bt.as_array().to_owned();
        let b_vec = bias.as_array().to_owned();
        let inner = RiemannianDecoder::new(u, a_mat, bt_mat, b_vec);
        Self { inner }
    }

    #[staticmethod]
    pub fn from_lm_head(
        w_lm: PyReadonlyArray2<f32>,
        b_lm: PyReadonlyArray1<f32>,
        u_basis: PyReadonlyArray2<f32>,
        basis_rank: usize,
        target_rank: usize,
    ) -> Self {
        let global_basis = crate::layers::rsulf::GlobalBasis {
            u: u_basis.as_array().to_owned(),
            rank: basis_rank,
        };
        let inner = RiemannianDecoder::from_lm_head(
            w_lm.as_array(),
            b_lm.as_array(),
            &global_basis,
            target_rank,
        );
        Self { inner }
    }

    pub fn forward<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> &'py PyArray2<f32> {
        let logits = self.inner.forward(x.as_array());
        logits.into_pyarray(py)
    }

    #[getter]
    pub fn d_model(&self) -> usize {
        self.inner.d_model
    }

    #[getter]
    pub fn rank(&self) -> usize {
        self.inner.r
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab
    }
}

#[pyclass]
pub struct PyHumanDecoder {
    inner: HumanStyleDecoder,
}

#[pymethods]
impl PyHumanDecoder {
    #[new]
    #[pyo3(signature=(
        embeddings,
        skeleton_ids,
        relation_ids,
        object_ids,
        alpha_logit=1.0,
        alpha_cos=0.4,
        beta_logit=1.0,
        beta_cos=0.8,
        beta_geo=0.3,
        curvature=1e-3
    ))]
    pub fn new(
        embeddings: PyReadonlyArray2<f32>,
        skeleton_ids: Vec<usize>,
        relation_ids: Vec<usize>,
        object_ids: Vec<usize>,
        alpha_logit: f32,
        alpha_cos: f32,
        beta_logit: f32,
        beta_cos: f32,
        beta_geo: f32,
        curvature: f32,
    ) -> Self {
        let relation_weights = StageWeights {
            logit: alpha_logit,
            cosine: alpha_cos,
            geodesic: 0.0,
        };
        let object_weights = StageWeights {
            logit: beta_logit,
            cosine: beta_cos,
            geodesic: beta_geo,
        };
        let inner = HumanStyleDecoder::new(
            embeddings.as_array().to_owned(),
            skeleton_ids,
            relation_ids,
            object_ids,
            relation_weights,
            object_weights,
            curvature,
        );
        Self { inner }
    }

    pub fn decode(
        &self,
        logits: PyReadonlyArray2<f32>,
        relation_ctx: PyReadonlyArray2<f32>,
        object_ctx: PyReadonlyArray2<f32>,
        topk_relation: usize,
        topk_object: usize,
    ) -> Vec<usize> {
        self.inner.decode_batch(
            logits.as_array(),
            relation_ctx.as_array(),
            object_ctx.as_array(),
            topk_relation,
            topk_object,
        )
    }
}

#[pyfunction(signature = (
    x, v1, s1, u1, v2, s2, u2, g_inv,
    v_mem=None,
    eta=0.01, alpha=0.02, gamma_param=0.99
))]
pub fn rsulf_forward_cuda_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v1: PyReadonlyArray2<f32>,
    s1: PyReadonlyArray1<f32>,
    u1: PyReadonlyArray2<f32>,
    v2: PyReadonlyArray2<f32>,
    s2: PyReadonlyArray1<f32>,
    u2: PyReadonlyArray2<f32>,
    g_inv: PyReadonlyArray1<f32>,
    v_mem: Option<PyReadonlyArray1<f32>>,
    eta: f32,
    alpha: f32,
    gamma_param: f32,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            &py,
            &x,
            &v1,
            &s1,
            &u1,
            &v2,
            &s2,
            &u2,
            &g_inv,
            &v_mem,
            eta,
            alpha,
            gamma_param,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::bindings::rsulf::rsulf_cuda_ffi::*;
        use numpy::PyArray1;
        use pyo3::exceptions::PyRuntimeError;
        use pyo3::PyErr;
        use std::ffi::c_void;
        use std::ptr;
        use std::slice;

        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(PyRuntimeError::new_err("x must be 2D array"));
        }
        let batch = x_shape[0] as i32;
        let d = x_shape[1] as i32;
        let r = s1.shape()[0] as i32;
        let ffn_dim = v2.shape()[0] as i32;

        unsafe fn alloc_and_copy(src: &[f32]) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = src.len() * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            Ok(dst)
        }

        unsafe fn alloc_zeroed(len: usize) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = len * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            for i in 0..len {
                ptr::write(dst.add(i), 0.0);
            }
            Ok(dst)
        }

        let x_slice = x.as_slice()?;
        let v1_slice = v1.as_slice()?;
        let s1_slice = s1.as_slice()?;
        let u1_slice = u1.as_slice()?;
        let v2_slice = v2.as_slice()?;
        let s2_slice = s2.as_slice()?;
        let u2_slice = u2.as_slice()?;
        let g_inv_slice = g_inv.as_slice()?;
        let v_mem_slice = v_mem.as_ref().map(|v| v.as_slice().ok()).flatten();

        unsafe {
            let mut to_free: Vec<*mut c_void> = Vec::new();

            let x_dev = alloc_and_copy(x_slice)?;
            to_free.push(x_dev as *mut c_void);
            let v1_dev = alloc_and_copy(v1_slice)?;
            to_free.push(v1_dev as *mut c_void);
            let s1_dev = alloc_and_copy(s1_slice)?;
            to_free.push(s1_dev as *mut c_void);
            let u1_dev = alloc_and_copy(u1_slice)?;
            to_free.push(u1_dev as *mut c_void);
            let v2_dev = alloc_and_copy(v2_slice)?;
            to_free.push(v2_dev as *mut c_void);
            let s2_dev = alloc_and_copy(s2_slice)?;
            to_free.push(s2_dev as *mut c_void);
            let u2_dev = alloc_and_copy(u2_slice)?;
            to_free.push(u2_dev as *mut c_void);
            let g_inv_dev = alloc_and_copy(g_inv_slice)?;
            to_free.push(g_inv_dev as *mut c_void);

            let v_mem_dev = if let Some(slice_v) = v_mem_slice {
                let ptr_vm = alloc_and_copy(slice_v)?;
                to_free.push(ptr_vm as *mut c_void);
                ptr_vm
            } else {
                ptr::null_mut()
            };

            let total_x = (batch as usize) * (d as usize);
            let x_out_dev = alloc_zeroed(total_x)?;
            to_free.push(x_out_dev as *mut c_void);

            let v_out_dev = alloc_zeroed(batch as usize)?;
            to_free.push(v_out_dev as *mut c_void);

            rsulf_forward_cuda(
                x_dev,
                v1_dev,
                s1_dev,
                u1_dev,
                v2_dev,
                s2_dev,
                u2_dev,
                g_inv_dev,
                v_mem_dev,
                eta,
                alpha,
                gamma_param,
                batch,
                d,
                r,
                ffn_dim,
                x_out_dev,
                v_out_dev,
            );

            let sync_err = cudaDeviceSynchronize();
            if sync_err != 0 {
                for ptr_raw in to_free {
                    let _ = cudaFree(ptr_raw);
                }
                return Err(PyRuntimeError::new_err(format!(
                    "cudaDeviceSynchronize failed: {}",
                    sync_err
                )));
            }

            let x_host = slice::from_raw_parts(x_out_dev, total_x).to_vec();
            let v_host = slice::from_raw_parts(v_out_dev, batch as usize).to_vec();

            for ptr_raw in to_free {
                let _ = cudaFree(ptr_raw);
            }

            let x_arr = Array2::from_shape_vec((batch as usize, d as usize), x_host)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let x_out = x_arr.into_pyarray(py);
            let v_out = PyArray1::from_vec(py, v_host);

            Ok((x_out, v_out))
        }
    }
}

#[pyfunction(signature = (
    x, v1, s1, u1, v2, s2, u2, g_inv,
    v_mem=None,
    eta=0.01, alpha=0.02, gamma_param=0.99,
    batch=1, seq_len=1
))]
pub fn rsulf_batch_forward_cuda_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v1: PyReadonlyArray2<f32>,
    s1: PyReadonlyArray1<f32>,
    u1: PyReadonlyArray2<f32>,
    v2: PyReadonlyArray2<f32>,
    s2: PyReadonlyArray1<f32>,
    u2: PyReadonlyArray2<f32>,
    g_inv: PyReadonlyArray1<f32>,
    v_mem: Option<PyReadonlyArray1<f32>>,
    eta: f32,
    alpha: f32,
    gamma_param: f32,
    batch: i32,
    seq_len: i32,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            &py,
            &x,
            &v1,
            &s1,
            &u1,
            &v2,
            &s2,
            &u2,
            &g_inv,
            &v_mem,
            eta,
            alpha,
            gamma_param,
            batch,
            seq_len,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::bindings::rsulf::rsulf_cuda_ffi::*;
        use numpy::PyArray1;
        use pyo3::exceptions::PyRuntimeError;
        use pyo3::PyErr;
        use std::ffi::c_void;
        use std::ptr;
        use std::slice;

        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(PyRuntimeError::new_err("x must be 2D array"));
        }
        let d = x_shape[1] as i32;
        let r = s1.shape()[0] as i32;
        let ffn_dim = v2.shape()[0] as i32;

        unsafe fn alloc_and_copy(src: &[f32]) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = src.len() * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            Ok(dst)
        }

        unsafe fn alloc_zeroed(len: usize) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = len * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            for i in 0..len {
                ptr::write(dst.add(i), 0.0);
            }
            Ok(dst)
        }

        let x_slice = x.as_slice()?;
        let v1_slice = v1.as_slice()?;
        let s1_slice = s1.as_slice()?;
        let u1_slice = u1.as_slice()?;
        let v2_slice = v2.as_slice()?;
        let s2_slice = s2.as_slice()?;
        let u2_slice = u2.as_slice()?;
        let g_inv_slice = g_inv.as_slice()?;
        let v_mem_slice = v_mem.as_ref().map(|v| v.as_slice().ok()).flatten();

        unsafe {
            let mut to_free: Vec<*mut c_void> = Vec::new();

            let x_dev = alloc_and_copy(x_slice)?;
            to_free.push(x_dev as *mut c_void);
            let v1_dev = alloc_and_copy(v1_slice)?;
            to_free.push(v1_dev as *mut c_void);
            let s1_dev = alloc_and_copy(s1_slice)?;
            to_free.push(s1_dev as *mut c_void);
            let u1_dev = alloc_and_copy(u1_slice)?;
            to_free.push(u1_dev as *mut c_void);
            let v2_dev = alloc_and_copy(v2_slice)?;
            to_free.push(v2_dev as *mut c_void);
            let s2_dev = alloc_and_copy(s2_slice)?;
            to_free.push(s2_dev as *mut c_void);
            let u2_dev = alloc_and_copy(u2_slice)?;
            to_free.push(u2_dev as *mut c_void);
            let g_inv_dev = alloc_and_copy(g_inv_slice)?;
            to_free.push(g_inv_dev as *mut c_void);

            let v_mem_dev = if let Some(slice_v) = v_mem_slice {
                let ptr_vm = alloc_and_copy(slice_v)?;
                to_free.push(ptr_vm as *mut c_void);
                ptr_vm
            } else {
                ptr::null_mut()
            };

            let total_tokens = (batch as usize) * (seq_len as usize);
            let total_x = total_tokens * (d as usize);
            let x_out_dev = alloc_zeroed(total_x)?;
            to_free.push(x_out_dev as *mut c_void);

            let v_out_dev = alloc_zeroed(total_tokens)?;
            to_free.push(v_out_dev as *mut c_void);

            rsulf_batch_forward_cuda(
                x_dev,
                v1_dev,
                s1_dev,
                u1_dev,
                v2_dev,
                s2_dev,
                u2_dev,
                g_inv_dev,
                v_mem_dev,
                eta,
                alpha,
                gamma_param,
                batch,
                seq_len,
                d,
                r,
                ffn_dim,
                x_out_dev,
                v_out_dev,
            );

            let sync_err = cudaDeviceSynchronize();
            if sync_err != 0 {
                for ptr_raw in to_free {
                    let _ = cudaFree(ptr_raw);
                }
                return Err(PyRuntimeError::new_err(format!(
                    "cudaDeviceSynchronize failed: {}",
                    sync_err
                )));
            }

            let x_host = slice::from_raw_parts(x_out_dev, total_x).to_vec();
            let v_host = slice::from_raw_parts(v_out_dev, total_tokens).to_vec();

            for ptr_raw in to_free {
                let _ = cudaFree(ptr_raw);
            }

            let x_arr = Array2::from_shape_vec((total_tokens, d as usize), x_host)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let x_out = x_arr.into_pyarray(py);
            let v_out = PyArray1::from_vec(py, v_host);

            Ok((x_out, v_out))
        }
    }
}

#[pyfunction(signature = (
    x, v1, s1, u1, v2, s2, u2, g_inv, laplacian,
    v_mem=None,
    eta=0.01, alpha=0.02, beta=0.0, gamma_param=0.99, curvature=0.0,
    batch=1, seq_len=1, window=1
))]
pub fn rsulf_unified_forward_cuda_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v1: PyReadonlyArray2<f32>,
    s1: PyReadonlyArray1<f32>,
    u1: PyReadonlyArray2<f32>,
    v2: PyReadonlyArray2<f32>,
    s2: PyReadonlyArray1<f32>,
    u2: PyReadonlyArray2<f32>,
    g_inv: PyReadonlyArray1<f32>,
    laplacian: PyReadonlyArray2<f32>,
    v_mem: Option<PyReadonlyArray1<f32>>,
    eta: f32,
    alpha: f32,
    beta: f32,
    gamma_param: f32,
    curvature: f32,
    batch: i32,
    seq_len: i32,
    window: i32,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            &py,
            &x,
            &v1,
            &s1,
            &u1,
            &v2,
            &s2,
            &u2,
            &g_inv,
            &laplacian,
            &v_mem,
            eta,
            alpha,
            beta,
            gamma_param,
            curvature,
            batch,
            seq_len,
            window,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::bindings::rsulf::rsulf_cuda_ffi::*;
        use numpy::PyArray1;
        use pyo3::exceptions::PyRuntimeError;
        use pyo3::PyErr;
        use std::ffi::c_void;
        use std::ptr;
        use std::slice;

        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(PyRuntimeError::new_err("x must be 2D array"));
        }
        let d = x_shape[1] as i32;
        let r = s1.shape()[0] as i32;
        let ffn_dim = v2.shape()[0] as i32;

        unsafe fn alloc_and_copy(src: &[f32]) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = src.len() * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            Ok(dst)
        }

        unsafe fn alloc_zeroed(len: usize) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = len * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            for i in 0..len {
                ptr::write(dst.add(i), 0.0);
            }
            Ok(dst)
        }

        let x_slice = x.as_slice()?;
        let v1_slice = v1.as_slice()?;
        let s1_slice = s1.as_slice()?;
        let u1_slice = u1.as_slice()?;
        let v2_slice = v2.as_slice()?;
        let s2_slice = s2.as_slice()?;
        let u2_slice = u2.as_slice()?;
        let g_inv_slice = g_inv.as_slice()?;
        let lap_slice = laplacian.as_slice()?;
        let v_mem_slice = v_mem.as_ref().map(|v| v.as_slice().ok()).flatten();

        unsafe {
            let mut to_free: Vec<*mut c_void> = Vec::new();

            let x_dev = alloc_and_copy(x_slice)?;
            to_free.push(x_dev as *mut c_void);
            let v1_dev = alloc_and_copy(v1_slice)?;
            to_free.push(v1_dev as *mut c_void);
            let s1_dev = alloc_and_copy(s1_slice)?;
            to_free.push(s1_dev as *mut c_void);
            let u1_dev = alloc_and_copy(u1_slice)?;
            to_free.push(u1_dev as *mut c_void);
            let v2_dev = alloc_and_copy(v2_slice)?;
            to_free.push(v2_dev as *mut c_void);
            let s2_dev = alloc_and_copy(s2_slice)?;
            to_free.push(s2_dev as *mut c_void);
            let u2_dev = alloc_and_copy(u2_slice)?;
            to_free.push(u2_dev as *mut c_void);

            let d_usize = d as usize;
            let g_inv_host: Vec<f32> = if g_inv_slice.len() >= d_usize {
                g_inv_slice[..d_usize].to_vec()
            } else {
                let mut v = vec![1.0_f32; d_usize];
                for (i, val) in g_inv_slice.iter().enumerate() {
                    v[i] = *val;
                }
                v
            };
            let g_inv_dev = alloc_and_copy(&g_inv_host)?;
            to_free.push(g_inv_dev as *mut c_void);
            let lap_dev = alloc_and_copy(lap_slice)?;
            to_free.push(lap_dev as *mut c_void);

            let v_mem_dev = if let Some(slice_v) = v_mem_slice {
                let ptr_vm = alloc_and_copy(slice_v)?;
                to_free.push(ptr_vm as *mut c_void);
                ptr_vm
            } else {
                ptr::null_mut()
            };

            let total_tokens = (batch as usize) * (seq_len as usize);
            let total_x = total_tokens * (d as usize);
            let x_out_dev = alloc_zeroed(total_x)?;
            to_free.push(x_out_dev as *mut c_void);

            let v_out_dev = alloc_zeroed(total_tokens)?;
            to_free.push(v_out_dev as *mut c_void);

            rsulf_unified_forward_cuda(
                x_dev,
                v1_dev,
                s1_dev,
                u1_dev,
                v2_dev,
                s2_dev,
                u2_dev,
                g_inv_dev,
                lap_dev,
                v_mem_dev,
                eta,
                alpha,
                beta,
                gamma_param,
                curvature,
                batch,
                seq_len,
                d,
                r,
                ffn_dim,
                window,
                x_out_dev,
                v_out_dev,
            );

            let sync_err = cudaDeviceSynchronize();
            if sync_err != 0 {
                for ptr_raw in to_free {
                    let _ = cudaFree(ptr_raw);
                }
                return Err(PyRuntimeError::new_err(format!(
                    "cudaDeviceSynchronize failed: {}",
                    sync_err
                )));
            }

            let total_tokens_usize = total_tokens;
            let x_host = slice::from_raw_parts(x_out_dev, total_x).to_vec();
            let v_host = slice::from_raw_parts(v_out_dev, total_tokens_usize).to_vec();

            for ptr_raw in to_free {
                let _ = cudaFree(ptr_raw);
            }

            let x_arr = Array2::from_shape_vec((total_tokens_usize, d as usize), x_host)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let x_out = x_arr.into_pyarray(py);
            let v_out = PyArray1::from_vec(py, v_host);

            Ok((x_out, v_out))
        }
    }
}

#[pyfunction]
pub fn fold_metric_svd<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    target_dim: usize,
) -> (
    &'py PyArray2<f32>,
    &'py PyArray1<f32>,
    &'py PyArray2<f32>,
    f32,
) {
    let folded = fold_dimension_svd(wq.as_array(), wk.as_array(), target_dim);
    let curvature = crate::layers::rsulf::compute_curvature(&folded.s_residual);
    (
        folded.u.into_pyarray(py),
        folded.s.into_pyarray(py),
        folded.v.into_pyarray(py),
        curvature,
    )
}

#[pyfunction]
pub fn build_causal_laplacian<'py>(
    py: Python<'py>,
    seq_len: usize,
    window: usize,
) -> &'py PyArray2<f32> {
    let l = create_causal_laplacian(seq_len, window);
    l.into_pyarray(py)
}

#[pyfunction]
pub fn fold_ffn<'py>(
    py: Python<'py>,
    w1: PyReadonlyArray2<f32>,
    w2: PyReadonlyArray2<f32>,
    target_dim: usize,
) -> (
    &'py PyArray2<f32>,
    &'py PyArray1<f32>,
    &'py PyArray2<f32>,
    &'py PyArray2<f32>,
    &'py PyArray1<f32>,
    &'py PyArray2<f32>,
) {
    let folded = fold_ffn_svd(w1.as_array(), w2.as_array(), target_dim);
    (
        folded.u1.into_pyarray(py),
        folded.s1.into_pyarray(py),
        folded.v1.into_pyarray(py),
        folded.u2.into_pyarray(py),
        folded.s2.into_pyarray(py),
        folded.v2.into_pyarray(py),
    )
}

#[pyfunction]
pub fn verify_metric_consistency<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    target_dim: usize,
) -> &'py PyDict {
    let folded = fold_dimension_svd(wq.as_array(), wk.as_array(), target_dim);
    let result = verify_fold_consistency(wq.as_array(), wk.as_array(), &folded);

    let dict = PyDict::new(py);
    dict.set_item("symmetry_error", result.symmetry_error)
        .unwrap();
    dict.set_item("reconstruction_error", result.reconstruction_error)
        .unwrap();
    dict.set_item("fold_accuracy", result.fold_accuracy)
        .unwrap();
    dict.set_item("min_eigenvalue", result.min_eigenvalue)
        .unwrap();
    dict.set_item("condition_number", result.condition_number)
        .unwrap();
    dict.set_item("is_valid", result.is_valid).unwrap();
    dict
}

#[pyfunction]
pub fn fold_metric_optimized<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    target_dim: usize,
    method: &str,
) -> (
    &'py PyArray2<f32>,
    &'py PyArray1<f32>,
    &'py PyArray2<f32>,
    f32,
    &'py PyDict,
) {
    let d_q = wq.as_array().nrows();
    let d_k = wk.as_array().nrows();
    let d_in = wq.as_array().ncols();

    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = ndarray::Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded
                .slice_mut(ndarray::s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk.as_array());
        }
        expanded
    } else {
        wk.as_array().to_owned()
    };

    let g = wq.as_array().t().dot(&wk_expanded);

    let (u, s, v) = match method {
        "block_lanczos" => block_lanczos_svd(&g, target_dim, 32, 10),
        "adaptive" => {
            let (u, s, v, _) = adaptive_rank_svd(&g, 0.95, target_dim);
            (u, s, v)
        }
        _ => crate::layers::rsulf::randomized_svd(&g, target_dim, 5, 2),
    };

    let frob_g: f32 = g.iter().map(|x| x * x).sum();
    let frob_approx: f32 = s.iter().map(|x| x * x).sum();
    let tail = frob_g - frob_approx;
    let curvature = if tail > 0.0 { tail.sqrt() } else { 0.0 };

    let folded = crate::layers::rsulf::FoldedMetric {
        u: u.clone(),
        s: s.clone(),
        v: v.clone(),
        s_residual: ndarray::Array1::from_elem(1, curvature),
    };
    let consistency = verify_fold_consistency(wq.as_array(), wk.as_array(), &folded);

    let info = PyDict::new(py);
    info.set_item("symmetry_error", consistency.symmetry_error)
        .unwrap();
    info.set_item("reconstruction_error", consistency.reconstruction_error)
        .unwrap();
    info.set_item("fold_accuracy", consistency.fold_accuracy)
        .unwrap();
    info.set_item("min_eigenvalue", consistency.min_eigenvalue)
        .unwrap();
    info.set_item("condition_number", consistency.condition_number)
        .unwrap();
    info.set_item("is_valid", consistency.is_valid).unwrap();
    info.set_item("method", method).unwrap();

    (
        u.into_pyarray(py),
        s.into_pyarray(py),
        v.into_pyarray(py),
        curvature,
        info,
    )
}

#[pyfunction]
pub fn nystrom_metric<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    target_dim: usize,
    n_samples: usize,
) -> (&'py PyArray2<f32>, &'py PyArray1<f32>) {
    let d_q = wq.as_array().nrows();
    let d_k = wk.as_array().nrows();
    let d_in = wq.as_array().ncols();

    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = ndarray::Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded
                .slice_mut(ndarray::s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk.as_array());
        }
        expanded
    } else {
        wk.as_array().to_owned()
    };

    let g = wq.as_array().t().dot(&wk_expanded);
    let (u, s) = nystrom_approximation(&g, target_dim, n_samples);

    (u.into_pyarray(py), s.into_pyarray(py))
}

#[pyfunction(name = "analyze_layer")]
pub fn analyze_layer_py<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    w1: PyReadonlyArray2<f32>,
    w2: PyReadonlyArray2<f32>,
    layer_idx: usize,
    target_rank: usize,
) -> &'py PyDict {
    let analysis = analyze_layer(
        wq.as_array(),
        wk.as_array(),
        w1.as_array(),
        w2.as_array(),
        layer_idx,
        target_rank,
    );
    let dict = PyDict::new(py);
    dict.set_item("layer_idx", analysis.layer_idx).unwrap();
    dict.set_item("param_count", analysis.param_count).unwrap();
    dict.set_item("spectral_decay", analysis.spectral_decay)
        .unwrap();
    dict.set_item("condition_number", analysis.condition_number)
        .unwrap();
    dict.set_item("recommended_rank", analysis.recommended_rank)
        .unwrap();
    dict.set_item("expected_accuracy", analysis.expected_accuracy)
        .unwrap();
    dict
}

#[pyfunction(name = "extract_global_basis")]
pub fn extract_global_basis_py<'py>(
    py: Python<'py>,
    layers_wq: Vec<PyReadonlyArray2<f32>>,
    layers_wk: Vec<PyReadonlyArray2<f32>>,
    target_rank: usize,
) -> &'py PyDict {
    let wq_views: Vec<_> = layers_wq.iter().map(|x| x.as_array()).collect();
    let wk_views: Vec<_> = layers_wk.iter().map(|x| x.as_array()).collect();

    let basis = crate::layers::rsulf::extract_global_basis(&wq_views, &wk_views, target_rank);

    let dict = PyDict::new(py);
    dict.set_item("u", basis.u.into_pyarray(py)).unwrap();
    dict.set_item("rank", basis.rank).unwrap();
    dict
}

#[pyfunction(name = "create_compression_plan")]
pub fn create_compression_plan_py<'py>(
    py: Python<'py>,
    analyses: Vec<&PyDict>,
    compression_ratio: f32,
) -> &'py PyDict {
    let mut layer_analyses = Vec::new();

    for d in analyses {
        let layer_idx = d
            .get_item("layer_idx")
            .unwrap()
            .expect("layer_idx missing")
            .extract::<usize>()
            .unwrap_or(0);
        let param_count = d
            .get_item("param_count")
            .unwrap()
            .expect("param_count missing")
            .extract::<usize>()
            .unwrap_or(0);
        let spectral_decay = d
            .get_item("spectral_decay")
            .unwrap()
            .expect("spectral_decay missing")
            .extract::<f32>()
            .unwrap_or(0.0);
        let condition_number = d
            .get_item("condition_number")
            .unwrap()
            .expect("condition_number missing")
            .extract::<f32>()
            .unwrap_or(0.0);
        let recommended_rank = d
            .get_item("recommended_rank")
            .unwrap()
            .expect("recommended_rank missing")
            .extract::<usize>()
            .unwrap_or(1);
        let expected_accuracy = d
            .get_item("expected_accuracy")
            .unwrap()
            .expect("expected_accuracy missing")
            .extract::<f32>()
            .unwrap_or(0.0);

        use crate::layers::rsulf::{CompressionStrategy, LayerAnalysis, LayerType};

        let strategy = CompressionStrategy::MetricSVD {
            target_rank: recommended_rank,
            expected_accuracy,
        };

        layer_analyses.push(LayerAnalysis {
            layer_idx,
            layer_type: LayerType::Attention,
            input_shape: (0, 0),
            output_shape: (0, 0),
            param_count,
            spectral_decay,
            condition_number,
            recommended_rank,
            expected_accuracy,
            strategy,
        });
    }

    let plan = crate::layers::rsulf::create_compression_plan(layer_analyses, compression_ratio);

    let dict = PyDict::new(py);
    dict.set_item("total_original_params", plan.total_original_params)
        .unwrap();
    dict.set_item("total_compressed_params", plan.total_compressed_params)
        .unwrap();
    dict.set_item(
        "expected_compression_ratio",
        plan.expected_compression_ratio,
    )
    .unwrap();
    dict.set_item("min_expected_accuracy", plan.min_expected_accuracy)
        .unwrap();

    dict
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRSULFLayer>()?;
    m.add_function(wrap_pyfunction!(fold_metric_svd, m)?)?;
    m.add_function(wrap_pyfunction!(fold_ffn, m)?)?;
    m.add_function(wrap_pyfunction!(build_causal_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(verify_metric_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(fold_metric_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(nystrom_metric, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_layer_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_global_basis_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_compression_plan_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(rsulf_forward_cuda_py, m)?)?;
        m.add_function(wrap_pyfunction!(rsulf_batch_forward_cuda_py, m)?)?;
        m.add_function(wrap_pyfunction!(rsulf_unified_forward_cuda_py, m)?)?;
    }
    m.add_class::<PyRiemannianDecoder>()?;
    m.add_class::<PyHumanDecoder>()?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/spline.rs`

```rust
use crate::layers::spline::SplineLayer;
use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use crate::layers::spline::cuda as spline_cuda;

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn spline_interpolate_cuda(
    control_points: usize,
    weights: usize,
    k: i32,
    in_features: i32,
    out_features: i32,
) -> PyResult<()> {
    spline_cuda::spline_interpolate_cuda(
        control_points as *const f32,
        weights as *mut f32,
        k,
        in_features,
        out_features,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn spline_forward_cuda(
    input: usize,
    control_points: usize,
    output: usize,
    batch_size: i32,
    k: i32,
    in_features: i32,
    out_features: i32,
) -> PyResult<()> {
    spline_cuda::spline_forward_cuda(
        input as *const f32,
        control_points as *const f32,
        output as *mut f32,
        batch_size,
        k,
        in_features,
        out_features,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn spline_backward_cuda(
    grad_output: usize,
    input: usize,
    grad_control_points: usize,
    batch_size: i32,
    k: i32,
    in_features: i32,
    out_features: i32,
) -> PyResult<()> {
    spline_cuda::spline_backward_cuda(
        grad_output as *const f32,
        input as *const f32,
        grad_control_points as *mut f32,
        batch_size,
        k,
        in_features,
        out_features,
    );
    Ok(())
}

/// Python 모듈에 SplineLayer를 등록합니다.
pub fn register_spline_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let spline_module = PyModule::new(py, "spline")?;
    spline_module.add_class::<SplineLayer>()?;

    #[cfg(feature = "cuda")]
    {
        spline_module.add_function(wrap_pyfunction!(spline_interpolate_cuda, spline_module)?)?;
        spline_module.add_function(wrap_pyfunction!(spline_forward_cuda, spline_module)?)?;
        spline_module.add_function(wrap_pyfunction!(spline_backward_cuda, spline_module)?)?;
    }

    parent_module.add_submodule(spline_module)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/spline_cache.rs`

```rust
use crate::layers::spline_cache::SplineCache;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

#[pyclass(name = "SplineCache")]
pub struct PySplineCache {
    inner: SplineCache,
}

#[pymethods]
impl PySplineCache {
    #[new]
    pub fn new(curvature: f32, dimension: usize) -> Self {
        Self {
            inner: SplineCache::new(curvature, dimension),
        }
    }

    pub fn add_point(
        &mut self,
        time: f32,
        state: PyReadonlyArray1<f32>,
        velocity: PyReadonlyArray1<f32>,
    ) {
        self.inner
            .add_point(time, state.as_array(), velocity.as_array());
    }

    pub fn reconstruct<'py>(&self, py: Python<'py>, t: f32) -> Option<&'py PyArray1<f32>> {
        self.inner.reconstruct(t).map(|arr| arr.to_pyarray(py))
    }

    pub fn batch_reconstruct<'py>(
        &self,
        py: Python<'py>,
        timestamps: PyReadonlyArray1<f32>,
    ) -> &'py PyArray2<f32> {
        let arr = self.inner.batch_reconstruct(timestamps.as_array());
        arr.to_pyarray(py)
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

pub fn register_spline_cache_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "spline_cache")?;
    m.add_class::<PySplineCache>()?;
    parent_module.add_submodule(m)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/suppression.rs`

```rust
use crate::layers::suppression::compute_dynamic_suppression;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyfunction]
pub fn compute_suppression_field<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    base: f32,
    linear: f32,
    hyp: f32,
    scale: f32,
) -> &'py PyArray2<f32> {
    let x = x.as_array();
    let out = compute_dynamic_suppression(&x, base, linear, hyp, scale);
    out.to_pyarray(py)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_suppression_field, m)?)?;
    Ok(())
}
```
---
## File: `reality_stone/src/bindings/unified_riemannian.rs`

```rust
// ============================================================================
// 파일: src/bindings/unified_riemannian.rs
// 목적: 통합 리만 레이어 Python 바인딩
// ============================================================================

use crate::layers::unified_riemannian::*;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct PyUnifiedRiemannianLayer {
    inner: UnifiedRiemannianLayer,
}

#[pymethods]
impl PyUnifiedRiemannianLayer {
    #[new]
    #[pyo3(signature = (metric_type, curvature=1.0, input_dim=64, enable_bellman=false, gamma=0.99))]
    fn new(
        metric_type: &str,
        curvature: f32,
        input_dim: usize,
        enable_bellman: bool,
        gamma: f32,
    ) -> PyResult<Self> {
        let mut layer =
            UnifiedRiemannianLayer::new(metric_type, curvature, input_dim, enable_bellman);
        layer.lagrangian_params.gamma = gamma;
        Ok(Self { inner: layer })
    }

    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        target: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<(&'py PyArray2<f32>, Option<PyObject>)> {
        let x_arr = x.as_array();
        let target_arr = target.as_ref().map(|t| t.as_array());

        let output = self.inner.forward(&x_arr, target_arr.as_ref());

        let output_py = output.output.into_pyarray(py);
        let energy_py = output.energy.map(|e| {
            let dict = PyDict::new(py);
            dict.set_item("kinetic", e.kinetic.into_pyarray(py))
                .unwrap();
            dict.set_item("potential", e.potential.into_pyarray(py))
                .unwrap();
            dict.set_item("lagrangian", e.lagrangian.into_pyarray(py))
                .unwrap();
            dict.set_item("bellman_residual", e.bellman_residual.into_pyarray(py))
                .unwrap();
            dict.into()
        });

        Ok((output_py, energy_py))
    }

    fn backward<'py>(
        &self,
        py: Python<'py>,
        grad_output: PyReadonlyArray2<f32>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let grad_arr = grad_output.as_array();
        let x_arr = x.as_array();

        // 더미 캐시 생성
        let cache = crate::layers::unified_riemannian::LayerCache {
            input: x_arr.to_owned(),
            velocity: None,
            metric_values: ndarray::Array2::zeros((x_arr.nrows(), x_arr.ncols())),
        };

        let grads = self.inner.backward(&grad_arr, &x_arr, &cache);
        Ok(grads.grad_input.into_pyarray(py))
    }

    fn geodesic_path<'py>(
        &self,
        py: Python<'py>,
        start: PyReadonlyArray2<f32>,
        end: PyReadonlyArray2<f32>,
        num_steps: usize,
    ) -> PyResult<Vec<&'py PyArray2<f32>>> {
        let start_arr = start.as_array();
        let end_arr = end.as_array();

        let path = self.inner.geodesic_path(&start_arr, &end_arr, num_steps);

        Ok(path.into_iter().map(|p| p.into_pyarray(py)).collect())
    }

    fn compute_energy<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        v: PyReadonlyArray2<f32>,
        x_next: PyReadonlyArray2<f32>,
        reward: PyReadonlyArray1<f32>,
    ) -> PyResult<PyObject> {
        let energy = self.inner.compute_energy(
            &x.as_array(),
            &v.as_array(),
            &x_next.as_array(),
            &reward.as_array(),
        );

        let dict = PyDict::new(py);
        dict.set_item("kinetic", energy.kinetic.into_pyarray(py))?;
        dict.set_item("potential", energy.potential.into_pyarray(py))?;
        dict.set_item("lagrangian", energy.lagrangian.into_pyarray(py))?;
        dict.set_item("bellman_residual", energy.bellman_residual.into_pyarray(py))?;
        Ok(dict.into())
    }

    fn flow_step<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        num_steps: usize,
        learning_rate: f32,
    ) -> PyResult<&'py PyArray2<f32>> {
        let result = self
            .inner
            .flow_step(&x.as_array(), num_steps, learning_rate);
        Ok(result.into_pyarray(py))
    }

    fn update_value_function(
        &mut self,
        x: PyReadonlyArray2<f32>,
        x_next: PyReadonlyArray2<f32>,
        reward: PyReadonlyArray1<f32>,
        learning_rate: f32,
    ) -> PyResult<()> {
        self.inner.update_value_function(
            &x.as_array(),
            &x_next.as_array(),
            &reward.as_array(),
            learning_rate,
        );
        Ok(())
    }

    fn update_metric(
        &mut self,
        x: PyReadonlyArray2<f32>,
        v: PyReadonlyArray2<f32>,
        learning_rate: f32,
    ) -> PyResult<()> {
        self.inner
            .update_metric(&x.as_array(), &v.as_array(), learning_rate);
        Ok(())
    }
}

// 독립 함수들
#[pyfunction]
pub fn compute_metric<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    metric_type: &str,
    curvature: f32,
) -> PyResult<&'py PyArray2<f32>> {
    use crate::layers::metric::*;

    let metric: Box<dyn MetricTensor> = match metric_type {
        "poincare" => Box::new(PoincareMetric::new(curvature)),
        "lorentz" => Box::new(LorentzMetric::new(curvature)),
        "klein" => Box::new(KleinMetric::new(curvature)),
        "diagonal" => Box::new(DiagonalMetric::new(x.shape()[1])),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown metric type: {}",
                metric_type
            )))
        }
    };

    let x_arr = x.as_array();
    let result = metric.compute_metric(&x_arr);
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn geodesic_distance<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    metric_type: &str,
    curvature: f32,
) -> PyResult<&'py PyArray1<f32>> {
    use crate::layers::metric::*;

    let metric_enum = match metric_type {
        "poincare" => MetricType::Poincare(PoincareMetric::new(curvature)),
        "lorentz" => MetricType::Lorentz(LorentzMetric::new(curvature)),
        "klein" => MetricType::Klein(KleinMetric::new(curvature)),
        "diagonal" => MetricType::Diagonal(DiagonalMetric::new(x.shape()[1])),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown metric type: {}",
                metric_type
            )))
        }
    };

    let distance =
        crate::layers::geodesic::geodesic_distance(&metric_enum, &x.as_array(), &y.as_array());
    Ok(distance.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (x, y, metric_type, curvature, t=0.5))]
pub fn geodesic_interpolate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    metric_type: &str,
    curvature: f32,
    t: f32,
) -> PyResult<&'py PyArray2<f32>> {
    use crate::layers::metric::*;

    let metric_enum = match metric_type {
        "poincare" => MetricType::Poincare(PoincareMetric::new(curvature)),
        "lorentz" => MetricType::Lorentz(LorentzMetric::new(curvature)),
        "klein" => MetricType::Klein(KleinMetric::new(curvature)),
        "diagonal" => MetricType::Diagonal(DiagonalMetric::new(x.shape()[1])),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown metric type: {}",
                metric_type
            )))
        }
    };

    let result = crate::layers::geodesic::geodesic_interpolation(
        &metric_enum,
        &x.as_array(),
        &y.as_array(),
        t,
    );
    Ok(result.into_pyarray(py))
}

#[pyfunction(name = "laplace_beltrami_matrix")]
pub fn laplace_beltrami_matrix_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    metric_type: &str,
    curvature: f32,
    sigma: f32,
    eps: f32,
) -> PyResult<&'py PyArray2<f32>> {
    use crate::layers::metric::*;
    let metric_enum = match metric_type {
        "poincare" => MetricType::Poincare(PoincareMetric::new(curvature)),
        "lorentz" => MetricType::Lorentz(LorentzMetric::new(curvature)),
        "klein" => MetricType::Klein(KleinMetric::new(curvature)),
        "diagonal" => MetricType::Diagonal(DiagonalMetric::new(x.shape()[1])),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown metric type: {}",
                metric_type
            )))
        }
    };
    let mat = crate::layers::unified_riemannian::laplace_beltrami_matrix(
        &metric_enum,
        &x.as_array(),
        sigma,
        eps,
    );
    Ok(mat.into_pyarray(py))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyUnifiedRiemannianLayer>()?;
    m.add_function(wrap_pyfunction!(compute_metric, m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_distance, m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_interpolate, m)?)?;
    m.add_function(wrap_pyfunction!(laplace_beltrami_matrix_py, m)?)?;
    Ok(())
}
```
---
## File: `reality_stone/src/layers/bellman.rs`

```rust
use ndarray::{Array2, ArrayView2, Zip};

/// Computes the Geodesic update for a batch of vectors using a diagonal metric approximation.
///
/// This implements the "Cheat Sheet" derivation:
/// 1. Metric w = Sigmoid(x) (representing importance/value density)
/// 2. Christoffel Gamma = 0.5 * (1/w) * dw/dx
/// 3. Force = -Gamma * velocity^2
/// 4. x_new = x + Force * dt
///
/// # Arguments
///
/// * `input` - Input features (Batch Size x Hidden Dim)
/// * `dt` - Time step for the geodesic flow (learning rate factor)
pub fn compute_diagonal_geodesic_update(input: &ArrayView2<f64>, dt: f64) -> Array2<f64> {
    let mut output = Array2::zeros(input.raw_dim());

    // velocity is assumed to be 1.0 (momentum unit) for this simplified flow
    let velocity_sq = 1.0;

    Zip::from(output.rows_mut())
        .and(input.rows())
        .par_for_each(|mut out_row, in_row| {
            for (i, &val) in in_row.iter().enumerate() {
                // 1. Metric Definition: w(x) = Sigmoid(x)
                // To avoid division by zero, we add epsilon or ensure sigmoid range is safe.
                // Sigmoid is naturally (0, 1), so it's safe for division if we don't hit exact 0.
                let w = 1.0 / (1.0 + (-val).exp());
                // 3. Christoffel Symbol (Diagonal): Gamma = 1/2 * (1/w) * dw
                // Gamma = 0.5 * (1/w) * (w * (1-w)) = 0.5 * (1 - w)
                // This simplification works specifically for Sigmoid metric.
                let gamma = 0.5 * (1.0 - w);

                // 4. Geodesic Force
                let force = -gamma * velocity_sq;

                // 5. Update
                out_row[i] = val + force * dt;
            }
        });

    output
}

/// Inverse computation for backpropagation (Simplified)
///
/// For a full layer, we would need the Jacobian of the update function.
/// Given x_new = x + F(x)*dt, dx_new/dx = 1 + F'(x)*dt
pub fn compute_diagonal_geodesic_backward(
    grad_output: &ArrayView2<f64>,
    input: &ArrayView2<f64>,
    dt: f64,
) -> Array2<f64> {
    let mut grad_input = Array2::zeros(input.raw_dim());
    let velocity_sq = 1.0;

    Zip::from(grad_input.rows_mut())
        .and(grad_output.rows())
        .and(input.rows())
        .par_for_each(|mut gin_row, gout_row, in_row| {
            for (i, &val) in in_row.iter().enumerate() {
                let w = 1.0 / (1.0 + (-val).exp());

                // F(x) = -0.5 * (1 - w) * v^2
                // F'(x) = -0.5 * (-dw/dx) * v^2 = 0.5 * w(1-w) * v^2

                let dw = w * (1.0 - w);
                let d_force = 0.5 * dw * velocity_sq;

                // Chain rule: dL/dx = dL/dy * dy/dx
                // dy/dx = 1 + F'(x) * dt
                let dy_dx = 1.0 + d_force * dt;

                gin_row[i] = gout_row[i] * dy_dx;
            }
        });

    grad_input
}
```
