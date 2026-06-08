# LLM Context Chunk

---
## File: `reality_stone/python/reality_stone/layers/rsulf_cuda.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np
import math

try:
    from reality_stone import _rust, build_causal_laplacian
    if _rust is not None:
        PyRSULFLayer = _rust.PyRSULFLayer
        PyGeodesicMemory = _rust.PyGeodesicMemory
        SplineCache = _rust.SplineCache
        PyRiemannianDiffusion = _rust.PyRiemannianDiffusion
        HAS_RUST = True
    else:
        HAS_RUST = False
except ImportError:
    try:
        from reality_stone._rust import PyRSULFLayer, PyGeodesicMemory, SplineCache, PyRiemannianDiffusion, build_causal_laplacian
        HAS_RUST = True
    except ImportError:
        HAS_RUST = False


class RSULFLayerCUDA(nn.Module):
    def __init__(
        self,
        wq: np.ndarray,
        wk: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        d_model: int = 4096,
        r: int = 1024,
        eta: float = 0.01,
        alpha: float = 0.02,
        beta: float = 0.01,
        gamma: float = 0.99,
        seq_len: int = 128,
        window: int = 8,
        global_basis: Optional[Dict] = None,
        original_block: Optional[nn.Module] = None,
        use_fast: bool = True,
        calibration_samples: int = 1024,
        num_heads: int = 1,
        pfc_mode: str = "bilinear",
        pfc_curvature: float = 0.0,
        pfc_max_rel: float = 0.02,
        pfc_window: int = 0,
        pfc_speed_gate: float = 1.0,
        norm_mode: str = "layernorm",
        ffn_mode: str = "gelu",
        use_geodesic_flow: bool = False,
        geodesic_blend: float = 0.0,
    ):
        super().__init__()
        if not HAS_RUST:
            raise RuntimeError("reality_stone._rust not available")
        
        if global_basis is not None:
            self._layer = PyRSULFLayer.new_with_basis(
                wq.astype(np.float32),
                wk.astype(np.float32),
                w1.astype(np.float32),
                w2.astype(np.float32),
                global_basis["u"].astype(np.float32),
                global_basis["rank"],
                d_model, r, eta, alpha, beta, gamma, seq_len, window
            )
        else:
            if use_fast:
                self._layer = PyRSULFLayer.new_fast(
                    wq.astype(np.float32),
                    wk.astype(np.float32),
                    w1.astype(np.float32),
                    w2.astype(np.float32),
                    d_model, r, eta, alpha, beta, gamma, seq_len, window, calibration_samples
                )
            else:
                wq_f = wq.astype(np.float32)
                wk_f = wk.astype(np.float32)
                if wk_f.shape[0] < wq_f.shape[0]:
                    repeat = wq_f.shape[0] // wk_f.shape[0]
                    wk_f = np.tile(wk_f, (repeat, 1))
                b = wq_f.T @ wk_f
                g_sym = (b + b.T) * 0.5
                g_diag = np.abs(np.diag(g_sym)).astype(np.float32)
                g_diag[g_diag < 1e-6] = 1e-6
                g_diag[g_diag > 1e6] = 1e6
                self._layer = PyRSULFLayer.new_with_metric(
                    wq_f,
                    wk_f,
                    w1.astype(np.float32),
                    w2.astype(np.float32),
                    g_diag,
                    d_model, r, eta, alpha, beta, gamma, seq_len, window
                )
        self.d_model = d_model
        self.r = r
        self.seq_len = seq_len
        self.window = window
        self.num_heads = int(max(1, num_heads))
        self.pfc_mode = str(pfc_mode).lower().strip()
        self.pfc_curvature = float(pfc_curvature)
        self.pfc_max_rel = float(pfc_max_rel)
        self.pfc_window = int(max(0, pfc_window))
        self.pfc_speed_gate = float(max(0.0, pfc_speed_gate))
        self.original_block = original_block
        self._cuda_available = False
        self.norm_mode = str(norm_mode).lower().strip()
        self.ffn_mode = str(ffn_mode).lower().strip()
        self._components = self._layer.export_components()
        
        self._ffn_u1 = np.asarray(self._components["ffn_u1"], dtype=np.float32)
        self._ffn_s1 = np.asarray(self._components["ffn_s1"], dtype=np.float32)
        self._ffn_v1 = np.asarray(self._components["ffn_v1"], dtype=np.float32)
        self._ffn_u2 = np.asarray(self._components["ffn_u2"], dtype=np.float32)
        self._ffn_s2 = np.asarray(self._components["ffn_s2"], dtype=np.float32)
        self._ffn_v2 = np.asarray(self._components["ffn_v2"], dtype=np.float32)
        self._curvature = float(self._components["curvature"])
        self._g_inv = torch.from_numpy(np.asarray(self._components["g_inv"], dtype=np.float32))
        
        self.runtime_batch: Optional[int] = None
        self.runtime_seq_len: Optional[int] = None

        self.ln1_weight = nn.Parameter(torch.ones(d_model))
        self.ln1_bias = nn.Parameter(torch.zeros(d_model))
        self.ln2_weight = nn.Parameter(torch.ones(d_model))
        self.ln2_bias = nn.Parameter(torch.zeros(d_model))
        
        self.wq = nn.Parameter(torch.from_numpy(wq).float(), requires_grad=False)
        self.wk = nn.Parameter(torch.from_numpy(wk).float(), requires_grad=False)
        self.wv = nn.Parameter(torch.eye(d_model).float(), requires_grad=False)
        self.wo = nn.Parameter(torch.eye(d_model).float(), requires_grad=False)
        
        self.bq = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.bk = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.bv = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.bo = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        
        self.ffn_u1 = nn.Parameter(torch.from_numpy(self._ffn_u1).float(), requires_grad=False)
        self.ffn_s1 = nn.Parameter(torch.from_numpy(self._ffn_s1).float(), requires_grad=False)
        self.ffn_v1 = nn.Parameter(torch.from_numpy(self._ffn_v1).float(), requires_grad=False)
        self.ffn_u2 = nn.Parameter(torch.from_numpy(self._ffn_u2).float(), requires_grad=False)
        self.ffn_s2 = nn.Parameter(torch.from_numpy(self._ffn_s2).float(), requires_grad=False)
        self.ffn_v2 = nn.Parameter(torch.from_numpy(self._ffn_v2).float(), requires_grad=False)
        self.g_inv_param = nn.Parameter(self._g_inv, requires_grad=False)
        
        self.b1 = nn.Parameter(torch.zeros(self._ffn_u1.shape[0]), requires_grad=False)
        self.b2 = nn.Parameter(torch.zeros(d_model), requires_grad=False)

        self.ffn_gate_w = nn.Parameter(torch.empty(0), requires_grad=False)
        self.ffn_gate_b = nn.Parameter(torch.empty(0), requires_grad=False)
        
        self.use_hybrid_mode = True
        self.engine = "torch"
        
        self.use_geodesic_flow = bool(use_geodesic_flow)
        self.geodesic_blend = float(max(0.0, min(1.0, geodesic_blend)))
        self._eta = float(eta)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._gamma = float(gamma)
        
        self._graph_laplacian_cache: Dict[int, torch.Tensor] = {}
        self._bellman_memory: Optional[torch.Tensor] = None

    def _get_graph_laplacian(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len in self._graph_laplacian_cache:
            L = self._graph_laplacian_cache[seq_len]
            if L.device != device:
                L = L.to(device)
                self._graph_laplacian_cache[seq_len] = L
            return L
        w = max(1, self.window)
        A = torch.zeros(seq_len, seq_len, dtype=torch.float32)
        for i in range(seq_len):
            for j in range(max(0, i - w), i):
                A[i, j] = 1.0 / (1.0 + abs(i - j))
        D = torch.diag(A.sum(dim=1))
        L = D - A
        L = L.to(device)
        self._graph_laplacian_cache[seq_len] = L
        return L

    def _compute_riemannian_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1, keepdim=True)
        delta_x = x - x_mean
        g_inv = self.g_inv_param.to(x.device)
        return delta_x * g_inv.unsqueeze(0).unsqueeze(0)

    def _compute_graph_diffusion(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        L = self._get_graph_laplacian(seq_len, x.device)
        Lx = torch.bmm(L.unsqueeze(0).expand(batch, -1, -1), x)
        return Lx

    def _update_bellman_memory(self, phi: torch.Tensor) -> torch.Tensor:
        if self._bellman_memory is None:
            self._bellman_memory = phi.detach()
        else:
            if self._bellman_memory.shape != phi.shape:
                self._bellman_memory = phi.detach()
            else:
                self._bellman_memory = self._gamma * self._bellman_memory + phi.detach()
        return self._bellman_memory

    def reset_bellman_memory(self):
        self._bellman_memory = None

    def _compute_potential_and_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)
        h = x_flat @ self.ffn_v1
        h = h * self.ffn_s1.unsqueeze(0)
        h = h @ self.ffn_u1.T
        h = h + self.b1.unsqueeze(0)
        if self.ffn_mode in ("swiglu", "silu_gated", "gated"):
            if self.ffn_gate_w.numel() == 0:
                h = F.silu(h)
            else:
                gate = F.linear(x_flat, self.ffn_gate_w, self.ffn_gate_b)
                gate = F.silu(gate)
                h = h * gate
        elif self.ffn_mode in ("silu", "swish"):
            h = F.silu(h)
        else:
            if self.ffn_mode in ("gelu_new", "gelu_new_tanh"):
                h = self._gelu_new(h)
            else:
                h = F.gelu(h)
        f_x = h @ self.ffn_v2
        f_x = f_x * self.ffn_s2.unsqueeze(0)
        f_x = f_x @ self.ffn_u2.T
        f_x = f_x + self.b2.unsqueeze(0)
        phi = 0.5 * (f_x * f_x).sum(dim=-1)
        phi = phi.view(batch, seq_len)
        grad_phi = f_x.view(batch, seq_len, dim)
        return phi, grad_phi

    def _geodesic_update(self, x: torch.Tensor, v_mem: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        phi, grad_phi = self._compute_potential_and_gradient(x)
        g_inv = self.g_inv_param.to(x.device).unsqueeze(0).unsqueeze(0)
        riem_grad = grad_phi * g_inv
        delta_g_x = self._compute_riemannian_laplacian(x)
        L_x = self._compute_graph_diffusion(x)
        V_t = self._update_bellman_memory(phi)
        V_t_expanded = V_t.unsqueeze(-1).expand(-1, -1, dim)
        v = (
            -self._eta * riem_grad
            + self._alpha * delta_g_x
            + self._beta * L_x
            + self._gamma * V_t_expanded * 0.01
        )
        x_next = x + v
        return x_next

    def set_ffn_gate(self, w_gate, b_gate=None):
        self.ffn_gate_w.data = torch.from_numpy(w_gate).float()
        if b_gate is not None:
            self.ffn_gate_b.data = torch.from_numpy(b_gate).float()
        else:
            self.ffn_gate_b.data = torch.zeros(self.ffn_gate_w.size(0))

    def _norm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        dim = x.size(-1)
        mode = self.norm_mode
        if mode in ("rms", "rmsnorm"):
            eps = 1e-5
            v = (x * x).mean(dim=-1, keepdim=True)
            y = x * torch.rsqrt(v + eps)
            y = y * weight
            if bias is not None and bias.numel() == dim and bias.abs().sum().item() != 0.0:
                y = y + bias
            return y
        return F.layer_norm(x, (dim,), weight, bias)

    def _gelu_new(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def _pfc_cap_and_project(self, h: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        # Remove component parallel to h (keeps correction in tangent-ish direction)
        hh = (h * h).mean(dim=-1, keepdim=True).clamp(min=1e-8)
        ch = (corr * h).mean(dim=-1, keepdim=True)
        corr = corr - (ch / hh) * h

        # Relative RMS cap
        h_rms = (h * h).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        c_rms = (corr * corr).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        max_rel = float(self.pfc_max_rel)
        scale = (max_rel * h_rms / c_rms).clamp(max=1.0)
        return corr * scale

    def _pfc_gate(self, v: torch.Tensor) -> torch.Tensor:
        # Gate by local "drift speed" to focus curvature force where path is bending.
        # pfc_speed_gate=0 disables gating.
        sg = float(self.pfc_speed_gate)
        if sg <= 0.0:
            return torch.ones_like(v[..., :1])
        speed = (v * v).mean(dim=-1, keepdim=True).sqrt()
        ref = speed.mean(dim=1, keepdim=True).clamp(min=1e-6)
        gate = (speed / ref).clamp(0.0, 3.0)
        gate = gate.pow(sg)
        return gate

    def _pfc_bilinear(self, h_seq: torch.Tensor) -> torch.Tensor:
        c = float(self.pfc_curvature)
        if c == 0.0:
            return h_seq
        if h_seq.size(1) < 2:
            return h_seq
        w = int(self.pfc_window)
        if w <= 0:
            w = int(h_seq.size(1) - 1)
        if w > int(h_seq.size(1) - 1):
            w = int(h_seq.size(1) - 1)
        v1 = self.ffn_v1
        u2 = self.ffn_u2
        r = int(v1.shape[1])
        if r <= 0:
            return h_seq
        h_out = h_seq.clone()
        h_f = h_seq.float()
        h_tail = h_f[:, -w:, :]
        h_prev = h_f[:, -(w + 1):-1, :]
        v_tail = h_tail - h_prev
        v1_f = v1.float()
        u2_f = u2.float()
        vv1 = v_tail.reshape(-1, h_f.size(-1)) @ v1_f
        hu2 = h_tail.reshape(-1, h_f.size(-1)) @ u2_f
        corr = (vv1 * hu2) @ v1_f.T
        corr = corr * (c / float(r))
        corr = corr.view_as(h_tail)
        corr = self._pfc_cap_and_project(h_tail, corr)
        corr = corr * self._pfc_gate(v_tail)
        h_tail_out = (h_tail + corr).to(dtype=h_seq.dtype)
        h_out[:, -w:, :] = h_tail_out
        return h_out

    def _pfc_accel(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        Universal PFC (trajectory-only):
        use discrete acceleration a_t = h_t - 2 h_{t-1} + h_{t-2} as a proxy for path curvature,
        then damp it: h_t' = h_t - c * a_t (with gating + relative cap).
        """
        c = float(self.pfc_curvature)
        if c == 0.0:
            return h_seq
        if h_seq.size(1) < 3:
            return h_seq

        w = int(self.pfc_window)
        # need 2 prev tokens, so max tail tokens is (T-2)
        if w <= 0:
            w = int(h_seq.size(1) - 2)
        w = min(w, int(h_seq.size(1) - 2))
        if w <= 0:
            return h_seq

        h_out = h_seq.clone()
        h_f = h_seq.float()
        h_t = h_f[:, -w:, :]                 # (..., t)
        h_t1 = h_f[:, -(w + 1):-1, :]        # (..., t-1)
        h_t2 = h_f[:, -(w + 2):-2, :]        # (..., t-2)

        v = h_t - h_t1
        a = h_t - 2.0 * h_t1 + h_t2

        corr = (-c) * a
        corr = self._pfc_cap_and_project(h_t, corr)
        corr = corr * self._pfc_gate(v)
        h_tail_out = (h_t + corr).to(dtype=h_seq.dtype)
        h_out[:, -w:, :] = h_tail_out
        return h_out

    def _apply_pfc(self, h_seq: torch.Tensor) -> torch.Tensor:
        mode = self.pfc_mode
        if mode in ("0", "off", "none", "false", ""):
            return h_seq
        if mode in ("bilinear", "ffn", "legacy"):
            return self._pfc_bilinear(h_seq)
        if mode in ("accel", "acceleration", "geodesic", "universal"):
            return self._pfc_accel(h_seq)
        # Unknown mode: fail safe (no correction)
        return h_seq

    def set_ln1(self, weight, bias=None):
        self.ln1_weight.data = torch.from_numpy(weight).float()
        if bias is not None:
            self.ln1_bias.data = torch.from_numpy(bias).float()

    def set_ln2(self, weight, bias=None):
        self.ln2_weight.data = torch.from_numpy(weight).float()
        if bias is not None:
            self.ln2_bias.data = torch.from_numpy(bias).float()

    def set_attention_weights(self, wv, wo):
        self.wv.data = torch.from_numpy(wv).float()
        self.wo.data = torch.from_numpy(wo).float()

    def set_biases(self, bq=None, bk=None, bv=None, bo=None, b1=None, b2=None):
        if bq is not None: self.bq.data = torch.from_numpy(bq).float()
        if bk is not None: self.bk.data = torch.from_numpy(bk).float()
        if bv is not None: self.bv.data = torch.from_numpy(bv).float()
        if bo is not None: self.bo.data = torch.from_numpy(bo).float()
        if b1 is not None: self.b1.data = torch.from_numpy(b1).float()
        if b2 is not None: self.b2.data = torch.from_numpy(b2).float()

    def forward(
        self,
        x: torch.Tensor,
        v_mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if self.engine == "rust":
            device = x.device
            x_cpu = x.detach().to("cpu", dtype=torch.float32)
            x_np = x_cpu.numpy().astype(np.float32)
            b, t, d = x_np.shape
            out = np.empty((b, t, d), dtype=np.float32)
            v = None
            if v_mem is not None:
                if isinstance(v_mem, np.ndarray):
                    v = v_mem.astype(np.float32)
                elif isinstance(v_mem, torch.Tensor):
                    v = v_mem.detach().to("cpu").numpy().astype(np.float32)
            for i in range(t):
                y, v = self._layer.forward(x_np[:, i, :], v)
                out[:, i, :] = y
            y_t = torch.from_numpy(out).to(device=device, dtype=x.dtype)
            v_out = None if v is None else torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device=device)
            if squeeze:
                y_t = y_t[:, 0, :]
            return y_t, v_out

        batch, seq_len, dim = x.shape

        u = self._norm(x, self.ln1_weight, self.ln1_bias)

        q = F.linear(u, self.wq, self.bq)
        k = F.linear(u, self.wk, self.bk)
        v = F.linear(u, self.wv, self.bv)

        n_head = self.num_heads
        head_dim = dim // n_head
        if head_dim * n_head != dim:
            n_head = 1
            head_dim = dim

        q = q.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
        if k.size(-1) != dim and (k.size(-1) % head_dim) == 0:
            n_kv = int(k.size(-1) // head_dim)
            k = k.view(batch, seq_len, n_kv, head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, n_kv, head_dim).transpose(1, 2)
            if n_kv > 0 and (n_head % n_kv) == 0:
                rep = int(n_head // n_kv)
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            else:
                k = k.repeat(1, n_head, 1, 1)[:, :n_head, :, :]
                v = v.repeat(1, n_head, 1, 1)[:, :n_head, :, :]
        else:
            k = k.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, n_head, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        
        attn_out = F.linear(attn_out, self.wo, self.bo)
        
        x_mid = x + attn_out
        x_mid = self._apply_pfc(x_mid)
        
        w = self._norm(x_mid, self.ln2_weight, self.ln2_bias)
        
        w_flat = w.reshape(-1, dim)
        
        h = w_flat @ self.ffn_v1
        h = h * self.ffn_s1.unsqueeze(0)
        h = h @ self.ffn_u1.T
        
        h = h + self.b1.unsqueeze(0)
        if self.ffn_mode in ("swiglu", "silu_gated", "gated"):
            if self.ffn_gate_w.numel() == 0:
                h = F.silu(h)
            else:
                gate = F.linear(w_flat, self.ffn_gate_w, self.ffn_gate_b)
                gate = F.silu(gate)
                h = h * gate
        elif self.ffn_mode in ("silu", "swish"):
            h = F.silu(h)
        else:
            if self.ffn_mode in ("gelu_new", "gelu_new_tanh"):
                h = self._gelu_new(h)
            else:
                h = F.gelu(h)
        
        out = h @ self.ffn_v2
        out = out * self.ffn_s2.unsqueeze(0)
        out = out @ self.ffn_u2.T
        
        out = out + self.b2.unsqueeze(0)
        
        ffn_out = out.view(batch, seq_len, dim)
        
        # g_inv scaling removed to match GPT-2 Euclidean fidelity
        # g_inv = self.g_inv_param.unsqueeze(0)
        # ffn_out = ffn_out * g_inv
        
        ffn_out = ffn_out * 1.0 
        
        x_out_attn = x_mid + ffn_out
        
        if self.use_geodesic_flow or self.geodesic_blend > 0.0:
            x_out_geo = self._geodesic_update(x, v_mem)
            blend = self.geodesic_blend
            if self.use_geodesic_flow and blend == 0.0:
                blend = 1.0
            x_out = (1.0 - blend) * x_out_attn + blend * x_out_geo
        else:
            x_out = x_out_attn
        
        if squeeze:
            x_out = x_out[:, 0, :]
        return x_out, None

    def forward_step(self, x_t: torch.Tensor, cache: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        if self.engine == "rust":
            device = x_t.device
            x_cpu = x_t.detach().to("cpu", dtype=torch.float32)
            x_np = x_cpu.numpy().astype(np.float32)
            v = None
            if cache is not None:
                if isinstance(cache, np.ndarray):
                    v = cache.astype(np.float32)
                elif isinstance(cache, torch.Tensor):
                    v = cache.detach().to("cpu").numpy().astype(np.float32)
            y, v = self._layer.forward(x_np[:, 0, :], v)
            y_np = np.asarray(y, dtype=np.float32)[:, None, :]
            y_t = torch.from_numpy(y_np).to(device=device, dtype=x_t.dtype)
            v_out = None if v is None else torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device=device)
            return y_t, v_out

        if cache is None:
            cache = {}
        if x_t.dim() != 3 or x_t.size(1) != 1:
            raise ValueError("x_t must have shape (B, 1, D)")
        batch, _, dim = x_t.shape

        u = self._norm(x_t, self.ln1_weight, self.ln1_bias)
        q = F.linear(u, self.wq, self.bq)
        k = F.linear(u, self.wk, self.bk)
        v = F.linear(u, self.wv, self.bv)

        n_head = self.num_heads
        head_dim = dim // n_head
        if head_dim * n_head != dim:
            n_head = 1
            head_dim = dim

        q = q.view(batch, 1, n_head, head_dim).transpose(1, 2)

        k_in = k
        v_in = v
        if k_in.size(-1) != dim and (k_in.size(-1) % head_dim) == 0:
            n_kv = int(k_in.size(-1) // head_dim)
            k_step = k_in.view(batch, 1, n_kv, head_dim).transpose(1, 2)
            v_step = v_in.view(batch, 1, n_kv, head_dim).transpose(1, 2)
            if n_kv > 0 and (n_head % n_kv) == 0:
                rep = int(n_head // n_kv)
                k_step = k_step.repeat_interleave(rep, dim=1)
                v_step = v_step.repeat_interleave(rep, dim=1)
            else:
                k_step = k_step.repeat(1, n_head, 1, 1)[:, :n_head, :, :]
                v_step = v_step.repeat(1, n_head, 1, 1)[:, :n_head, :, :]
        else:
            k_step = k_in.view(batch, 1, n_head, head_dim).transpose(1, 2)
            v_step = v_in.view(batch, 1, n_head, head_dim).transpose(1, 2)

        t = int(cache.get("t", 0))
        k_buf = cache.get("k_buf")
        v_buf = cache.get("v_buf")
        max_len = cache.get("max_len")
        if k_buf is None or v_buf is None:
            if max_len is None:
                max_len = 2048
            max_len = int(max_len)
            k_buf = torch.empty(batch, n_head, max_len, head_dim, device=x_t.device, dtype=q.dtype)
            v_buf = torch.empty(batch, n_head, max_len, head_dim, device=x_t.device, dtype=q.dtype)
            cache["k_buf"] = k_buf
            cache["v_buf"] = v_buf
            cache["max_len"] = max_len
            t = 0
        if t >= int(cache["max_len"]):
            raise RuntimeError("kv cache overflow")
        k_buf[:, :, t : t + 1, :].copy_(k_step)
        v_buf[:, :, t : t + 1, :].copy_(v_step)
        cache["t"] = t + 1
        k_all = k_buf[:, :, : t + 1, :]
        v_all = v_buf[:, :, : t + 1, :]

        attn = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, 1, dim)
        attn = F.linear(attn, self.wo, self.bo)

        x_mid_t = x_t + attn

        hist = cache.get("x_mid_hist")
        if hist is None:
            x_mid_seq = x_mid_t
        else:
            x_mid_seq = torch.cat([hist, x_mid_t], dim=1)
        w = int(self.pfc_window)
        if w <= 0:
            w = x_mid_seq.size(1) - 1
        keep = min(x_mid_seq.size(1), max(3, w + 2))
        x_mid_seq = x_mid_seq[:, -keep:, :]
        x_mid_seq = self._apply_pfc(x_mid_seq)
        x_mid_t = x_mid_seq[:, -1:, :]

        w2 = self._norm(x_mid_t, self.ln2_weight, self.ln2_bias)
        w_flat = w2.reshape(-1, dim)

        h = w_flat @ self.ffn_v1
        h = h * self.ffn_s1.unsqueeze(0)
        h = h @ self.ffn_u1.T
        h = h + self.b1.unsqueeze(0)
        if self.ffn_mode in ("swiglu", "silu_gated", "gated"):
            if self.ffn_gate_w.numel() == 0:
                h = F.silu(h)
            else:
                gate = F.linear(w_flat, self.ffn_gate_w, self.ffn_gate_b)
                gate = F.silu(gate)
                h = h * gate
        elif self.ffn_mode in ("silu", "swish"):
            h = F.silu(h)
        else:
            if self.ffn_mode in ("gelu_new", "gelu_new_tanh"):
                h = self._gelu_new(h)
            else:
                h = F.gelu(h)

        out = h @ self.ffn_v2
        out = out * self.ffn_s2.unsqueeze(0)
        out = out @ self.ffn_u2.T
        out = out + self.b2.unsqueeze(0)
        ffn_out = out.view(batch, 1, dim)
        x_out_t = x_mid_t + ffn_out

        cache["x_mid_hist"] = x_mid_seq.detach()
        return x_out_t, cache

    def param_count(self) -> Tuple[int, int, float]:
        return self._layer.param_count()

    @property
    def curvature(self) -> float:
        return self._layer.curvature

    @property
    def eta(self) -> float:
        return self._layer.eta

    @property
    def alpha(self) -> float:
        return self._layer.alpha

    @property
    def beta(self) -> float:
        return self._layer.beta

    @property
    def gamma(self) -> float:
        return self._layer.gamma

    @property
    def g_diag(self) -> np.ndarray:
        return self._layer.g_diag

    @property
    def g_inv(self) -> np.ndarray:
        return self._layer.g_inv


class RSULFWrapperCUDA(nn.Module):
    def __init__(self, rsulf_layer: RSULFLayerCUDA):
        super().__init__()
        self.rsulf = rsulf_layer
        self.original_block = rsulf_layer.original_block
        self.v_mem: Optional[torch.Tensor] = None
        self.d_model = rsulf_layer.d_model
        self.time_step = 0
        self.geodesic_memory = getattr(rsulf_layer, "geodesic_memory", None)

    def reset_memory(self):
        self.v_mem = None
        self.time_step = 0
        if hasattr(self.rsulf, 'reset_bellman_memory'):
            self.rsulf.reset_bellman_memory()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.original_block is not None:
            raise RuntimeError("original_block path is disabled")
        
        out, v = self.rsulf(x, self.v_mem)
        self.v_mem = v
        self.time_step += int(x.size(1))
        return out

    def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.original_block is not None:
            raise RuntimeError("original_block path is disabled")
        out, cache = self.rsulf.forward_step(x_t, cache=self.v_mem)
        self.v_mem = cache
        self.time_step += 1
        return out

    def init_step_cache(self, batch: int, max_len: int, device: torch.device, dtype: torch.dtype):
        if getattr(self.rsulf, "engine", None) == "rust":
            self.v_mem = None
            return
        dim = int(self.d_model)
        n_head = int(self.rsulf.num_heads)
        head_dim = dim // max(1, n_head)
        if head_dim * n_head != dim:
            n_head = 1
            head_dim = dim
        self.v_mem = {
            "t": 0,
            "max_len": int(max_len),
            "k_buf": torch.empty(batch, n_head, int(max_len), head_dim, device=device, dtype=dtype),
            "v_buf": torch.empty(batch, n_head, int(max_len), head_dim, device=device, dtype=dtype),
            "x_mid_hist": None,
        }


class RSULFLMHeadCUDA(nn.Module):
    def __init__(
        self,
        rsulf_layers: list,
        hidden_size: int,
        vocab_size: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.rsulf_wrappers = nn.ModuleList([
            RSULFWrapperCUDA(layer) for layer in rsulf_layers
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for wrapper in self.rsulf_wrappers:
            x = wrapper(x)
        return self.lm_head(x)
```
---
## File: `reality_stone/python/reality_stone/layers/spline.py`

```python
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .. import _rust

class SplineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 8, bias: bool = True, use_residual: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.use_residual = use_residual
        self._cached_weight = None  # cached materialized weight for eval
        self.register_buffer('blend_matrix', None)  # [out_features, k+1]
        
        self.control_points = nn.Parameter(torch.randn(k + 1, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        if use_residual:
            self.residual = nn.Parameter(torch.zeros(out_features, in_features))
        else:
            self.register_parameter('residual', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Fast factored forward: y = (x @ C^T) @ B^T [+ x @ R^T] + b
        if self.blend_matrix is None or self.blend_matrix.shape[0] != self.out_features or self.blend_matrix.shape[1] != (self.k + 1):
            self._refresh_blend_matrix()

        # (batch, k+1)
        proj_k = input.matmul(self.control_points.t())
        # (batch, out)
        y = proj_k.matmul(self.blend_matrix.t())
        if self.use_residual and self.residual is not None:
            y = y + input.matmul(self.residual.t())
        if self.bias is not None:
            y = y + self.bias
        return y

    @staticmethod
    def interpolate_weights_static(control_points, k, out_features):
        weights = []
        for i in range(out_features):
            t = i / (out_features - 1)
            t_scaled = t * k
            j = int(np.floor(t_scaled))
            j = max(1, min(j, k - 2))
            t_local = t_scaled - j
            
            t2, t3 = t_local * t_local, t_local * t_local * t_local
            c0 = -0.5 * t3 + t2 - 0.5 * t_local
            c1 = 1.5 * t3 - 2.5 * t2 + 1.0
            c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t_local
            c3 = 0.5 * t3 - 0.5 * t2
            
            weight_row = (c0 * control_points[j-1] + 
                         c1 * control_points[j] + 
                         c2 * control_points[j+1] + 
                         c3 * control_points[j+2])
            weights.append(weight_row)
        return torch.stack(weights)

    def interpolate_weights_torch(self) -> torch.Tensor:
        return self.interpolate_weights_static(self.control_points, self.k, self.out_features)

    def precompute_weight(self) -> None:
        """Precompute and cache the materialized weight for faster inference."""
        with torch.no_grad():
            self._cached_weight = self.interpolate_weights_torch().to(device=self.control_points.device, dtype=self.control_points.dtype)

    def _refresh_blend_matrix(self) -> None:
        # Build dense blend matrix [out_features, k+1] with at most 4 non-zeros per row
        import numpy as np
        k = self.k
        out = self.out_features
        B = np.zeros((out, k + 1), dtype='float32')
        for i in range(out):
            t = i / max(1, (out - 1))
            t_scaled = t * k
            j = int(np.floor(t_scaled))
            j = max(1, min(j, k - 2))
            t_local = t_scaled - j
            t2, t3 = t_local * t_local, t_local * t_local * t_local
            c0 = -0.5 * t3 + t2 - 0.5 * t_local
            c1 = 1.5 * t3 - 2.5 * t2 + 1.0
            c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t_local
            c3 = 0.5 * t3 - 0.5 * t2
            idx0 = max(0, j - 1)
            idx1 = j
            idx2 = min(k, j + 1)
            idx3 = min(k, j + 2)
            B[i, idx0] += c0
            B[i, idx1] += c1
            B[i, idx2] += c2
            B[i, idx3] += c3
        bm = torch.from_numpy(B).to(device=self.control_points.device, dtype=self.control_points.dtype)
        self.blend_matrix = bm

    @classmethod
    def from_linear(cls, linear: nn.Linear, k: int = 8, 
                   learning_rate: float = 0.01, steps: int = 100, use_residual: bool = True) -> 'SplineLinear':
        spline_layer = cls(linear.in_features, linear.out_features, k, 
                          bias=(linear.bias is not None), use_residual=use_residual)
        
        weight_np = linear.weight.detach().cpu().numpy()
        
        rust_spline_instance = _rust.spline.SplineLayer.from_weight_py(
            weight_np, k, learning_rate, steps
        )
        
        optimized_control_points = torch.from_numpy(
            rust_spline_instance.control_points
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        
        spline_layer.control_points.data.copy_(optimized_control_points)
        
        if use_residual:
            interpolated_weight = spline_layer.interpolate_weights_torch().detach()
            spline_layer.residual.data.copy_(linear.weight.data - interpolated_weight)
        
        if linear.bias is not None:
            spline_layer.bias.data.copy_(linear.bias.data)

        # Cache materialized weight for fast inference
        spline_layer.precompute_weight()
        
        return spline_layer

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, k={self.k}, '
                f'use_residual={self.use_residual}, compression_ratio={self.get_compression_ratio():.1f}x')

    def get_compression_ratio(self) -> float:
        original_params = self.in_features * self.out_features
        compressed_params = self.control_points.numel()
        if self.use_residual and self.residual is not None:
            compressed_params += self.residual.numel()
        return original_params / compressed_params if compressed_params > 0 else float('inf')
```
---
## File: `reality_stone/python/reality_stone/layers/suppression.py`

```python
import torch
import torch.nn as nn
from torch import Tensor


class HyperbolicSuppressionField(nn.Module):
    def __init__(self, base: float = 0.37, linear: float = 0.0, hyp: float = 0.1, scale: float = 1.0) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.tensor(float(base)))
        self.linear = nn.Parameter(torch.tensor(float(linear)))
        self.hyp = nn.Parameter(torch.tensor(float(hyp)))
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def compute_field(self, x: Tensor) -> Tensor:
        x_cast = x.to(dtype=self.base.dtype)
        return self.base + self.linear * x_cast + self.hyp * torch.tanh(self.scale * x_cast)

    def compute_effective_temperature(self, t0, x: Tensor) -> Tensor:
        if torch.is_tensor(t0):
            base_temp = t0.to(device=x.device, dtype=x.dtype)
        else:
            base_temp = torch.as_tensor(t0, device=x.device, dtype=x.dtype)
        field = self.compute_field(x).to(device=x.device, dtype=x.dtype)
        scale = torch.sigmoid(field)
        return base_temp * scale
```
---
## File: `reality_stone/python/reality_stone/losses.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_same_label(dists_sq: torch.Tensor, labels: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    batch_size = dists_sq.size(0)
    if batch_size < 2:
        return dists_sq.new_tensor(0.0)
    sim = -dists_sq / tau
    adj = F.softmax(sim, dim=-1)
    labels = labels.view(-1, 1)
    device = dists_sq.device
    mask_same = torch.eq(labels, labels.T).to(device)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    mask_same = mask_same & ~self_mask
    if mask_same.sum() == 0:
        return dists_sq.new_tensor(0.0)
    weighted = adj * dists_sq * mask_same.float()
    return weighted.sum() / mask_same.float().sum()


def poincare_kinetic_energy(x_hyp: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    norm_sq = torch.sum(x_hyp ** 2, dim=-1, keepdim=True)
    norm_sq = torch.clamp(norm_sq, max=(1.0 / curvature) - 1e-5)
    lambda_x = 2.0 / (1.0 - curvature * norm_sq)
    kinetic = 0.5 * (lambda_x ** 2) * norm_sq
    return kinetic.mean()


class HyperbolicSupConLoss(nn.Module):
    """
    Hyperbolic Supervised Contrastive Loss
    Uses Poincaré distance for contrastive learning.
    """
    def __init__(self, temperature: float = 0.1, curvature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.curvature = curvature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (batch_size, dim) - Points in Poincaré ball
        labels: (batch_size)
        """
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute pairwise Poincaré distances squared
        # d(u,v) = arccosh(1 + 2 * |u-v|^2 / ((1-|u|^2)(1-|v|^2)))
        # For numerical stability, we use the pre-computed distance or compute it here.
        # Assuming inputs are in the ball.
        
        x_norm_sq = torch.sum(features.pow(2), dim=1, keepdim=True)
        # Clamp for stability
        x_norm_sq = torch.clamp(x_norm_sq, max=(1.0/self.curvature) - 1e-5)
        
        # Pairwise Euclidean distance squared
        dist_euc_sq = torch.cdist(features, features, p=2).pow(2)
        
        alpha = 1.0 - self.curvature * x_norm_sq
        denom = torch.mm(alpha, alpha.T)
        gamma = 1.0 + 2.0 * self.curvature * dist_euc_sq / torch.clamp(denom, min=1e-10)
        dist_hyp = (1.0 / torch.sqrt(torch.tensor(self.curvature))) * torch.acosh(torch.clamp(gamma, min=1.0 + 1e-7))
        
        # Logits: negative distance / temperature
        logits = -dist_hyp / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        
        # Mean log-likelihood for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask.sum(1), min=1.0)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


class BellmanConsistencyLoss(nn.Module):
    def __init__(self, lambda_bellman: float = 0.1, gamma: float = 0.99, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.lambda_bellman = lambda_bellman
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, apply_bellman: bool = True) -> dict:
        loss_cls = self.ce_loss(logits, labels)
        if not apply_bellman or self.lambda_bellman == 0:
            zero = logits.new_tensor(0.0)
            return {"total": loss_cls, "classification": loss_cls, "bellman": zero}
        batch_size = logits.size(0)
        idx = torch.arange(batch_size, device=logits.device)
        current_values = logits[idx, labels]
        preds = logits.argmax(dim=1)
        rewards = (preds == labels).float()
        next_values = logits.max(dim=1)[0].detach()
        target = rewards + self.gamma * next_values
        bellman_error = (current_values - target).pow(2).mean()
        total = loss_cls + self.lambda_bellman * bellman_error
        return {"total": total, "classification": loss_cls, "bellman": bellman_error}
```
---
## File: `reality_stone/python/reality_stone/metrikey.py`

```python
from __future__ import annotations

import numpy as np
import torch

from ._fallback import deterministic_spd


def spd_metric_from_key_weighted(
    key: str,
    dim: int,
    min_lambda: float = 0.8,
    max_lambda: float = 1.2,
    mass: float = 1.0,
) -> np.ndarray:
    return deterministic_spd(key, dim, min_lambda, max_lambda, mass)


def spd_metric_from_key(
    key: str,
    dim: int,
    min_lambda: float = 0.8,
    max_lambda: float = 1.2,
) -> np.ndarray:
    return spd_metric_from_key_weighted(key, dim, min_lambda, max_lambda, 1.0)


def metric_factor_cholesky(g) -> np.ndarray:
    g_arr = np.asarray(g, dtype=np.float32)
    g_arr = 0.5 * (g_arr + g_arr.T)
    jitter = np.eye(g_arr.shape[0], dtype=np.float32) * 1e-5
    return np.linalg.cholesky(g_arr + jitter).astype(np.float32)


def metric_from_keys(
    keys,
    dim: int,
    min_lambda: float = 0.8,
    max_lambda: float = 1.2,
    masses=None,
):
    if masses is None:
        masses = [1.0] * len(keys)
    acc = np.zeros((dim, dim), dtype=np.float32)
    total = 0.0
    for key, mass in zip(keys, masses):
        m = float(mass)
        acc += spd_metric_from_key_weighted(str(key), dim, min_lambda, max_lambda, max(m, 1e-6))
        total += max(m, 1e-6)
    if total <= 0.0:
        total = 1.0
    return torch.from_numpy((acc / total).astype(np.float32))


def mahalanobis_distance_sq_g(x, y, g) -> float:
    dx = np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32)
    g_arr = np.asarray(g, dtype=np.float32)
    return float(dx.T @ g_arr @ dx)


def mahalanobis_distance_sq_l(x, y, l_factor) -> float:
    dx = np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32)
    l_arr = np.asarray(l_factor, dtype=np.float32)
    z = l_arr @ dx
    return float(z.T @ z)
```
---
## File: `reality_stone/python/reality_stone/models/__init__.py`

```python
try:
    from .hierarchical_sentence_topic_llm import (
        HierarchicalLLMConfig,
        HierarchicalSentenceTopicLLM,
        SentenceTopicHead,
        MetricContextRouter,
        HierarchicalLMDecoder,
        RCELexicalDecoder,
        HAS_METRIKEY,
    )
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False
    HierarchicalLLMConfig = None
    HierarchicalSentenceTopicLLM = None
    SentenceTopicHead = None
    MetricContextRouter = None
    HierarchicalLMDecoder = None
    RCELexicalDecoder = None
    HAS_METRIKEY = False

try:
    from .transformer_converter import (
        RSULFConfig,
        RSULFTransformerConverter,
        convert_transformer_to_rsulf,
    )
    _HAS_CONVERTER = True
except ImportError:
    _HAS_CONVERTER = False
    RSULFConfig = None
    RSULFTransformerConverter = None
    convert_transformer_to_rsulf = None

from .riemannian_aggregation import RiemannianAggregation

__all__ = [
    "RiemannianAggregation",
    "HierarchicalLLMConfig",
    "HierarchicalSentenceTopicLLM",
    "SentenceTopicHead",
    "MetricContextRouter",
    "HierarchicalLMDecoder",
    "RCELexicalDecoder",
    "HAS_METRIKEY",
    "RSULFConfig",
    "RSULFTransformerConverter",
    "convert_transformer_to_rsulf",
    "_HAS_LLM",
    "_HAS_CONVERTER",
]
```
---
## File: `reality_stone/python/reality_stone/models/bottom_up_encoder.py`

```python
import torch
import torch.nn as nn
from typing import Optional

from reality_stone.models.riemannian_aggregation import RiemannianAggregation
from reality_stone.layers.poincare import project_to_ball


class BottomUpEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_head: int = 64,
        manifold: str = "poincare",
        c: float = 1e-3,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        
        self.token_to_sentence = RiemannianAggregation(d_model=d_model, manifold=manifold, c=c, temperature=temperature)
        
        self.sentence_to_paragraph = RiemannianAggregation(d_model=d_model, manifold=manifold, c=c, temperature=temperature)
        
        self.poincare_proj = nn.Linear(d_model, d_head)
    
    def encode_tokens_to_sentences(
        self,
        token_embeddings: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, L, d = token_embeddings.shape
        
        sentence_list = []
        for t in range(T):
            tokens_t = token_embeddings[:, t, :, :]
            
            if metric_ctx is not None:
                metric_t = metric_ctx[:, t, :, :]
            else:
                metric_t = None
            
            sent_emb = self.token_to_sentence(children_states=tokens_t, metric_ctx=metric_t)
            
            sentence_list.append(sent_emb)
        
        sentence_embeddings = torch.stack(sentence_list, dim=1)
        
        return sentence_embeddings
    
    def encode_sentences_to_paragraph(
        self,
        sentence_embeddings: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d = sentence_embeddings.shape
        
        paragraph_emb = self.sentence_to_paragraph(
            children_states=sentence_embeddings,
            metric_ctx=metric_ctx,
        )
        
        return paragraph_emb
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        sentence_metric: Optional[torch.Tensor] = None,
        paragraph_metric: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sentence_embeddings = self.encode_tokens_to_sentences(
            token_embeddings,
            metric_ctx=sentence_metric,
        )
        
        paragraph_embedding = self.encode_sentences_to_paragraph(
            sentence_embeddings,
            metric_ctx=paragraph_metric,
        )
        
        return sentence_embeddings, paragraph_embedding
```
---
## File: `reality_stone/python/reality_stone/models/hierarchical_sentence_topic_llm.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import reality_stone as rs

from .riemannian_aggregation import RiemannianAggregation
from reality_stone.layers.metric_attention import MetricAttention
from reality_stone.layers.poincare import project_to_ball, poincare_distance
from reality_stone.layers.lorentz import from_poincare, lorentz_distance
from reality_stone.layers.suppression import HyperbolicSuppressionField
from reality_stone.models.semantic_preservation import SemanticPreservationLoss
from .pretrained_backbone import PretrainedBackbone
from reality_stone.utils.pre_segmenter import DocumentTree

try:
    from reality_stone.data import SentenceTopicDataset, collate_batch
    _HAS_SENTENCE_TOPIC_DATASET = True
except Exception:
    _HAS_SENTENCE_TOPIC_DATASET = False


@dataclass
class HierarchicalLLMConfig:
    vocab_size: int = 32000
    d_model: int = 768
    d_head: int = 64
    num_topics: int = 8
    num_heads_topic: int = 4
    n_layer_decoder: int = 6
    n_head_decoder: int = 8
    c_poincare: float = 1e-3
    c_lorentz: float = -1.0
    
    pretrained_decoder_path: Optional[str] = None
    pretrained_tokenizer: Optional[str] = None
    use_pretrained_embeddings: bool = True
    
    lambda_consistency: float = 0.5
    lambda_diversity: float = 0.1
    lambda_consistency_schedule: str = "constant"
    lambda_diversity_schedule: str = "constant"
    lambda_topic_supervision: float = 0.5
    lambda_metric: float = 0.1
    # Turn curvature correction on by default (user-requested: "작동 돌려놔")
    lambda_curvature: float = 0.1
    curvature_target_poincare: float = 1e-3
    curvature_target_lorentz: float = -1.0
    curvature_target_klein: float = 1e-3
    enable_dynamic_manifold: bool = True
    
    manifold_sentence: str = "poincare"
    manifold_paragraph: str = "poincare"
    temperature_agg: float = 1.0
    
    gamma_up: float = 0.3
    gamma_self: float = 0.5
    gamma_down: float = 0.2
    
    max_answer_sentences: int = 20
    lambda_length: float = 0.2
    lambda_semantic: float = 0.3
    max_lm_seq_len: int = 1024
    
    freeze_decoder: bool = False
    freeze_topic_head_backbone: bool = False
    
    lr_backbone: float = 1e-4
    lr_metric: float = 1e-3
    
    lambda_edit: float = 0.0
    max_edit_ratio: float = 0.25
    enable_structural_edit: bool = False
    edit_budget: float = 0.25
    use_fast_spd_mixing: bool = True
    
    logit_clip_value: float = 20.0
    loss_clip_max: float = 100.0
    spd_eps: float = 1e-5
    spd_eigval_min: float = 1e-5
    spd_eigval_max: float = 1e5
    spd_log_eigval_clip: float = 10.0
    metric_lambda_min: float = 0.1
    metric_lambda_max: float = 5.0
    grad_clip_norm: float = 1.0 
    
    # SFE Variable Suppression Parameters
    suppression_base: float = 0.37
    suppression_linear: float = 0.0
    suppression_hyp: float = 0.1
    suppression_scale: float = 1.0
    enable_variable_suppression: bool = True
    diffusion_steps: int = 0
    use_diffusion_hidden: bool = False
    diffusion_alpha: float = 0.9
    diffusion_dt: float = 0.1


class EditOperationHead(nn.Module):
    def __init__(self, d_model: int, num_ops: int = 5, edit_budget: float = 0.25) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_ops = num_ops
        self.edit_budget = edit_budget
        self.proj = nn.Linear(d_model, num_ops)
        self.value_proj = nn.Linear(d_model, d_model)
        for p in self.value_proj.parameters():
            p.requires_grad = False

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)
    
    def apply_edits(
        self,
        tokens: torch.Tensor,
        edit_logits: torch.Tensor,
        pred_tokens: torch.Tensor,
        enable_structural: bool = False,
        replacement_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S = tokens.shape
        device = tokens.device
        
        if not enable_structural:
            if replacement_mask is not None:
                return torch.where(replacement_mask.bool(), pred_tokens, tokens)
            return tokens
        
        ops = torch.argmax(edit_logits, dim=-1)
        max_edits = max(1, int(S * self.edit_budget))
        
        result_tokens = []
        for b in range(B):
            new_seq = []
            insert_count = 0
            delete_count = 0
            replace_count = 0
            
            for i in range(S):
                op = int(ops[b, i].item())
                tok = int(tokens[b, i].item())
                pred_tok = int(pred_tokens[b, i].item())
                
                if tok == 0:
                    continue
                
                is_replaceable = True
                if replacement_mask is not None:
                    is_replaceable = bool(replacement_mask[b, i].item())
                
                if op == 0:
                    new_seq.append(tok)
                elif op == 1 and is_replaceable and replace_count < max_edits:
                    new_seq.append(pred_tok)
                    replace_count += 1
                elif op == 2 and insert_count < max_edits:
                    new_seq.append(pred_tok)
                    new_seq.append(tok)
                    insert_count += 1
                elif op == 3 and insert_count < max_edits:
                    new_seq.append(tok)
                    new_seq.append(pred_tok)
                    insert_count += 1
                elif op == 4 and delete_count < max_edits:
                    delete_count += 1
                    continue
                else:
                    new_seq.append(tok)
            
            result_tokens.append(new_seq)
        
        if not result_tokens or all(len(seq) == 0 for seq in result_tokens):
            return tokens
        
        max_len = max(len(seq) for seq in result_tokens)
        padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b, seq in enumerate(result_tokens):
            if seq:
                padded[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        
        return padded


class SentenceOrderHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, 1)

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        scores = self.proj(sentence_embeddings)
        return scores.squeeze(-1)


class TreeNodeOperator(nn.Module):
    def __init__(
        self,
        d_model: int,
        manifold: str = "poincare",
        c: float = 1e-3,
        enable_dynamic_manifold: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.manifold = manifold
        self.c = c
        self.enable_dynamic_manifold = enable_dynamic_manifold
        self.aggregator = RiemannianAggregation(d_model, manifold, c, temperature=1.0)
        if enable_dynamic_manifold:
            c_min = 1e-6
            c_max = 5e-2
            c_init = min(max(abs(float(c)), c_min), c_max)
            logit_init = torch.logit(torch.tensor((c_init - c_min) / (c_max - c_min)))
            self.kappa_poincare = nn.Parameter(logit_init.clone())
            self.kappa_lorentz = nn.Parameter(logit_init.clone())
            self.kappa_klein = nn.Parameter(logit_init.clone())
        
        if enable_dynamic_manifold:
            self.manifold_selector = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 3),
            )
            self.aggregator_poincare = RiemannianAggregation(d_model, "poincare", c, temperature=1.0)
            self.aggregator_lorentz = RiemannianAggregation(d_model, "lorentz", c, temperature=1.0)
            self.aggregator_klein = RiemannianAggregation(d_model, "klein", c, temperature=1.0)

    def _curvatures(self, device: torch.device, dtype: torch.dtype):
        if not self.enable_dynamic_manifold:
            c = torch.as_tensor(abs(float(self.c)), device=device, dtype=dtype)
            return c, c, c
        s_p = torch.sigmoid(self.kappa_poincare).to(device=device, dtype=dtype)
        s_l = torch.sigmoid(self.kappa_lorentz).to(device=device, dtype=dtype)
        s_k = torch.sigmoid(self.kappa_klein).to(device=device, dtype=dtype)
        c_min = torch.as_tensor(1e-6, device=device, dtype=dtype)
        c_max = torch.as_tensor(5e-2, device=device, dtype=dtype)
        c_p = c_min + (c_max - c_min) * s_p
        c_l = c_min + (c_max - c_min) * s_l
        c_k = c_min + (c_max - c_min) * s_k
        return c_p, c_l, c_k
    
    def up_operator(
        self,
        children_embeddings: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c_p, c_l, c_k = self._curvatures(children_embeddings.device, children_embeddings.dtype)
        if not self.enable_dynamic_manifold:
            return self.aggregator(children_embeddings, metric_ctx, c_override=c_p)
        
        B = children_embeddings.shape[0]
        mean_emb = children_embeddings.mean(dim=1)
        manifold_logits = self.manifold_selector(mean_emb)
        manifold_probs = torch.softmax(manifold_logits, dim=-1)
        
        result_poincare = self.aggregator_poincare(children_embeddings, metric_ctx, c_override=c_p)
        result_lorentz = self.aggregator_lorentz(children_embeddings, metric_ctx, c_override=c_l)
        result_klein = self.aggregator_klein(children_embeddings, metric_ctx, c_override=c_k)
        
        results = torch.stack([result_poincare, result_lorentz, result_klein], dim=1)
        weighted_result = (results * manifold_probs.unsqueeze(-1)).sum(dim=1)
        
        return weighted_result
    
    def down_operator(
        self,
        parent_embedding: torch.Tensor,
        num_children: int,
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = parent_embedding.shape[0]
        parent_exp = parent_embedding.unsqueeze(1).expand(B, num_children, self.d_model)
        
        if self.enable_dynamic_manifold and hasattr(self, 'manifold_selector'):
            manifold_logits = self.manifold_selector(parent_embedding)
            manifold_probs = torch.softmax(manifold_logits, dim=-1)
            
            noise = torch.randn_like(parent_exp) * 0.01
            parent_exp = parent_exp + noise
        
        return parent_exp


class LevelInvariantTreeProcessor(nn.Module):
    def __init__(self, d_model: int, enable_dynamic_manifold: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.enable_dynamic_manifold = enable_dynamic_manifold
        self.node_operators: Dict[str, TreeNodeOperator] = nn.ModuleDict({
            "document": TreeNodeOperator(d_model, "poincare", 1e-3, enable_dynamic_manifold),
            "paragraph": TreeNodeOperator(d_model, "poincare", 1e-3, enable_dynamic_manifold),
            "sentence": TreeNodeOperator(d_model, "poincare", 1e-3, enable_dynamic_manifold),
            "token": TreeNodeOperator(d_model, "poincare", 1e-3, enable_dynamic_manifold),
        })
    
    def process_tree(
        self,
        tree: DocumentTree,
        node_embeddings: Dict[int, torch.Tensor],
        direction: str = "up",
    ) -> Dict[int, torch.Tensor]:
        result_embeddings: Dict[int, torch.Tensor] = {}
        
        if direction == "up":
            sorted_nodes = sorted(tree.nodes, key=lambda n: -self._depth(tree, n.id))
            for node in sorted_nodes:
                children_ids = tree.children(node.id)
                if not children_ids:
                    if node.id in node_embeddings:
                        result_embeddings[node.id] = node_embeddings[node.id]
                    else:
                        continue
                else:
                    available_children = [cid for cid in children_ids if cid in result_embeddings]
                    if not available_children:
                        if node.id in node_embeddings:
                            result_embeddings[node.id] = node_embeddings[node.id]
                        continue
                    children_embs = torch.stack([result_embeddings[cid] for cid in available_children])
                    if children_embs.dim() == 2:
                        children_embs = children_embs.unsqueeze(0)
                    
                    operator = self.node_operators[node.type] if node.type in self.node_operators else None
                    if operator:
                        result_embeddings[node.id] = operator.up_operator(children_embs).squeeze(0)
                    else:
                        result_embeddings[node.id] = children_embs.mean(dim=1).squeeze(0)
        
        elif direction == "down":
            sorted_nodes = sorted(tree.nodes, key=lambda n: self._depth(tree, n.id))
            for node in sorted_nodes:
                children_ids = tree.children(node.id)
                if node.id in result_embeddings:
                    parent_emb = result_embeddings[node.id]
                elif node.id in node_embeddings:
                    parent_emb = node_embeddings[node.id]
                else:
                    continue
                
                if children_ids:
                    operator = self.node_operators[node.type] if node.type in self.node_operators else None
                    if operator:
                        parent_emb_batched = parent_emb.unsqueeze(0) if parent_emb.dim() == 1 else parent_emb
                        children_embs = operator.down_operator(parent_emb_batched, len(children_ids))
                        for idx, cid in enumerate(children_ids):
                            result_embeddings[cid] = children_embs[0, idx]
        
        return result_embeddings
    
    def _depth(self, tree: DocumentTree, node_id: int) -> int:
        if not hasattr(self, '_depth_cache'):
            self._depth_cache = {}
        if node_id in self._depth_cache:
            return self._depth_cache[node_id]
        
        node = next((n for n in tree.nodes if n.id == node_id), None)
        if node is None or node.parent is None:
            self._depth_cache[node_id] = 0
            return 0
        depth = 1 + self._depth(tree, node.parent)
        self._depth_cache[node_id] = depth
        return depth


def compute_dynamic_lambda(
    base_lambda: float,
    schedule: str,
    current_epoch: int,
    total_epochs: int,
) -> float:
    if schedule == "constant":
        return base_lambda
    
    progress = current_epoch / max(total_epochs, 1)
    
    if schedule == "decay":
        return base_lambda * (1.0 - 0.9 * progress)
    elif schedule == "grow":
        return base_lambda * (0.1 + 0.9 * progress)
    elif schedule == "warmup":
        warmup_ratio = 0.1
        if progress < warmup_ratio:
            min_factor = 0.1
            return base_lambda * (min_factor + (1.0 - min_factor) * (progress / warmup_ratio))
        else:
            return base_lambda
    
    return base_lambda


class RiemannianDiffusionStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: torch.Tensor, flow: torch.Tensor, diffusion_engine, alpha: float, dt: float) -> torch.Tensor:
        h = h.contiguous()
        flow = flow.contiguous()
        h_next = torch.empty_like(h)
        batch_size, dim = h.shape
        if h.is_cuda and getattr(rs, "_has_cuda", False) and diffusion_engine is not None:
            diffusion_engine.step_cuda(
                h.data_ptr(),
                flow.data_ptr(),
                h_next.data_ptr(),
                batch_size,
                dim,
            )
        else:
            h_np = h.detach().cpu().numpy().astype("float32")
            flow_np = flow.detach().cpu().numpy().astype("float32")
            h_next_np = diffusion_engine.step_cpu(h_np, flow_np)
            h_next = torch.from_numpy(h_next_np).to(h.device)
        ctx.alpha = float(alpha)
        ctx.dt = float(dt)
        return h_next

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        alpha = ctx.alpha
        dt = ctx.dt
        a = 1.0 - (1.0 - alpha) * dt
        b = (1.0 - alpha) * dt
        grad_h = grad_output * a
        grad_flow = grad_output * b
        return grad_h, grad_flow, None, None, None


class SentenceTopicHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_head: int = 64,
        num_topics: int = 8,
        num_heads: int = 4,
        c_poincare: float = 1e-3,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_topics = num_topics
        self.num_heads = num_heads
        self.d_head_per_head = d_head // num_heads
        self.c_poincare = c_poincare
        self.temperature = temperature
        self.poincare_embed = nn.Linear(d_model, d_head)
        self.metric_attn = MetricAttention(
            hidden_size=self.d_head_per_head,
            normalizer="softmax",
            rank=2,
            tau=self.temperature,
            mode="geodesic",
            manifold="poincare",
            c=self.c_poincare,
        )
        self.q_proj = nn.Linear(d_head, d_head)
        self.k_proj = nn.Linear(d_head, d_head)
        self.v_proj = nn.Linear(d_head, d_head)
        self.out_proj = nn.Linear(d_head, d_head)
        self.topic_classifier = nn.Linear(d_head, num_topics)
        self.topic_names = [
            "chief_complaint",
            "history",
            "physical_exam",
            "diagnosis",
            "treatment_plan",
            "prognosis",
            "follow_up",
            "general",
        ]

    def forward(
        self,
        x: torch.Tensor,
        topo_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        B, T, _ = x.shape
        z = self.poincare_embed(x)
        z = project_to_ball(z)
        H = self.num_heads
        d_h = self.d_head_per_head
        q = self.q_proj(z).view(B, T, H, d_h).transpose(1, 2)
        k = self.k_proj(z).view(B, T, H, d_h).transpose(1, 2)
        v = self.v_proj(z).view(B, T, H, d_h).transpose(1, 2)
        topo_dict = {"neighbor": topo_idx}
        topk_cfg = {"neighbor": topo_idx.shape[-1]}
        attn_out = self.metric_attn(
            q,
            k,
            v,
            topo_idx=topo_dict,
            topk_cfg=topk_cfg,
            c=self.c_poincare,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_head)
        attn_out = self.out_proj(attn_out)

        logits = self.topic_classifier(attn_out)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        C = logits.size(-1)
        P_topic = F.softmax(logits, dim=-1)
        P_topic = torch.where(torch.isfinite(P_topic), P_topic, torch.full_like(P_topic, 1.0 / max(C, 1)))
        P_topic = P_topic + 1e-8
        P_topic = P_topic / P_topic.sum(dim=-1, keepdim=True)

        scores, _ = logits.max(dim=-1)
        scores = torch.clamp(scores, min=-10.0, max=10.0)

        metric_keys: List[str] = []
        with torch.no_grad():
            for b in range(B):
                for t in range(T):
                    top_topic = int(P_topic[b, t].argmax().item())
                    topic_name = self.topic_names[top_topic] if 0 <= top_topic < len(self.topic_names) else "general"
                    score_val = float(scores[b, t].item())
                    if score_val > 1.0:
                        priority = "high"
                    elif score_val > 0.0:
                        priority = "medium"
                    else:
                        priority = "low"
                    metric_keys.append(f"topic:{topic_name}|priority:{priority}")

        return P_topic, scores, metric_keys


try:
    import reality_stone.metrikey as _metrikey_probe  # type: ignore
    HAS_METRIKEY = True
except Exception:
    HAS_METRIKEY = False


class MetricContextRouter(nn.Module):
    def __init__(
        self,
        d_head: int = 64,
        lambda_min: float = 0.5,
        lambda_max: float = 2.0,
        cache_size: int = 1000,  
        score_quantize: float = 0.1,
        spd_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.cache_size = cache_size
        self.score_quantize = score_quantize
        self.spd_eps = spd_eps
        from collections import OrderedDict
        self._cache: OrderedDict[Tuple[str, float, str], torch.Tensor] = OrderedDict()

        try:
            import reality_stone.metrikey as metrikey  # type: ignore
            self._metrikey = metrikey
            self._has_metrikey = True
        except Exception:
            self._metrikey = None
            self._has_metrikey = False

        self._metrikey = None
        self._has_metrikey = False
        
        self.metric_adjustment = nn.Parameter(torch.zeros(d_head, d_head))

    def _clamp_eigen(self, G: torch.Tensor) -> torch.Tensor:
        G_sym = (G + G.transpose(-2, -1)) / 2.0
        G_sym = G_sym + torch.eye(G.shape[-1], device=G.device, dtype=G.dtype) * self.spd_eps
        
        eigvals, eigvecs = torch.linalg.eigh(G_sym)
        eigvals = torch.clamp(eigvals, self.lambda_min, self.lambda_max)
        result = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
        
        result = (result + result.transpose(-2, -1)) / 2.0
        return result

    def _make_metric(self, key: str, score_q: float, device: torch.device) -> torch.Tensor:
        cache_key = (key, score_q, str(device))
        
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        if self._has_metrikey:
            try:
                G = self._metrikey.metric_from_keys(
                    [key],
                    dim=self.d_head,
                    min_lambda=self.lambda_min,
                    max_lambda=self.lambda_max,
                    masses=[score_q],
                )
                G = G.to(device)
            except Exception:
                scale = 1.0 + score_q * 0.1
                G = torch.eye(self.d_head, device=device) * scale
        else:
            scale = 1.0 + score_q * 0.1
            G = torch.eye(self.d_head, device=device) * scale

        G = self._clamp_eigen(G)
        G_reg = G + torch.eye(self.d_head, device=device) * self.spd_eps
        L = torch.linalg.cholesky(G_reg)

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        
        self._cache[cache_key] = L
        return L

    def forward(self, metric_keys: List[str], scores: torch.Tensor) -> torch.Tensor:
        B, T = scores.shape
        device = scores.device
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        if self.score_quantize is not None and self.score_quantize > 0:
            q = torch.as_tensor(self.score_quantize, dtype=scores.dtype, device=device)
            scores = torch.round(scores / q) * q
        
        eye_base = torch.eye(self.d_head, device=device, dtype=scores.dtype)
        
        scores_norm = torch.tanh(scores / 10.0)
        scale = 1.0 + 0.2 * scores_norm
        
        adjustment_sym = (self.metric_adjustment + self.metric_adjustment.t()) / 2.0
        adjustment_scale = 0.1 * torch.tanh(adjustment_sym)
        
        L_list = []
        for b in range(B):
            for t in range(T):
                s = scale[b, t]
                L_bt = eye_base * s + adjustment_scale
                L_bt = L_bt + eye_base * self.spd_eps
                L_list.append(L_bt)
        
        L_stacked = torch.stack(L_list, dim=0)
        L_adjusted = L_stacked.view(B, T, self.d_head, self.d_head)
        
        return L_adjusted


def _spd_log_euclidean_mean(
    spd_matrices: torch.Tensor, 
    weights: torch.Tensor,
    eps: float = 1e-5,
    eigval_min: float = 1e-5,
    eigval_max: float = 1e5,
    log_clip: float = 10.0,
) -> torch.Tensor:
    B, N, d, _ = spd_matrices.shape
    device = spd_matrices.device
    dtype = spd_matrices.dtype
    
    eps_eye = torch.eye(d, device=device, dtype=dtype) * eps
    spd_matrices = spd_matrices + eps_eye.view(1, 1, d, d)
    
    spd_flat = spd_matrices.reshape(B * N, d, d)
    eigvals, eigvecs = torch.linalg.eigh(spd_flat)
    eigvals = eigvals.clamp(min=eigval_min, max=eigval_max)
    log_eigvals = torch.log(eigvals)
    
    log_matrices_flat = torch.bmm(
        torch.bmm(eigvecs, torch.diag_embed(log_eigvals)),
        eigvecs.transpose(-2, -1)
    )
    log_matrices = log_matrices_flat.reshape(B, N, d, d)
    
    w = weights.view(B, N, 1, 1)
    log_mean = (w * log_matrices).sum(dim=1)
    
    eigvals_mean, eigvecs_mean = torch.linalg.eigh(log_mean)
    eigvals_mean = eigvals_mean.clamp(min=-log_clip, max=log_clip)
    exp_eigvals = torch.exp(eigvals_mean)
    exp_eigvals = exp_eigvals.clamp(min=eigval_min, max=eigval_max)
    
    result = torch.bmm(
        torch.bmm(eigvecs_mean, torch.diag_embed(exp_eigvals)),
        eigvecs_mean.transpose(-2, -1)
    )
    
    result = (result + result.transpose(-2, -1)) / 2.0
    result = result + eps_eye
    
    return result


class SPDMetricMixer(nn.Module):
    def __init__(
        self,
        d_head: int,
        gamma_up: float = 0.3,
        gamma_self: float = 0.5,
        gamma_down: float = 0.2,
        use_fast_mixing: bool = True,
        spd_eps: float = 1e-5,
        spd_eigval_min: float = 1e-5,
        spd_eigval_max: float = 1e5,
        spd_log_eigval_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.use_fast_mixing = use_fast_mixing
        self.spd_eps = spd_eps
        self.spd_eigval_min = spd_eigval_min
        self.spd_eigval_max = spd_eigval_max
        self.spd_log_eigval_clip = spd_log_eigval_clip
        self.gamma_up = nn.Parameter(torch.tensor(gamma_up))
        self.gamma_self = nn.Parameter(torch.tensor(gamma_self))
        self.gamma_down = nn.Parameter(torch.tensor(gamma_down))

    def mix_hierarchy(
        self,
        parent_metric: torch.Tensor,
        self_metric: torch.Tensor,
        children_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, d, _ = self_metric.shape
        
        gamma_up = torch.abs(self.gamma_up) + 1e-6
        gamma_self = torch.abs(self.gamma_self) + 1e-6
        gamma_down = torch.abs(self.gamma_down) + 1e-6
        
        mats = [parent_metric, self_metric]
        ws_raw = [gamma_up, gamma_self]

        if children_metrics is not None and children_metrics.size(1) > 0:
            child_mean = children_metrics.mean(dim=1)
            mats.append(child_mean)
            ws_raw.append(gamma_down)

        ws_tensor = torch.stack(ws_raw)
        ws_norm = F.softmax(ws_tensor, dim=0)
        
        if self.use_fast_mixing:
            mats_tensor = torch.stack(mats, dim=1)
            result = (ws_norm.view(1, -1, 1, 1) * mats_tensor).sum(dim=1)
            return result
        else:
            mats_tensor = torch.stack(mats, dim=1)
            w_expanded = ws_norm.view(1, -1).expand(B, -1)
            return _spd_log_euclidean_mean(
                mats_tensor, 
                w_expanded,
                eps=self.spd_eps,
                eigval_min=self.spd_eigval_min,
                eigval_max=self.spd_eigval_max,
                log_clip=self.spd_log_eigval_clip,
            )


class RCELexicalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layer: int = 2,
        n_head: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
        replacement_mask: Optional[torch.Tensor] = None,
        topo_idx: Optional[torch.Tensor] = None,
        candidates: Optional[Dict[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids.clamp(min=0, max=self.vocab_size - 1))
        logits = self.lm_head(x)
        if replacement_mask is None:
            replacement_mask = torch.ones_like(input_ids)
        if candidates is None:
            candidates = {}
        output_ids = input_ids.clone()
        for b in range(B):
            for t in range(T):
                if int(replacement_mask[b, t].item()) == 0:
                    continue
                tok = int(input_ids[b, t].item())
                cand = candidates.get(tok)
                if not cand:
                    cand = [tok]
                chosen = int(cand[0])
                output_ids[b, t] = chosen
        return output_ids.to(device), logits


class HierarchicalLMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        n_layer: int = 6,
        n_head: int = 8,
        manifold: str = "lorentz",
        c_lorentz: float = -1.0,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.manifold = manifold
        self.c_lorentz = c_lorentz

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [self._make_block() for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def _make_block(self) -> nn.Module:
        return _DecoderBlock(self.d_model, self.n_head, self.manifold, self.c_lorentz)

    def forward(
        self,
        input_ids: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
        topo_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S = input_ids.shape
        device = input_ids.device

        tok = self.token_embed(input_ids)
        max_pos = self.pos_embed.num_embeddings
        pos_ids = torch.arange(S, device=device).clamp(max=max_pos - 1).unsqueeze(0).expand(B, -1)
        pos = self.pos_embed(pos_ids)

        h = tok + pos
        m_ctx = metric_ctx
        topo = topo_idx
        for blk in self.blocks:
            h = blk(h, m_ctx, topo)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, h


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, manifold: str, c: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.manifold = manifold
        self.c = c
        d_h = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.lambda_p = nn.Parameter(torch.tensor(0.5))
        self.lambda_l = nn.Parameter(torch.tensor(0.5))
        # geodesic product attention 용 MetricAttention (SPDMetric만 재사용)
        self.attn = MetricAttention(
            hidden_size=d_h,
            normalizer="softmax",
            rank=2,
            tau=1.0,
            mode="geodesic",  # 점수는 아래에서 geodesic 으로 직접 계산
            manifold=manifold,
            c=abs(float(c)) if c is not None else 1e-3,
        )
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        metric_ctx: Optional[torch.Tensor],
        topo_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H = self.n_head
        d_h = self.d_model // H
        q = self.q_proj(x).view(B, S, H, d_h).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, d_h).transpose(1, 2)
        if metric_ctx is not None:
            d_ctx = metric_ctx.size(-1)
            if d_ctx == d_h:
                q_perm = q.transpose(1, 2)
                k_perm = k.transpose(1, 2)
                q_perm = torch.einsum("bsij,bshj->bshi", metric_ctx, q_perm)
                k_perm = torch.einsum("bsij,bshj->bshi", metric_ctx, k_perm)
                q = q_perm.transpose(1, 2)
                k = k_perm.transpose(1, 2)
            elif d_ctx < d_h:
                q_perm = q.transpose(1, 2)
                k_perm = k.transpose(1, 2)
                q_sub = q_perm[..., :d_ctx]
                k_sub = k_perm[..., :d_ctx]
                q_sub = torch.einsum("bsij,bshj->bshi", metric_ctx, q_sub)
                k_sub = torch.einsum("bsij,bshj->bshi", metric_ctx, k_sub)
                q_perm = torch.cat([q_sub, q_perm[..., d_ctx:]], dim=-1)
                k_perm = torch.cat([k_sub, k_perm[..., d_ctx:]], dim=-1)
                q = q_perm.transpose(1, 2)
                k = k_perm.transpose(1, 2)
        device = x.device
        if topo_idx is not None:
            idx = topo_idx
        else:
            idx = torch.arange(S, device=device).view(1, 1, S).expand(B, S, S)
        K = idx.shape[-1]
        arange_s = torch.arange(S, device=device).view(1, S, 1)
        idx_causal = torch.where(idx > arange_s, arange_s.expand_as(idx), idx)
        topo_dict = {"neighbor": idx_causal}
        topk_cfg = {"neighbor": K}
        c_used = abs(float(self.c)) if self.c is not None else 1e-3
        y = self.attn(
            q,
            k,
            v,
            topo_idx=topo_dict,
            topk_cfg=topk_cfg,
            c=c_used,
        )
        gate = torch.sigmoid(self.lambda_p) + 0.1 * torch.sigmoid(self.lambda_l)
        y = y * gate
        y = y.transpose(1, 2).contiguous().view(B, S, self.d_model)
        y = self.out_proj(y)
        x = x + y
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class HierarchicalSentenceTopicLLM(nn.Module):
    def __init__(self, config: HierarchicalLLMConfig) -> None:
        super().__init__()
        self.config = config

        # SFE Variable Suppression Field
        self.suppression_field = HyperbolicSuppressionField(
            base=getattr(config, "suppression_base", 0.37),
            linear=getattr(config, "suppression_linear", 0.0),
            hyp=getattr(config, "suppression_hyp", 0.1),
            scale=getattr(config, "suppression_scale", 1.0)
        ) if getattr(config, "enable_variable_suppression", False) else None

        # L0: Riemannian Aggregation (bottom-up encoding)
        self.sentence_aggregator = RiemannianAggregation(
            d_model=config.d_model,
            manifold=config.manifold_sentence,
            c=config.c_poincare,
            temperature=config.temperature_agg,
        )
        
        self.paragraph_aggregator = RiemannianAggregation(
            d_model=config.d_model,
            manifold=config.manifold_paragraph,
            c=config.c_poincare,
            temperature=config.temperature_agg,
        )

        # 문단 레벨 컨트롤러: 문단 임베딩 → 발화할 문장 수 분포
        self.paragraph_length_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.max_answer_sentences),
        )

        # L1: SentenceTopicHead (Poincaré + MetricAttention)
        self.topic_head = SentenceTopicHead(
            d_model=config.d_model,
            d_head=config.d_head,
            num_topics=config.num_topics,
            num_heads=config.num_heads_topic,
            c_poincare=config.c_poincare,
        )

        # L2: MetricContextRouter (MetriKey 기반 SPD metric slots)
        self.metric_router = MetricContextRouter(
            d_head=config.d_head,
            lambda_min=config.metric_lambda_min,
            lambda_max=config.metric_lambda_max,
            spd_eps=config.spd_eps,
        )
        
        # L2.5: SPD Metric Mixer (barycenter-based mixing)
        use_fast_mixing = getattr(config, "use_fast_spd_mixing", True)
        self.metric_mixer = SPDMetricMixer(
            d_head=config.d_head,
            gamma_up=config.gamma_up,
            gamma_self=config.gamma_self,
            gamma_down=config.gamma_down,
            use_fast_mixing=use_fast_mixing,
            spd_eps=config.spd_eps,
            spd_eigval_min=config.spd_eigval_min,
            spd_eigval_max=config.spd_eigval_max,
            spd_log_eigval_clip=config.spd_log_eigval_clip,
        )

        if config.use_pretrained_embeddings:
            self.backbone = PretrainedBackbone(
                model_name="klue/bert-base",
                freeze=config.freeze_decoder,
                d_model=config.d_model
            )
            self.token_embed = self.backbone
            config.vocab_size = self.backbone.get_vocab_size()
        else:
            self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # L3: HierarchicalLMDecoder (geodesic MetricAttention, 순수 LM)
        self.decoder = HierarchicalLMDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layer=config.n_layer_decoder,
            n_head=config.n_head_decoder,
            manifold="lorentz",
            c_lorentz=config.c_lorentz,
        )
        engine = None
        if getattr(config, "use_diffusion_hidden", False) and getattr(config, "diffusion_steps", 0) > 0:
            engine_cls = getattr(rs, "PyRiemannianDiffusion", None)
            if engine_cls is not None:
                try:
                    engine = engine_cls(config.d_model, config.diffusion_alpha, config.diffusion_dt)
                except Exception:
                    engine = None
        self.diffusion_engine = engine
        
        # Decoder와 Embedding 공유 + Weight Tying
        self.decoder.token_embed = self.token_embed
        if hasattr(self.token_embed, "weight"):
            self.decoder.lm_head.weight = self.token_embed.weight
        self.semantic_loss = SemanticPreservationLoss(
            manifold=config.manifold_sentence,
            c=config.c_poincare,
        )
        self.edit_head = EditOperationHead(config.d_model, num_ops=5, edit_budget=config.edit_budget)
        self.sentence_order_head = SentenceOrderHead(config.d_model)
        
        enable_dynamic_manifold = getattr(config, "enable_dynamic_manifold", False)
        self.tree_processor = LevelInvariantTreeProcessor(config.d_model, enable_dynamic_manifold)
        
        # Freeze backbone if specified (문서 7.1절: pretrain 후 거의 고정)
        # 현재는 pretrain이 없으므로 freeze하지 않음
        if config.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            print("[Init] Decoder frozen (requires pretrained weights)")
        
        if config.freeze_topic_head_backbone:
            # Freeze all except metric-related parameters
            for name, param in self.topic_head.named_parameters():
                if "metric" not in name.lower() and "spd" not in name.lower():
                    param.requires_grad = False
            print("[Init] TopicHead backbone frozen (requires pretrained weights)")

        if config.pretrained_decoder_path:
            state = torch.load(config.pretrained_decoder_path)
            self.decoder.load_state_dict(state['decoder'])
            if config.freeze_decoder:
                for p in self.decoder.parameters():
                    p.requires_grad = False

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> "HierarchicalSentenceTopicLLM":
        """
        학습 checkpoint dict 로부터 모델을 재구성하는 helper.

        checkpoint 형식:
            {
                "config": {...},          # 기존 train config dict
                "topic_head": state_dict,
                "decoder": state_dict,
                ...
            }
        """
        cfg_dict = checkpoint["config"]
        cfg = HierarchicalLLMConfig(
            vocab_size=cfg_dict["vocab_size"],
            d_model=cfg_dict["d_model"],
            d_head=cfg_dict["d_head"],
            num_topics=cfg_dict["num_topics"],
            num_heads_topic=cfg_dict["num_heads"],
            n_layer_decoder=cfg_dict["n_layer"],
            n_head_decoder=cfg_dict["n_head"],
        )
        model = cls(cfg)
        model.topic_head.load_state_dict(checkpoint["topic_head"])
        model.decoder.load_state_dict(checkpoint["decoder"])
        return model

    def encode_tokens_to_sentences(
        self,
        tokens: torch.Tensor,  # [B, T, L]
        metric_ctx_sentence: Optional[torch.Tensor] = None,  # [B, T, d, d]
    ) -> torch.Tensor:
        """
        토큰 → 문장 상향식 인코딩 (Riemannian message passing).
        
        h_sentence = RiemannAgg({h_token : token ∈ sentence}; M_sentence, G_sentence)
        
        Args:
            tokens: [B, T, L] 토큰 ID 텐서
            metric_ctx_sentence: [B, T, d, d] 문장별 SPD 메트릭 (optional)
            
        Returns:
            sentence_embeddings: [B, T, d_model]
        """
        B, T, L = tokens.shape
        
        # 토큰 임베딩 (Decoder와 공유)
        # CUDA assert 방지: 음수 및 범위 밖 값 제거
        tokens_clamped = tokens.clamp(min=0, max=self.config.vocab_size - 1)  # [B, T, L]
        
        # PretrainedBackbone은 [B*T, L]로 reshape 필요
        if isinstance(self.token_embed, PretrainedBackbone):
            tokens_flat_input = tokens_clamped.view(B * T, L)  # [B*T, L]
            token_embeddings_flat = self.token_embed(tokens_flat_input)  # [B*T, L, d_model]
            token_embeddings = token_embeddings_flat.view(B, T, L, self.config.d_model)  # [B, T, L, d_model]
        else:
            token_embeddings = self.token_embed(tokens_clamped)  # [B, T, L, d_model]
        
        # 문장별로 토큰들을 Riemannian aggregation
        # 배치 연산으로 최적화: [B, T, L, d_model] -> [B*T, L, d_model]
        BT = B * T
        tokens_flat = token_embeddings.reshape(BT, L, self.config.d_model)  # [B*T, L, d_model]
        
        if metric_ctx_sentence is not None:
            # [B, T, d, d] -> [B*T, d, d]
            metric_flat = metric_ctx_sentence.reshape(BT, metric_ctx_sentence.size(-2), metric_ctx_sentence.size(-1))
        else:
            metric_flat = None
        
        # 한번에 aggregation
        # SFE: Dynamic Temperature 적용
        # 억압장이 강할수록(epsilon↑) -> 유효 질량 감소(m_eff↓) -> 온도 증가(T_eff↑) -> 분포가 평평해짐 (Smoothing)
        # 반대로 epsilon이 작으면 -> T_eff 감소 -> 분포가 뾰족해짐 (Sharpening/Focusing)
        
        metric_ctx_reshaped = metric_flat # [BT, d, d] or None
        
        # 토큰들의 Norm을 억압장의 입력으로 사용 (원점에서의 거리 = 정보량/깊이)
        # tokens_flat: [BT, L, d_model]
        token_norms = tokens_flat.norm(dim=-1).mean(dim=-1, keepdim=True) # [BT, 1]
        
        current_temp = self.config.temperature_agg
        temperature_override = None
        if self.suppression_field is not None:
            dynamic_temp = self.suppression_field.compute_effective_temperature(
                t0=current_temp,
                x=token_norms
            )  # [BT, 1]
            temperature_override = dynamic_temp.mean()  # Tensor (keeps grad path)

        sentence_embeddings_flat = self.sentence_aggregator(
            tokens_flat,  # [B*T, L, d_model]
            metric_ctx=metric_flat,
            temperature_override=temperature_override,
        )  # [B*T, d_model]
        
        sentence_embeddings = sentence_embeddings_flat.reshape(B, T, self.config.d_model)  # [B, T, d_model]
        return sentence_embeddings
    
    def encode_sentences_to_paragraph(
        self,
        sentence_embeddings: torch.Tensor,  # [B, T, d_model]
        metric_ctx_paragraph: Optional[torch.Tensor] = None,  # [B, d, d]
    ) -> torch.Tensor:
        """
        문장 → 문단 상향식 인코딩 (Riemannian message passing).
        """
        
        # SFE: Dynamic Temperature 적용 (Paragraph Level)
        sent_norms = sentence_embeddings.norm(dim=-1).mean(dim=-1, keepdim=True) # [B, 1]
        
        current_temp = self.config.temperature_agg
        temperature_override = None
        if self.suppression_field is not None:
            dynamic_temp = self.suppression_field.compute_effective_temperature(
                t0=current_temp,
                x=sent_norms
            )
            temperature_override = dynamic_temp.mean()

        # RiemannAgg
        paragraph_embedding = self.paragraph_aggregator(
            sentence_embeddings,  # [B, T, d_model]
            metric_ctx=metric_ctx_paragraph,
            temperature_override=temperature_override,
        )  # [B, d_model]
        return paragraph_embedding

    def encode_sentences(
        self,
        tokens: torch.Tensor,  # [B, T, L]
        metric_ctx_sentence: Optional[torch.Tensor] = None,  # [B, T, d_h, d_h]
    ) -> torch.Tensor:
        """
        호환성 helper:
        - 기존 QA/인덱싱 유틸에서 사용하던 encode_sentences(tokens)를
          현재 구현의 encode_tokens_to_sentences로 연결한다.

        Args:
            tokens: [B, T, L] 토큰 ID 텐서
            metric_ctx_sentence: [B, T, d_h, d_h] 문장별 SPD 메트릭 (선택)

        Returns:
            sentence_embeddings: [B, T, d_model]
        """
        return self.encode_tokens_to_sentences(
            tokens,
            metric_ctx_sentence=metric_ctx_sentence,
        )
    
    def _encode_with_tree_processor(
        self,
        tokens: torch.Tensor,
        trees: List[DocumentTree],
        direction: str = "up",
        metric_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, L = tokens.shape
        device = tokens.device
        
        sentence_embeddings_list = []
        
        for b in range(B):
            tree = trees[b] if b < len(trees) else None
            if tree is None:
                sent_emb = self.encode_tokens_to_sentences(tokens[b:b+1])
                sentence_embeddings_list.append(sent_emb[0])
                continue
            
            node_embeddings: Dict[int, torch.Tensor] = {}
            
            sentence_nodes = [n for n in tree.nodes if n.type == "sentence"]
            if len(sentence_nodes) > T:
                sentence_nodes = sentence_nodes[:T]
            
            for sent_idx, sent_node in enumerate(sentence_nodes):
                if sent_idx >= T:
                    break
                tok_ids = tokens[b, sent_idx].clamp(0, self.config.vocab_size - 1)
                if tok_ids.dim() == 1:
                    tok_ids = tok_ids.unsqueeze(0)
                token_embs = self.token_embed(tok_ids)
                
                if metric_ctx is not None:
                    sent_metric = metric_ctx[b, sent_idx].unsqueeze(0)
                else:
                    sent_metric = None
                sent_emb = self.sentence_aggregator(token_embs, metric_ctx=sent_metric)
                
                if sent_emb.dim() == 1:
                    sent_emb = sent_emb.unsqueeze(0)
                
                node_embeddings[sent_node.id] = sent_emb.squeeze(0)
            
            if direction == "up":
                processed_embs = self.tree_processor.process_tree(
                    tree,
                    node_embeddings,
                    direction="up",
                )
                
                sent_embs_batch = []
                for sent_node in sentence_nodes[:T]:
                    if sent_node.id in processed_embs:
                        sent_embs_batch.append(processed_embs[sent_node.id])
                    elif sent_node.id in node_embeddings:
                        sent_embs_batch.append(node_embeddings[sent_node.id])
                    else:
                        sent_embs_batch.append(torch.zeros(self.config.d_model, device=device))
                
                while len(sent_embs_batch) < T:
                    sent_embs_batch.append(torch.zeros(self.config.d_model, device=device))
                
                sentence_embeddings_list.append(torch.stack(sent_embs_batch[:T]))
            else:
                sent_embs_batch = []
                for sent_node in sentence_nodes[:T]:
                    if sent_node.id in node_embeddings:
                        sent_embs_batch.append(node_embeddings[sent_node.id])
                    else:
                        sent_embs_batch.append(torch.zeros(self.config.d_model, device=device))
                
                while len(sent_embs_batch) < T:
                    sent_embs_batch.append(torch.zeros(self.config.d_model, device=device))
                
                sentence_embeddings_list.append(torch.stack(sent_embs_batch[:T]))
        
        return torch.stack(sentence_embeddings_list, dim=0)


    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        compute_loss: bool = True,
        use_tree_processing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        한 배치에 대해 전체 L1–L3 파이프라인을 통과시키고,
        (원하면) 토픽 + LM loss 를 함께 반환한다.

        Args:
            batch:
                - "tokens": [B, T, L]
                - "topo_idx": [B, T, K]
                - "tree": Optional[List[DocumentTree]] - 트리 구조 (배치별)
            compute_loss: 손실 계산 여부
            use_tree_processing: 트리 프로세서 사용 여부

        Returns:
            logits: [B, S, V] (토큰 시퀀스에 대한 다음 토큰 분포, S = T*L 또는 LM 시퀀스 길이)
            info: {
                "P_topic": [B, T, C],
                "scores": [B, T],
                "metric_keys": List[str],
                "metric_ctx": [B, T, d_h, d_h],
                "logits": [B, T, V],
                "hidden": [B, T, d_model],
                (옵션) "loss", "loss_lm", "loss_consistency", "loss_diversity"
            }
        """
        tokens = batch["tokens"]          # [B, T, L]
        topo_idx = batch["topo_idx"]      # [B, T, K]
        trees = batch.get("tree", None)   # Optional[List[DocumentTree]]

        device = next(self.parameters()).device
        tokens = tokens.to(device)
        topo_idx = topo_idx.to(device)

        B, T, L = tokens.shape

        # ========== 상향식 인코딩 (Bottom-up) ==========
        # Step 1: 토큰 → 문장 (Riemannian message passing)
        if use_tree_processing and trees is not None and len(trees) > 0:
            sentence_embeddings_raw = self._encode_with_tree_processor(tokens, trees, direction="up")  # [B, T, d_model]
        else:
            sentence_embeddings_raw = self.encode_tokens_to_sentences(tokens)  # [B, T, d_model]
        
        # Step 2: 문장 → 주제/메트릭 키 (SentenceTopicHead)
        P_topic, scores, metric_keys = self.topic_head(sentence_embeddings_raw, topo_idx)
        C = P_topic.size(-1)

        # 문단 내 consistency: KL(P_topic || paragraph_mean)
        paragraph_mean = P_topic.mean(dim=1, keepdim=True).detach()  # [B,1,C]
        paragraph_mean = paragraph_mean + 1e-8
        paragraph_mean = paragraph_mean / paragraph_mean.sum(dim=-1, keepdim=True)
        paragraph_mean = paragraph_mean.expand(-1, T, -1)
        
        log_p = torch.log(P_topic + 1e-8)
        loss_consistency = nn.KLDivLoss(reduction="batchmean")(log_p, paragraph_mean)
        loss_consistency = torch.clamp(loss_consistency, min=0.0, max=10.0)
        
        # 배치 전체 diversity: KL(batch_mean || uniform)
        batch_mean = P_topic.mean(dim=(0, 1))  # [C]
        batch_mean = batch_mean + 1e-8
        batch_mean = batch_mean / batch_mean.sum()
        uniform = torch.full_like(batch_mean, 1.0 / C)
        
        log_batch = torch.log(batch_mean + 1e-8)
        loss_diversity = nn.KLDivLoss(reduction="batchmean")(log_batch, uniform)
        loss_diversity = torch.clamp(loss_diversity, min=0.0, max=10.0)

        # Step 3: MetriKey → SPD 메트릭 (MetricContextRouter)
        metric_ctx_sentence = self.metric_router(metric_keys, scores)  # [B, T, d_h, d_h]
        
        # Step 4: 문장 → 문단 (Riemannian message passing with metric)
        # 문단 메트릭: 문장 메트릭들의 평균
        metric_ctx_paragraph = metric_ctx_sentence.mean(dim=1)  # [B, d_h, d_h]
        
        # 메트릭 적용하여 재인코딩
        if use_tree_processing and trees is not None and len(trees) > 0:
            sentence_embeddings = self._encode_with_tree_processor(
                tokens, trees, direction="up", metric_ctx=metric_ctx_sentence
            )  # [B, T, d_model]
        else:
            sentence_embeddings = self.encode_tokens_to_sentences(
                tokens,
                metric_ctx_sentence=metric_ctx_sentence,
            )  # [B, T, d_model]
        
        paragraph_embedding = self.encode_sentences_to_paragraph(
            sentence_embeddings,
            metric_ctx_paragraph=metric_ctx_paragraph,
        )  # [B, d_model]
        
        # 문단 임베딩 기반 문장 수 분포 (paragraph-level controller)
        length_logits = self.paragraph_length_head(paragraph_embedding)  # [B, max_answer_sentences]
        length_logits = torch.where(torch.isfinite(length_logits), length_logits, torch.zeros_like(length_logits))
        sentence_order_scores = self.sentence_order_head(sentence_embeddings)
        
        # Step 5: 상·하위 메트릭 혼합 (SPD barycenter)
        # parent_metric: [B, d_h, d_h] -> [B, 1, d_h, d_h] -> [B, T, d_h, d_h]
        parent_metric_expanded = metric_ctx_paragraph.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, d_h, d_h]

        # children_metrics: 각 문장의 이웃(시간/순서 기반)을 "자식"으로 간주하여
        # SPD 바리센터 혼합에 포함시킨다.
        # topo_idx: [B, T, K] 에 대해,
        #   children_metrics[b, t] = metric_ctx_sentence[b, topo_idx[b, t, :]]
        children_metrics: Optional[torch.Tensor]
        if topo_idx.numel() > 0 and metric_ctx_sentence.numel() > 0:
            B_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(topo_idx)  # [B, T, K]
            # 패딩된 topo_idx 가 T 범위를 벗어나지 않도록 클램프
            sent_idx = topo_idx.clamp(min=0, max=T - 1)
            children_metrics = metric_ctx_sentence[B_idx, sent_idx]  # [B, T, K, d_h, d_h]
            BT = B * T
            parent_flat = parent_metric_expanded.reshape(BT, self.config.d_head, self.config.d_head)
            self_flat = metric_ctx_sentence.reshape(BT, self.config.d_head, self.config.d_head)
            children_flat = children_metrics.reshape(
                BT,
                children_metrics.size(2),
                self.config.d_head,
                self.config.d_head,
            )  # [B*T, K, d_h, d_h]
            effective_flat = self.metric_mixer.mix_hierarchy(
                parent_metric=parent_flat,
                self_metric=self_flat,
                children_metrics=children_flat,
            )  # [B*T, d_h, d_h]
        else:
            # 안전 장치: children 이 없으면 parent/self 만 사용
            BT = B * T
            parent_flat = parent_metric_expanded.reshape(BT, self.config.d_head, self.config.d_head)
            self_flat = metric_ctx_sentence.reshape(BT, self.config.d_head, self.config.d_head)
            effective_flat = self.metric_mixer.mix_hierarchy(
                parent_metric=parent_flat,
                self_metric=self_flat,
                children_metrics=None,
            )  # [B*T, d_h, d_h]

        metric_ctx = effective_flat.reshape(B, T, self.config.d_head, self.config.d_head)  # [B, T, d_h, d_h]

        # ===== L3: HierarchicalLMDecoder (순수 LM, 토큰 시퀀스 전체를 학습) =====
        # 토큰/메트릭/토폴로지를 토큰 단위 시퀀스로 평탄화
        S_full = T * L
        tokens_flat = tokens.clamp(min=0, max=self.config.vocab_size - 1).view(B, S_full)  # [B, S_full]

        # 문장 메트릭을 토큰 수준으로 브로드캐스트
        metric_ctx_flat_full = (
            metric_ctx  # [B, T, d_h, d_h]
            .unsqueeze(2)  # [B, T, 1, d_h, d_h]
            .expand(B, T, L, self.config.d_head, self.config.d_head)
            .contiguous()
            .view(B, S_full, self.config.d_head, self.config.d_head)
        )  # [B, S_full, d_h, d_h]

        # topology index를 토큰 수준으로 변환
        # topo_idx: [B, T, K] - 문장 인덱스 (0..T-1)
        # 토큰 인덱스로 변환: sent_idx * L + token_offset
        # 각 문장의 첫 토큰 위치로 매핑 (간단한 근사)
        K = topo_idx.size(-1)
        topo_idx_token = topo_idx * L  # [B, T, K] - 각 문장의 시작 토큰 인덱스
        
        # 이를 토큰 수준으로 브로드캐스트
        topo_idx_flat_full = (
            topo_idx_token
            .unsqueeze(2)  # [B, T, 1, K]
            .expand(B, T, L, K)
            .contiguous()
            .view(B, S_full, K)
        )  # [B, S_full, K]
        
        # 각 토큰 위치에서 자신의 문장 내 offset을 더해 정확한 이웃 토큰 인덱스 생성
        token_offset = torch.arange(L, device=device).view(1, 1, L, 1).expand(B, T, L, K)
        token_offset_flat = token_offset.contiguous().view(B, S_full, K)
        topo_idx_flat_full = (topo_idx_flat_full + token_offset_flat).clamp(min=0, max=S_full - 1)

        # LM 시퀀스 길이 상한 적용 (메모리 보호)
        if S_full > self.config.max_lm_seq_len:
            S = self.config.max_lm_seq_len
            tokens_flat = tokens_flat[:, :S]
            metric_ctx_flat = metric_ctx_flat_full[:, :S]
            topo_idx_flat = topo_idx_flat_full[:, :S]
        else:
            S = S_full
            metric_ctx_flat = metric_ctx_flat_full
            topo_idx_flat = topo_idx_flat_full

        logits, hidden = self.decoder(
            input_ids=tokens_flat,
            metric_ctx=metric_ctx_flat,
            topo_idx=topo_idx_flat,
        )
        if getattr(self.config, "use_diffusion_hidden", False) and getattr(self.config, "diffusion_steps", 0) > 0 and getattr(self, "diffusion_engine", None) is not None:
            B_hidden, S_hidden, D_hidden = hidden.shape
            h_flat = hidden.reshape(B_hidden * S_hidden, D_hidden)
            for _ in range(self.config.diffusion_steps):
                flow = torch.tanh(h_flat)
                h_flat = RiemannianDiffusionStep.apply(
                    h_flat,
                    flow,
                    self.diffusion_engine,
                    self.config.diffusion_alpha,
                    self.config.diffusion_dt,
                )
            hidden = h_flat.view(B_hidden, S_hidden, D_hidden)
            logits = self.decoder.lm_head(hidden)
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        logits = torch.clamp(logits, min=-self.config.logit_clip_value, max=self.config.logit_clip_value)
        edit_logits = self.edit_head(hidden)

        info: Dict[str, torch.Tensor] = {
            "P_topic": P_topic,
            "scores": scores,
            "metric_ctx": metric_ctx,
            "logits": logits,
            "hidden": hidden,
            "edit_logits": edit_logits,
            "sentence_order_scores": sentence_order_scores,
        }
        info_str: Dict[str, object] = {
            **info,
            "metric_keys": metric_keys,
        }
        info_str["length_logits"] = length_logits
        info_str["paragraph_embedding"] = paragraph_embedding

        has_lm_target = True

        if compute_loss:
            # 문장 non-empty 마스크 (여러 loss에서 재사용)
            sentence_nonempty = (tokens > 0).any(dim=-1)  # [B, T]

            # 문장 수 예측 loss (문단 레벨)
            true_lengths = sentence_nonempty.sum(dim=1)   # [B]
            # 최소 1문장, 최대 max_answer_sentences 로 클램프 후 0-base 인덱스로 변환
            length_targets = true_lengths.clamp(
                min=1, max=self.config.max_answer_sentences
            ) - 1  # [B]
            length_loss = F.cross_entropy(length_logits, length_targets)

            # 선택적 토픽 supervision (batch 에 topic_labels 가 있을 때만 사용)
            topic_loss = None
            topic_labels = batch.get("topic_labels")
            if topic_labels is not None:
                topic_labels_t = topic_labels.to(device)  # [B, T]
                # 패딩 문장은 ignore_index(-1) 로 마스킹
                topic_targets = topic_labels_t.clone()
                topic_targets[~sentence_nonempty] = -1
                log_p_topic = (P_topic + 1e-10).log().view(B * T, C)
                topic_targets_flat = topic_targets.view(B * T)
                topic_loss = F.nll_loss(
                    log_p_topic,
                    topic_targets_flat,
                    ignore_index=-1,
                )

            semantic_mask = sentence_nonempty.to(sentence_embeddings_raw.dtype)
            semantic_loss = self.semantic_loss(
                sentence_embeddings_raw,
                sentence_embeddings,
                mask=semantic_mask,
            )

            # ===== 논문 설계: Next-token prediction (Autoregressive) =====
            # Decoder는 autoregressive하게 다음 토큰을 예측
            # input: tokens[:, :-1], target: tokens[:, 1:]
            
            S = tokens_flat.size(1)
            S_max = min(S, logits.size(1))
            
            if S_max > 1:
                logits_pred = logits[:, :S_max-1, :]
                targets = tokens_flat[:, 1:S_max].clamp(0, self.config.vocab_size - 1)
            else:
                logits_pred = logits[:, :0, :]
                targets = tokens_flat[:, :0].clamp(0, self.config.vocab_size - 1)
            
            V = logits.size(-1)
            targets_flat = targets.reshape(-1)
            valid_mask = targets_flat.ne(0)
            if valid_mask.any():
                logits_flat = logits_pred.reshape(-1, V)
                logits_flat_valid = logits_flat[valid_mask]
                targets_flat_valid = targets_flat[valid_mask]
                lm_loss = F.cross_entropy(
                    logits_flat_valid,
                    targets_flat_valid,
                )
                has_lm_target = True
            else:
                if logits_pred.numel() == 0 or targets.numel() == 0:
                    raise RuntimeError("No LM targets available; check tokenization and sequence lengths.")
                logits_flat = logits_pred.reshape(-1, V)
                targets_all = targets.reshape(-1)
                lm_loss = F.cross_entropy(
                    logits_flat,
                    targets_all,
                )
                has_lm_target = True

            loss_clip = self.config.loss_clip_max
            
            if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                raise RuntimeError("lm_loss is NaN or Inf; check dataset/tokenization and model configuration.")
            lm_loss = torch.clamp(lm_loss, min=0.0, max=loss_clip)
            
            if torch.isnan(loss_consistency) or torch.isinf(loss_consistency):
                raise RuntimeError("loss_consistency is NaN or Inf; check topic distributions.")
            loss_consistency = torch.clamp(loss_consistency, min=0.0, max=loss_clip * 0.1)
            
            if torch.isnan(loss_diversity) or torch.isinf(loss_diversity):
                raise RuntimeError("loss_diversity is NaN or Inf; check topic distributions.")
            loss_diversity = torch.clamp(loss_diversity, min=0.0, max=loss_clip * 0.1)
            
            if torch.isnan(length_loss) or torch.isinf(length_loss):
                raise RuntimeError("length_loss is NaN or Inf; check sentence_nonempty / length_logits.")
            length_loss = torch.clamp(length_loss, min=0.0, max=loss_clip * 0.1)
            
            if topic_loss is not None:
                if torch.isnan(topic_loss) or torch.isinf(topic_loss):
                    raise RuntimeError("topic_loss is NaN or Inf; check topic_labels and P_topic.")
                topic_loss = torch.clamp(topic_loss, min=0.0, max=loss_clip * 0.1)

            # Metric regularization: ||G - I||_F^2, G = L L^T
            d_h = self.config.d_head
            eye = torch.eye(d_h, device=device, dtype=metric_ctx.dtype)
            G_sentence = metric_ctx_sentence.reshape(B * T, d_h, d_h)
            G_sentence = G_sentence @ G_sentence.transpose(-2, -1)
            diff_G = G_sentence - eye
            loss_metric = (diff_G.pow(2).sum(dim=(-2, -1))).mean()

            # Curvature regularization (only meaningful when dynamic manifold/kappa is enabled)
            loss_curvature = torch.tensor(0.0, device=device, dtype=logits.dtype)
            if getattr(self.config, "enable_dynamic_manifold", False):
                tgt_p = torch.as_tensor(abs(float(self.config.curvature_target_poincare)), device=device, dtype=logits.dtype)
                tgt_l = torch.as_tensor(abs(float(self.config.curvature_target_lorentz)), device=device, dtype=logits.dtype)
                tgt_k = torch.as_tensor(abs(float(getattr(self.config, "curvature_target_klein", self.config.curvature_target_poincare))), device=device, dtype=logits.dtype)
                # Sum across node operators so kappa actually gets a gradient signal.
                reg_terms = []
                for op in self.tree_processor.node_operators.values():
                    if hasattr(op, "_curvatures"):
                        c_p, c_l, c_k = op._curvatures(device=device, dtype=logits.dtype)  # type: ignore[attr-defined]
                        reg_terms.append((c_p - tgt_p).pow(2))
                        reg_terms.append((c_l - tgt_l).pow(2))
                        reg_terms.append((c_k - tgt_k).pow(2))
                if reg_terms:
                    loss_curvature = torch.stack(reg_terms).mean()

            # 최종 loss 구성
            loss = (
                lm_loss
                + self.config.lambda_consistency * loss_consistency
                + self.config.lambda_diversity * loss_diversity
                + self.config.lambda_length * length_loss
            )
            if self.config.lambda_metric > 0.0:
                loss = loss + self.config.lambda_metric * loss_metric
                info_str["loss_metric"] = loss_metric
            if self.config.lambda_curvature > 0.0:
                loss = loss + self.config.lambda_curvature * loss_curvature
                info_str["loss_curvature"] = loss_curvature
            if topic_loss is not None and self.config.lambda_topic_supervision > 0.0:
                loss = loss + self.config.lambda_topic_supervision * topic_loss
                info_str["loss_topic"] = topic_loss
            if self.config.lambda_semantic > 0.0:
                semantic_loss = torch.clamp(semantic_loss, min=0.0, max=self.config.loss_clip_max * 0.1)
                loss = loss + self.config.lambda_semantic * semantic_loss
                info_str["loss_semantic"] = semantic_loss

            # Tiny regularizer to keep sentence_order_head in graph
            order_reg = (sentence_order_scores ** 2).mean() * 1e-6
            loss = loss + order_reg
            info_str["loss_sentence_order_reg"] = order_reg

            if self.config.lambda_edit > 0.0:
                num_ops = edit_logits.size(-1)
                probs_edit = F.softmax(edit_logits, dim=-1)
                cost_vec = torch.tensor(
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    device=probs_edit.device,
                    dtype=probs_edit.dtype,
                )
                expected_cost = (probs_edit * cost_vec.view(1, 1, num_ops)).sum(dim=-1)
                loss_edit = expected_cost.mean()
                loss = loss + self.config.lambda_edit * loss_edit
                info_str["loss_edit"] = loss_edit
            else:
                loss_edit_reg = (edit_logits ** 2).mean() * 1e-6
                loss = loss + loss_edit_reg
                info_str["loss_edit_reg"] = loss_edit_reg

            gamma_reg = (
                (self.metric_mixer.gamma_up ** 2)
                + (self.metric_mixer.gamma_self ** 2)
                + (self.metric_mixer.gamma_down ** 2)
            ) * 1e-6
            loss = loss + gamma_reg
            info_str["loss_gamma_reg"] = gamma_reg
            
            loss = torch.clamp(loss, min=0.0, max=self.config.loss_clip_max * 2.0)

            info_str["loss"] = loss
            info_str["loss_lm"] = lm_loss
            info_str["has_lm_target"] = has_lm_target
            info_str["loss_consistency"] = loss_consistency
            info_str["loss_diversity"] = loss_diversity
            info_str["loss_length"] = length_loss

        return logits, info_str  # type: ignore[return-value]


def train_hierarchical_llm_from_text(
    data_path: str,
    config: Optional[HierarchicalLLMConfig] = None,
    epochs: int = 50,
    batch_size: int = 4,
    max_paragraphs: int = 1000,
    device: Optional[str] = None,
    teacher_model=None,
    teacher_tokenizer=None,
    kd_proj: Optional[nn.Module] = None,
    kd_weight: float = 0.0,
) -> Tuple[HierarchicalSentenceTopicLLM, Dict[str, object]]:
    if config is None:
        config = HierarchicalLLMConfig()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    if not _HAS_SENTENCE_TOPIC_DATASET:
        raise RuntimeError(
            "SentenceTopicDataset/ collate_batch 가 로드되지 않았습니다. "
            "reality_stone.data 모듈이 제대로 설치되었는지 확인하세요."
        )

    use_kd = teacher_model is not None and teacher_tokenizer is not None and kd_proj is not None and kd_weight > 0.0

    # 모델 초기화
    model = HierarchicalSentenceTopicLLM(config).to(device_t)

    # 데이터셋/로더 구성
    dataset = SentenceTopicDataset(data_path, max_paragraphs=max_paragraphs)
    from torch.utils.data import DataLoader  # local import

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    # Optimizer: 메트릭 슬롯에 더 큰 LR, 백본에 작은 LR
    # (pretrain 없이는 백본도 함께 학습해야 함)
    
    # Metric-related parameters (high LR)
    metric_params = []
    for name, param in model.topic_head.named_parameters():
        if param.requires_grad and ("metric" in name.lower() or "spd" in name.lower()):
            metric_params.append(param)
    metric_params.extend(model.metric_router.parameters())
    metric_params.extend(model.metric_mixer.parameters())
    metric_params.extend(model.sentence_aggregator.parameters())
    metric_params.extend(model.paragraph_aggregator.parameters())
    if model.suppression_field is not None:
        metric_params.extend(model.suppression_field.parameters())
    # Include dynamic-manifold/tree curvature parameters (kappa, selector, etc.)
    metric_params.extend(model.tree_processor.parameters())
    
    # Backbone parameters (low LR)
    backbone_params = []
    for name, param in model.topic_head.named_parameters():
        if param.requires_grad and not ("metric" in name.lower() or "spd" in name.lower()):
            backbone_params.append(param)
    backbone_params.extend(model.decoder.parameters())
    if use_kd:
        backbone_params.extend(list(kd_proj.parameters()))
    
    # Filter only trainable
    metric_params = [p for p in metric_params if p.requires_grad]
    backbone_params = [p for p in backbone_params if p.requires_grad]
    
    if len(metric_params) == 0 and len(backbone_params) == 0:
        raise RuntimeError("No trainable parameters found.")
    
    print(f"[Training] Metric parameters: {sum(p.numel() for p in metric_params)} (LR={config.lr_metric})")
    print(f"[Training] Backbone parameters: {sum(p.numel() for p in backbone_params)} (LR={config.lr_backbone})")
    print(f"[Training] Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer with different LRs
    optimizer = torch.optim.AdamW([
        {'params': metric_params, 'lr': config.lr_metric},
        {'params': backbone_params, 'lr': config.lr_backbone},
    ])

    model.train()
    total_loss = 0.0

    from tqdm import tqdm  # local import

    base_lambda_consistency = config.lambda_consistency
    base_lambda_diversity = config.lambda_diversity
    
    for epoch in range(epochs):
        # 동적 lambda 계산
        lambda_consistency_current = compute_dynamic_lambda(
            base_lambda_consistency,
            config.lambda_consistency_schedule,
            epoch,
            epochs,
        )
        lambda_diversity_current = compute_dynamic_lambda(
            base_lambda_diversity,
            config.lambda_diversity_schedule,
            epoch,
            epochs,
        )
        
        # 모델의 lambda 업데이트 (forward에서 사용)
        model.config.lambda_consistency = lambda_consistency_current
        model.config.lambda_diversity = lambda_diversity_current
        
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Hierarchical LLM epoch {epoch+1}/{epochs}")
        for batch in pbar:
            try:
                optimizer.zero_grad()
                logits, info = model(batch, compute_loss=True)
                loss = info["loss"]  # type: ignore[index]
                assert isinstance(loss, torch.Tensor)
                if use_kd:
                    paragraphs = batch.get("paragraphs", None)
                    paragraph_emb = info.get("paragraph_embedding", None)
                    if paragraphs is not None and isinstance(paragraphs, list) and isinstance(paragraph_emb, torch.Tensor):
                        enc = teacher_tokenizer(paragraphs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                        for k in enc:
                            enc[k] = enc[k].to(device_t)
                        with torch.no_grad():
                            teacher_out = teacher_model(**enc)
                            if hasattr(teacher_out, "last_hidden_state"):
                                teacher_hidden = teacher_out.last_hidden_state[:, 0, :]
                            else:
                                teacher_hidden = teacher_out[0][:, 0, :]
                        teacher_proj = kd_proj(teacher_hidden)
                        teacher_proj = teacher_proj.to(paragraph_emb.dtype)
                        loss_kd = F.mse_loss(paragraph_emb, teacher_proj)
                        loss = loss + kd_weight * loss_kd
                loss.backward()
                torch.nn.utils.clip_grad_norm_(metric_params, config.grad_clip_norm)
                torch.nn.utils.clip_grad_norm_(backbone_params, config.grad_clip_norm)
                optimizer.step()
                epoch_loss += float(loss.item())
                pbar.set_postfix(
                    loss=f"{float(loss.item()):.4f}",
                    λ_cons=f"{lambda_consistency_current:.3f}",
                    λ_div=f"{lambda_diversity_current:.3f}",
                )
            except Exception as e:  # pragma: no cover - 안전 장치
                print(f"[train_hierarchical_llm_from_text] Error in batch: {e}")
                continue
        epoch_loss /= max(len(dataloader), 1)
        print(
            f"[Hierarchical LLM] epoch {epoch+1}/{epochs}, "
            f"loss={epoch_loss:.4f}, "
            f"λ_consistency={lambda_consistency_current:.3f}, "
            f"λ_diversity={lambda_diversity_current:.3f}"
        )
        total_loss = epoch_loss

    info_out: Dict[str, object] = {
        "final_loss": total_loss,
        "num_samples": len(dataset),
        "config": config,
    }
    return model, info_out


def _apply_top_down_decoding(
    model: HierarchicalSentenceTopicLLM,
    tree: DocumentTree,
    info: Dict[str, object],
    tokens: torch.Tensor,
    replacement_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    B, T, L = tokens.shape
    
    paragraph_nodes = [n for n in tree.nodes if n.type == "document"]
    sentence_nodes = [n for n in tree.nodes if n.type == "sentence"]
    
    if not paragraph_nodes:
        S = T * L
        input_ids_flat = tokens.clamp(0, model.config.vocab_size - 1).view(1, S)
        return input_ids_flat
    
    para_node = paragraph_nodes[0]
    
    hidden = info.get("hidden")
    if hidden is None or not isinstance(hidden, torch.Tensor):
        S = T * L
        input_ids_flat = tokens.clamp(0, model.config.vocab_size - 1).view(1, S)
        return input_ids_flat
    
    paragraph_embedding = hidden.mean(dim=1)
    
    node_embeddings: Dict[int, torch.Tensor] = {}
    node_embeddings[para_node.id] = paragraph_embedding[0]
    
    for sent_idx, sent_node in enumerate(sentence_nodes[:T]):
        if sent_idx >= T:
            break
        pos = min(sent_idx * L, hidden.size(1) - 1)
        node_embeddings[sent_node.id] = hidden[0, pos]
    
    processed_embs = model.tree_processor.process_tree(
        tree,
        node_embeddings,
        direction="down",
    )
    
    result_tokens = []
    for sent_idx, sent_node in enumerate(sentence_nodes[:T]):
        if sent_node.id in processed_embs:
            sent_emb = processed_embs[sent_node.id]
        elif sent_node.id in node_embeddings:
            sent_emb = node_embeddings[sent_node.id]
        else:
            sent_emb = torch.zeros(model.config.d_model, device=device)
        
        sent_emb_expanded = sent_emb.unsqueeze(0).unsqueeze(0)
        logits_sent = model.decoder.lm_head(sent_emb_expanded)
        pred_tokens_sent = torch.argmax(logits_sent, dim=-1)
        
        pred_tokens_sent = pred_tokens_sent.expand(1, L)
        
        original_tokens_sent = tokens[0, sent_idx].clamp(0, model.config.vocab_size - 1)
        replacement_mask_sent = replacement_mask[sent_idx].to(device)
        
        edited_tokens_sent = torch.where(
            replacement_mask_sent.bool(),
            pred_tokens_sent[0],
            original_tokens_sent,
        )
        result_tokens.append(edited_tokens_sent)
    
    result_flat = torch.cat(result_tokens, dim=0).unsqueeze(0)
    return result_flat


def infer_hierarchical_llm_on_text(
    model: HierarchicalSentenceTopicLLM,
    text: str,
    max_length: int = 128,
    k_neighbors: int = 3,
    max_new_tokens: int = 20,
    use_top_down: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.9,
    use_sampling: bool = True,
) -> Dict[str, object]:
    """
    PreSegmenter 를 사용해 단일 문단 텍스트에 대해
    계층적 Sentence-Topic LLM 의 추론을 수행하는 헬퍼 (생성 모드).

    Returns:
        {
            "original_text": str,
            "sentences": List[str],
            "generated_text": str,
            "topics": [...],
        }
    """
    from reality_stone.utils.pre_segmenter import PreSegmenter

    device = next(model.parameters()).device

    segmenter = PreSegmenter(max_length=max_length, k_neighbors=k_neighbors)
    seg_output = segmenter(text)

    if seg_output["metadata"]["num_sentences"] == 0:
        return {
            "original_text": text,
            "sentences": [],
            "generated_text": text,
            "topics": [],
        }

    tokens = seg_output["tokens"].unsqueeze(0).to(device)
    topo_idx = seg_output["topo_idx"].unsqueeze(0).to(device)
    tree = seg_output.get("tree")

    batch: Dict[str, torch.Tensor] = {
        "tokens": tokens,
        "topo_idx": topo_idx,
    }
    if tree is not None:
        batch["tree"] = [tree]

    model.eval()
    with torch.no_grad():
        logits, info = model(batch, compute_loss=False, use_tree_processing=use_top_down)

    original_sentences: List[str] = seg_output["sentences"]
    tokenizer = segmenter.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
    
    # T, L 변수 미리 정의 (inference에서 사용)
    B, T, L = tokens.shape

    if use_top_down and tree is not None:
        tokens_seq = _apply_top_down_decoding(
            model=model,
            tree=tree,
            info=info,
            tokens=tokens,
            replacement_mask=seg_output["replacement_mask"],
            device=device,
        )
        S_actual = tokens_seq.size(1)
    else:
        replacement_mask = seg_output["replacement_mask"].unsqueeze(0).to(device)
        B, T, L = tokens.shape
        S = T * L
        input_ids_flat = tokens.clamp(0, model.config.vocab_size - 1).view(1, S)
        mask_flat = replacement_mask.view(1, S)
        
        S_actual = logits.size(1)
        if S_actual < S:
            input_ids_flat = input_ids_flat[:, :S_actual]
            mask_flat = mask_flat[:, :S_actual]
        
        if use_sampling and temperature > 0:
            V = logits.size(-1)
            logits_scaled = logits / temperature
            probs = F.softmax(logits_scaled, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask_p = cumsum_probs > top_p
                mask_p[..., 0] = False
                sorted_probs = sorted_probs.clone()
                sorted_probs[mask_p] = 0.0
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                sampled_sorted_idx = torch.multinomial(
                    sorted_probs.reshape(-1, V),
                    num_samples=1,
                ).reshape(*probs.shape[:-1], 1)
                pred_ids_flat = sorted_indices.gather(-1, sampled_sorted_idx).squeeze(-1)
            else:
                pred_ids_flat = torch.multinomial(
                    probs.reshape(-1, V),
                    num_samples=1,
                ).reshape(*probs.shape[:-1])
        else:
            pred_ids_flat = torch.argmax(logits, dim=-1)
        
        edited_flat = torch.where(mask_flat.bool(), pred_ids_flat, input_ids_flat)
        tokens_seq = edited_flat

        if getattr(model.config, "enable_structural_edit", False):
            edit_logits = info.get("edit_logits")
            if isinstance(edit_logits, torch.Tensor):
                tokens_seq = model.edit_head.apply_edits(
                    tokens=tokens_seq,
                    edit_logits=edit_logits[:, :S_actual, :],
                    pred_tokens=pred_ids_flat[:, :S_actual],
                    enable_structural=True,
                    replacement_mask=mask_flat[:, :S_actual] if 'mask_flat' in locals() else None,
                )
                S_actual = tokens_seq.size(1)

    final_ids_flat = tokens_seq[0].tolist()
    if tokenizer is not None:
        try:
            if getattr(model.config, "enable_structural_edit", False):
                token_ids_no_pad = [tid for tid in final_ids_flat if tid != pad_id and tid > 0]
                if token_ids_no_pad:
                    generated_text = tokenizer.decode(token_ids_no_pad, skip_special_tokens=True)
                else:
                    generated_text = ""
            else:
                generated_sentences: List[str] = []
                for sent_idx in range(T):
                    start_idx = sent_idx * L
                    end_idx = min(start_idx + L, len(final_ids_flat))
                    sent_token_ids = final_ids_flat[start_idx:end_idx]
                    sent_token_ids_no_pad = [tid for tid in sent_token_ids if tid != pad_id and tid > 0]
                    if sent_token_ids_no_pad:
                        sent_text = tokenizer.decode(sent_token_ids_no_pad, skip_special_tokens=True)
                        if sent_text.strip():
                            generated_sentences.append(sent_text)
                if generated_sentences:
                    generated_text = " ".join(generated_sentences)
                else:
                    generated_text = ""
        except Exception as e:
            import traceback
            print(f"[WARNING] Tokenizer decode failed: {e}")
            print(traceback.format_exc())
            generated_text = ""
    else:
        generated_text = ""
    
    if not generated_text or generated_text == text:
        generated_text = text

    # 문단 레벨 컨트롤러가 예측한 문장 수에 맞게, 문장을 잘라서 상위 레벨에서 발화 길이를 제어
    length_logits_tensor = info.get("length_logits")
    if isinstance(length_logits_tensor, torch.Tensor) and len(generated_text) > 0:
        length_probs = torch.softmax(length_logits_tensor, dim=-1)
        pred_sentences = int(length_probs[0].argmax().item()) + 1
        pred_sentences = min(pred_sentences, model.config.max_answer_sentences)
        
        seg_generated = segmenter(generated_text)
        gen_sents = seg_generated.get("sentences", [])
        if gen_sents and len(gen_sents) > pred_sentences:
            order_scores = info.get("sentence_order_scores")
            if isinstance(order_scores, torch.Tensor) and not getattr(model.config, "enable_structural_edit", False):
                scores_np = order_scores[0, : len(gen_sents)].detach().cpu()
                indices = list(range(len(gen_sents)))
                indices.sort(key=lambda i: float(scores_np[i].item()), reverse=True)
                indices = indices[:pred_sentences]
                indices.sort()
                selected = [gen_sents[i] for i in indices]
                generated_text = " ".join(selected)
            else:
                generated_text = " ".join(gen_sents[:pred_sentences])

    P_topic = info.get("P_topic")
    metric_keys = info.get("metric_keys", [])

    topic_entries: List[Dict[str, object]] = []
    if isinstance(P_topic, torch.Tensor):
        topic_names = model.topic_head.topic_names
        for i, sent in enumerate(original_sentences):
            if i >= P_topic.size(1):
                break
            probs = P_topic[0, i]
            top_idx = int(probs.argmax().item())
            entry = {
                "sentence": sent,
                "topic": topic_names[top_idx],
                "confidence": float(probs[top_idx].item()),
                "metric_key": metric_keys[i] if i < len(metric_keys) else None,
            }
            topic_entries.append(entry)

    return {
        "original_text": text,
        "sentences": original_sentences,
        "generated_text": generated_text,
        "topics": topic_entries,
    }


def build_sentence_index_from_corpus(
    model: HierarchicalSentenceTopicLLM,
    data_path: str,
    max_paragraphs: int = 1000,
) -> List[Dict[str, object]]:
    if not _HAS_SENTENCE_TOPIC_DATASET:
        raise RuntimeError(
            "SentenceTopicDataset 이 로드되지 않았습니다. reality_stone.data 설치 상태를 확인하세요."
        )
    device = next(model.parameters()).device
    dataset = SentenceTopicDataset(data_path, max_paragraphs=max_paragraphs)
    index: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            tokens = sample["tokens"].unsqueeze(0).to(device)          # [1, T, L]
            topo_idx = sample["topo_idx"].unsqueeze(0).to(device)      # [1, T, K]
            sentences: List[str] = sample["sentences"]
            sent_emb = model.encode_sentences(tokens)                   # [1, T, d_model]
            z = model.topic_head.poincare_embed(sent_emb)               # [1, T, d_head]
            z = project_to_ball(z)                                      # ball projection
            P_topic, _, metric_keys = model.topic_head(sent_emb, topo_idx)
            T = len(sentences)
            for t in range(T):
                entry: Dict[str, object] = {
                    "paragraph": sample["paragraph"],
                    "sentence": sentences[t],
                    "z": z[0, t].detach().cpu(),          # [d_head]
                    "topic_probs": P_topic[0, t].detach().cpu(),
                    "metric_key": metric_keys[t] if t < len(metric_keys) else None,
                }
                index.append(entry)

    return index


def answer_question_from_corpus(
    model: HierarchicalSentenceTopicLLM,
    question: str,
    data_path: str,
    max_paragraphs: int = 1000,
    top_k: int = 3,
) -> Dict[str, object]:
    # NOTE: 테스트 코드에서 이 import 라인을 직접 검사한다.
    from reality_stone.utils.pre_segmenter import PreSegmenter  # noqa: F401

    index = build_sentence_index_from_corpus(
        model, data_path=data_path, max_paragraphs=max_paragraphs
    )
    if not index:
        return {"question": question, "answers": [], "support": []}
    device = next(model.parameters()).device
    segmenter = PreSegmenter(max_length=128, k_neighbors=3)
    seg_q = segmenter(question)
    if seg_q["metadata"]["num_sentences"] == 0:
        return {"question": question, "answers": [], "support": []}

    q_tokens = seg_q["tokens"].unsqueeze(0).to(device)  # [1, Tq, Lq]
    q_tokens_first = q_tokens[:, :1, :]                 # [1, 1, Lq]
    with torch.no_grad():
        q_emb = model.encode_sentences(q_tokens_first)          # [1,1,d_model]
        q_z = model.topic_head.poincare_embed(q_emb)            # [1,1,d_head]
        q_z = project_to_ball(q_z)[0, 0]                        # [d_head] - device 유지

    # 3) 코퍼스의 모든 문장 임베딩과 거리 계산 (Poincaré + Lorentz product manifold 거리)
    import torch as _torch
    z_corpus = _torch.stack([e["z"] for e in index], dim=0).to(device)  # [N,d_head] - device 통일
    # Poincaré 거리: 문서 3.3, 5.2의 d_{M^(ℓ)} 항
    c_p = float(model.config.c_poincare)
    N = z_corpus.shape[0]
    q_rep = q_z.unsqueeze(0).expand(N, -1)  # [N,d_head] - 이미 device에 있음
    d_p = poincare_distance(q_rep, z_corpus, c_p)  # [N] - 둘 다 같은 device
    # Lorentz 거리: Poincaré 임베딩을 Hyperboloid 로 올려서 second manifold 로 사용
    c_l = abs(float(model.config.c_lorentz)) if hasattr(model.config, "c_lorentz") else c_p
    q_l = from_poincare(q_rep, c=c_p)              # [N, d_l]
    z_l = from_poincare(z_corpus, c=c_p)           # [N, d_l] - device 통일
    d_l = lorentz_distance(q_l, z_l, c_l)          # [N]

    # Product manifold 거리: d_total^2 = λ_p d_p^2 + λ_l d_l^2
    # 학습된 lambda 사용 (첫 번째 decoder block에서)
    if hasattr(model.decoder.blocks[0], 'lambda_p'):
        lambda_p = torch.sigmoid(model.decoder.blocks[0].lambda_p).item()
        lambda_l = torch.sigmoid(model.decoder.blocks[0].lambda_l).item()
        lambda_sum = lambda_p + lambda_l + 1e-8
        lambda_p = lambda_p / lambda_sum
        lambda_l = lambda_l / lambda_sum
    else:
        lambda_p = 0.5
        lambda_l = 0.5
    dists = lambda_p * (d_p ** 2) + lambda_l * (d_l ** 2)  # [N]

    k = min(top_k, z_corpus.shape[0])
    topk_vals, topk_idx = _torch.topk(dists, k=k, largest=False)

    answers: List[Dict[str, object]] = []
    for rank, idx_i in enumerate(topk_idx.tolist(), start=1):
        e = index[idx_i]
        answers.append(
            {
                "rank": rank,
                "sentence": e["sentence"],
                "paragraph": e["paragraph"],
                "distance": float(topk_vals[rank - 1].item()),
                "metric_key": e["metric_key"],
            }
        )

    return {
        "question": question,
        "answers": answers,
        "support": [a["paragraph"] for a in answers],
    }


def answer_question_with_llm(
    model: HierarchicalSentenceTopicLLM,
    question: str,
    data_path: str,
    max_paragraphs: int = 1000,
    top_k: int = 3,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> Dict[str, object]:
    qa_ret = answer_question_from_corpus(
        model=model,
        question=question,
        data_path=data_path,
        max_paragraphs=max_paragraphs,
        top_k=top_k,
    )
    support = qa_ret.get("support", [])
    if not support:
        prompt_text = (
            f"질문: {question}\n\n"
            f"답변: (한국어로, 가능한 한 자세하고 쉽게 설명해 주세요.)"
        )
    else:
        context = "\n\n".join(support)
        prompt_text = (
            f"{context}\n\n"
            f"질문: {question}\n\n"
            f"답변: (위 컨텍스트를 참고하여, 한국어로 자세하고 쉽게 설명해 주세요.)"
        )

    # 2) 계층적 LLM을 이용한 디코딩 (문단 단위 편집 + autoregressive 확장)
    infer_out = infer_hierarchical_llm_on_text(
        model=model,
        text=prompt_text,
        max_length=256,
        k_neighbors=3,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_sampling=True,
    )
    generated_text = infer_out.get("generated_text", "")

    answer_text = generated_text
    marker = "답변:"
    if marker in generated_text:
        parts = generated_text.split(marker)
        if len(parts) > 1:
            tail = parts[-1].strip()
            if tail and len(tail) > 3:
                answer_text = tail
    
    if not answer_text or answer_text == prompt_text:
        answer_text = "죄송합니다. 답변을 생성할 수 없습니다."


    return {
        "question": question,
        "answer": answer_text,
        "support": support,
        "retrieval": qa_ret,
    }


__all__ = [
    "HierarchicalLLMConfig",
    "HierarchicalSentenceTopicLLM",
    "SentenceTopicHead",
    "MetricContextRouter",
    "HierarchicalLMDecoder",
    "RCELexicalDecoder",
    "train_hierarchical_llm_from_text",
    "infer_hierarchical_llm_on_text",
    "build_sentence_index_from_corpus",
    "answer_question_from_corpus",
    "answer_question_with_llm",
]
```
---
## File: `reality_stone/python/reality_stone/models/manifold_learner.py`

```python
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import json
import reality_stone._rust as rs_rust

class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

class GlobalManifoldLearner:
    def __init__(
        self, 
        model: nn.Module, 
        d_model: int,
        r: int = 128, 
        hyper_hidden_dim: int = 64,
        layer_emb_dim: int = 64
    ):
        self.model = model
        self.d_model = d_model
        self.r = r
        self.hyper_hidden_dim = hyper_hidden_dim
        self.layer_emb_dim = layer_emb_dim
        
        self.layers_wq = []
        self.layers_wk = []
        self.layer_indices = []
        self.layer_map = {} 
        
        self.u_global = None
        self.v_global = None
        self.hypernet = None
        self.layer_embeddings = None
        
    def collect_weights(self):
        print("Collecting weights...")
        idx = 0
        for name, module in self.model.named_modules():
            wq = None
            wk = None
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                wq = module.q_proj.weight.detach().cpu().numpy().astype(np.float32)
                wk = module.k_proj.weight.detach().cpu().numpy().astype(np.float32)
            elif hasattr(module, 'c_attn') and hasattr(module.c_attn, 'weight'):
                c_attn_w = module.c_attn.weight.detach().cpu().numpy().astype(np.float32)
                d = self.d_model
                if c_attn_w.shape == (d, 3 * d):
                    wq = c_attn_w[:, :d].T
                    wk = c_attn_w[:, d:2*d].T
                elif c_attn_w.shape == (3 * d, d):
                    wq = c_attn_w[:d, :]
                    wk = c_attn_w[d:2*d, :]
            if wq is not None and wk is not None:
                self.layers_wq.append(np.ascontiguousarray(wq))
                self.layers_wk.append(np.ascontiguousarray(wk))
                self.layer_indices.append(idx)
                self.layer_map[idx] = name
                idx += 1
                
        print(f"Collected {len(self.layers_wq)} layers.")

    def extract_global_basis(self):
        if not self.layers_wq:
            self.collect_weights()
            
        print("Extracting Global Basis (SVD)...")
        basis_dict = rs_rust.extract_global_basis(
            self.layers_wq, 
            self.layers_wk, 
            self.r
        )
        
        self.u_global = torch.from_numpy(basis_dict['u']) 
        self.v_global = self.u_global.clone() 
        
        print(f"Basis extracted. Rank: {basis_dict['rank']}")

    def train_hypernet(self, epochs=1000, batch_size=32, lr=1e-3, device=None):
        if self.u_global is None:
            self.extract_global_basis()
            
        print("Preparing Training Data (Core Tensors)...")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        u = self.u_global.to(device) 
        v = self.v_global.to(device) 
        
        targets = []
        layer_embs = []
        
        self.layer_embeddings = nn.Embedding(len(self.layers_wq), self.layer_emb_dim).to(device)
        self.hypernet = TinyMLP(self.layer_emb_dim, self.hyper_hidden_dim, self.r * self.r).to(device)
        
        optimizer = torch.optim.Adam(
            list(self.hypernet.parameters()) + list(self.layer_embeddings.parameters()), 
            lr=lr
        )
        
        with torch.no_grad():
            for i in range(len(self.layers_wq)):
                wq = torch.from_numpy(self.layers_wq[i]).to(device)
                wk = torch.from_numpy(self.layers_wk[i]).to(device)
                
                g = torch.matmul(wq.T, wk)
                
                g_sym = (g + g.T) * 0.5
                
                c = torch.matmul(torch.matmul(u.T, g_sym), u)
                
                targets.append(c.reshape(-1))
                layer_embs.append(i)
        
        targets = torch.stack(targets)
        indices = torch.tensor(layer_embs, device=device)
        
        print("Training HyperNet...")
        pbar = tqdm(range(epochs))
        loss_fn = nn.MSELoss()
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            emb = self.layer_embeddings(indices)
            pred = self.hypernet(emb)
            
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})
            
        print("HyperNet Trained.")
        
    def create_rust_hyper_metric(self):
        if self.hypernet is None:
            raise ValueError("HyperNet not trained yet.")
            
        w1 = self.hypernet.l1.weight.detach().cpu().numpy().astype(np.float32).T
        b1 = self.hypernet.l1.bias.detach().cpu().numpy().astype(np.float32)
        w2 = self.hypernet.l2.weight.detach().cpu().numpy().astype(np.float32).T
        b2 = self.hypernet.l2.bias.detach().cpu().numpy().astype(np.float32)
        
        u_np = self.u_global.detach().cpu().numpy().astype(np.float32)
        v_np = self.v_global.detach().cpu().numpy().astype(np.float32)
        
        return rs_rust.PyHyperMetric(u_np, v_np, w1, b1, w2, b2)

    def get_layer_embedding(self, idx: int):
        if self.layer_embeddings is None:
            raise ValueError("Embeddings not initialized.")
        return self.layer_embeddings(torch.tensor(idx, device=self.layer_embeddings.weight.device)).detach().cpu().numpy().astype(np.float32)

    def replace_layers(self):
        rust_hm = self.create_rust_hyper_metric()
        return SymplecticModelWrapper(self.model, self.layer_indices, rust_hm, self.layer_embeddings)

    def save_rsu_v2(self, path):
        if self.u_global is None or self.v_global is None:
            raise ValueError("Global basis not set.")
        if self.hypernet is None or self.layer_embeddings is None:
            raise ValueError("HyperNet or layer embeddings not trained.")
        path_obj = Path(path)
        header = {
            "magic": "RSULF2",
            "version": 2,
            "d_model": int(self.d_model),
            "rank": int(self.r),
            "hyper_hidden_dim": int(self.hyper_hidden_dim),
            "layer_emb_dim": int(self.layer_emb_dim),
            "num_layers": int(len(self.layers_wq) if self.layers_wq else self.layer_embeddings.num_embeddings),
            "model_type": type(self.model).__name__,
        }
        u_np = self.u_global.detach().cpu().numpy().astype(np.float32)
        v_np = self.v_global.detach().cpu().numpy().astype(np.float32)
        w1_np = self.hypernet.l1.weight.detach().cpu().numpy().astype(np.float32)
        b1_np = self.hypernet.l1.bias.detach().cpu().numpy().astype(np.float32)
        w2_np = self.hypernet.l2.weight.detach().cpu().numpy().astype(np.float32)
        b2_np = self.hypernet.l2.bias.detach().cpu().numpy().astype(np.float32)
        emb_np = self.layer_embeddings.weight.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(
            str(path_obj),
            header=json.dumps(header),
            u=u_np,
            v=v_np,
            w1=w1_np,
            b1=b1_np,
            w2=w2_np,
            b2=b2_np,
            layer_embeddings=emb_np,
        )

    @classmethod
    def from_rsu_v2(
        cls,
        model: nn.Module,
        path,
    ):
        path_obj = Path(path)
        data = np.load(str(path_obj), allow_pickle=False)
        header_raw = data["header"].item() if isinstance(data["header"], np.ndarray) else data["header"]
        header = json.loads(header_raw)
        d_model = int(header.get("d_model", 0))
        rank = int(header.get("rank", 0))
        hyper_hidden_dim = int(header.get("hyper_hidden_dim", 0))
        layer_emb_dim = int(header.get("layer_emb_dim", 0))
        num_layers = int(header.get("num_layers", 0))
        learner = cls(
            model=model,
            d_model=d_model,
            r=rank,
            hyper_hidden_dim=hyper_hidden_dim,
            layer_emb_dim=layer_emb_dim,
        )
        learner.u_global = torch.from_numpy(data["u"])
        learner.v_global = torch.from_numpy(data["v"])
        learner.layers_wq = []
        learner.layers_wk = []
        learner.layer_indices = list(range(num_layers))
        learner.layer_map = {i: str(i) for i in range(num_layers)}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hypernet = TinyMLP(layer_emb_dim, hyper_hidden_dim, rank * rank)
        hypernet.l1.weight.data.copy_(torch.from_numpy(data["w1"]).to(hypernet.l1.weight.dtype))
        hypernet.l1.bias.data.copy_(torch.from_numpy(data["b1"]).to(hypernet.l1.bias.dtype))
        hypernet.l2.weight.data.copy_(torch.from_numpy(data["w2"]).to(hypernet.l2.weight.dtype))
        hypernet.l2.bias.data.copy_(torch.from_numpy(data["b2"]).to(hypernet.l2.bias.dtype))
        learner.hypernet = hypernet.to(device)
        emb_weight = torch.from_numpy(data["layer_embeddings"])
        num_embeddings, emb_dim = emb_weight.shape
        embedding = nn.Embedding(num_embeddings, emb_dim)
        embedding.weight.data.copy_(emb_weight)
        learner.layer_embeddings = embedding.to(device)
        return learner

class SymplecticModelWrapper(nn.Module):
    def __init__(self, original_model, layer_indices, rust_hyper_metric, layer_embeddings):
        super().__init__()
        self.original_model = original_model
        self.layer_indices = set(layer_indices)
        self.hyper_metric = rust_hyper_metric
        self.layer_embeddings = layer_embeddings
        self.dt = 0.01
        
        self.symplectic_layers = {}
        device = layer_embeddings.weight.device
        
        for idx in layer_indices:
            emb = layer_embeddings(torch.tensor(idx, device=device)).detach().cpu().numpy().astype(np.float32)
            
            self.symplectic_layers[idx] = rs_rust.PySymplecticLayer(
                layer_idx=idx,
                layer_emb=emb,
                hyper_metric=rust_hyper_metric,
                dt=self.dt
            )

    def _get_layers(self):
        if hasattr(self.original_model, 'layers'):
            return list(self.original_model.layers)
        if hasattr(self.original_model, 'transformer') and hasattr(self.original_model.transformer, 'h'):
            return list(self.original_model.transformer.h)
        if hasattr(self.original_model, 'model') and hasattr(self.original_model.model, 'layers'):
            return list(self.original_model.model.layers)
        raise AttributeError("Could not find transformer layers")
            
    def forward(self, x):
        q = x
        p = torch.zeros_like(q)
        layers = self._get_layers()
        
        for i, layer in enumerate(layers):
            if i in self.symplectic_layers:
                out = layer(q)
                base_out = out[0] if isinstance(out, (tuple, list)) else out
                kick = base_out - q
                q_np = q.detach().cpu().numpy().astype(np.float32)
                p_np = p.detach().cpu().numpy().astype(np.float32)
                kick_np = kick.detach().cpu().numpy().astype(np.float32)
                
                q_out_np, p_out_np = self.symplectic_layers[i].step(q_np, p_np, kick_np)
                
                q = torch.from_numpy(q_out_np).to(q.device)
                p = torch.from_numpy(p_out_np).to(p.device)
            else:
                out = layer(q)
                q = out[0] if isinstance(out, (tuple, list)) else out
                
        return q
```
---
## File: `reality_stone/python/reality_stone/models/pretrained_backbone.py`

```python
import torch
import torch.nn as nn
from typing import Optional

class PretrainedBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "klue/bert-base",
        freeze: bool = True,
        d_model: int = 768,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.d_model = d_model
        
        try:
            from transformers import AutoModel, AutoTokenizer
            self.backbone = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            backbone_dim = self.backbone.config.hidden_size
            if backbone_dim != d_model:
                self.proj = nn.Linear(backbone_dim, d_model)
            else:
                self.proj = nn.Identity()
            
            if freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"[PretrainedBackbone] Loaded {model_name}, frozen")
            else:
                print(f"[PretrainedBackbone] Loaded {model_name}, trainable")
        except Exception as e:
            print(f"[PretrainedBackbone] Failed: {e}")
            print("[PretrainedBackbone] Random init")
            self.backbone = None
            self.tokenizer = None
            self.proj = nn.Identity()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            B, S = input_ids.shape
            return torch.randn(B, S, self.d_model, device=input_ids.device)
        outputs = self.backbone(input_ids, return_dict=True)
        hidden = outputs.last_hidden_state
        embeddings = self.proj(hidden)
        return embeddings
    
    def get_vocab_size(self) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer)
        return 32000
```
---
## File: `reality_stone/python/reality_stone/models/riemannian_aggregation.py`

```python
import torch
import torch.nn as nn
from typing import Optional
from reality_stone.layers.poincare import (
    poincare_distance,
    project_to_ball
)
from reality_stone.layers.lorentz import lorentz_distance
from reality_stone.layers.klein import klein_distance, project_to_klein

class RiemannianAggregation(nn.Module):
    def __init__(
        self,
        d_model: int,
        manifold: str = "poincare",
        c: float = 1e-3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.manifold = manifold
        self.c = abs(c)
        self.temperature = temperature
        
    def forward(
        self,
        children_states: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[torch.Tensor] = None,
        c_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = children_states.shape
        device = children_states.device
        
        if N == 0:
            return torch.zeros(B, d, device=device)
        
        if metric_ctx is not None:
            d_metric = metric_ctx.size(-1)
            if d_metric != d:
                if d_metric < d:
                    pad_size = d - d_metric
                    metric_ctx_resized = torch.nn.functional.pad(
                        metric_ctx, (0, pad_size, 0, pad_size), value=0.0
                    )
                    for i in range(d_metric, d):
                        metric_ctx_resized[:, i, i] = 1.0
                    metric_ctx = metric_ctx_resized
                else:
                    metric_ctx = metric_ctx[:, :d, :d]
            
            children_states = torch.einsum("bdk,bnk->bnd", metric_ctx, children_states)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
            mu = (children_states * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            mu = children_states.mean(dim=1)  # [B, d]
        c_val = c_override if c_override is not None else torch.as_tensor(self.c, device=children_states.device, dtype=children_states.dtype)
        
        # Riemannian aggregation
        if self.manifold == "poincare":
            return self._poincare_agg(children_states, mu, c_val, mask, temperature_override)
        elif self.manifold == "lorentz":
            return self._lorentz_agg(children_states, mu, c_val, mask, temperature_override)
        elif self.manifold == "klein":
            return self._klein_agg(children_states, mu, c_val, mask, temperature_override)
        else:
            return mu
    
    def _poincare_agg(
        self,
        children_states: torch.Tensor,
        mu: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = children_states.shape

        children_states = project_to_ball(children_states.reshape(-1, d)).reshape(B, N, d)
        mu = project_to_ball(mu)

        mu_exp = mu.unsqueeze(1).expand(B, N, d).reshape(B * N, d)
        child_flat = children_states.reshape(B * N, d)
        dist_flat = poincare_distance(mu_exp, child_flat, c)
        distances = dist_flat.reshape(B, N)
        temp = temperature_override if temperature_override is not None else torch.as_tensor(self.temperature, device=distances.device, dtype=distances.dtype)
        scores = -distances / temp

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        alpha = torch.softmax(scores, dim=1)

        weighted_mean = (alpha.unsqueeze(-1) * children_states).sum(dim=1)
        result = project_to_ball(weighted_mean)

        return result
    
    def _lorentz_agg(
        self,
        children_states: torch.Tensor,
        mu: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = children_states.shape
        
        distances = []
        for i in range(N):
            dist = lorentz_distance(
                mu,
                children_states[:, i, :],
                c
            )
            distances.append(dist)
        
        distances = torch.stack(distances, dim=1)
        temp = temperature_override if temperature_override is not None else torch.as_tensor(self.temperature, device=distances.device, dtype=distances.dtype)
        scores = -distances / temp
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        alpha = torch.softmax(scores, dim=1)
        
        weighted_mean = (alpha.unsqueeze(-1) * children_states).sum(dim=1)
        
        return weighted_mean

    def _klein_agg(
        self,
        children_states: torch.Tensor,
        mu: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = children_states.shape

        children_states = project_to_klein(children_states.reshape(-1, d), c).reshape(B, N, d)
        mu = project_to_klein(mu, c)

        mu_exp = mu.unsqueeze(1).expand(B, N, d).reshape(B * N, d)
        child_flat = children_states.reshape(B * N, d)
        dist_flat = klein_distance(mu_exp, child_flat, c)
        distances = dist_flat.reshape(B, N)
        temp = temperature_override if temperature_override is not None else torch.as_tensor(self.temperature, device=distances.device, dtype=distances.dtype)
        scores = -distances / temp

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        alpha = torch.softmax(scores, dim=1)
        weighted_mean = (alpha.unsqueeze(-1) * children_states).sum(dim=1)
        return project_to_klein(weighted_mean, c)
```
---
## File: `reality_stone/python/reality_stone/models/semantic_preservation.py`

```python
import torch
import torch.nn as nn
from typing import Optional
from reality_stone.layers.poincare import poincare_distance, project_to_ball

class SemanticPreservationLoss(nn.Module):
    def __init__(
        self,
        manifold: str = "poincare",
        c: float = 1e-3,
        reduction: str = "mean",
    ):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.reduction = reduction
    
    def forward(
        self,
        original_embeddings: torch.Tensor,
        edited_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d = original_embeddings.shape
        if self.manifold == "poincare":
            orig_proj = project_to_ball(original_embeddings.reshape(B * T, d))
            edit_proj = project_to_ball(edited_embeddings.reshape(B * T, d))
            distances = poincare_distance(orig_proj, edit_proj, self.c)
            distances = distances.reshape(B, T)
        elif self.manifold == "euclidean":
            distances = torch.norm(
                original_embeddings - edited_embeddings,
                dim=-1,
            )
        else:
            raise ValueError(f"Unsupported manifold: {self.manifold}")
        
        if mask is not None:
            distances = distances * mask
            if self.reduction == "mean":
                loss = distances.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                loss = distances.sum()
            else:
                loss = distances
        else:
            if self.reduction == "mean":
                loss = distances.mean()
            elif self.reduction == "sum":
                loss = distances.sum()
            else:
                loss = distances
        
        return loss


class ContrastiveSemanticLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        manifold: str = "poincare",
        c: float = 1e-3,
    ):
        super().__init__()
        self.temperature = temperature
        self.manifold = manifold
        self.c = c
    
    def forward(
        self,
        original_embeddings: torch.Tensor,
        edited_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d = original_embeddings.shape
        
        orig_flat = original_embeddings.reshape(B * T, d)
        edit_flat = edited_embeddings.reshape(B * T, d)
        
        if self.manifold == "poincare":
            orig_flat = project_to_ball(orig_flat)
            edit_flat = project_to_ball(edit_flat)
            
            pos_distances = poincare_distance(orig_flat, edit_flat, self.c)
            
            orig_exp = orig_flat.unsqueeze(1)
            edit_exp = edit_flat.unsqueeze(0)
            
            neg_distances = torch.cdist(orig_exp, edit_exp, p=2).squeeze(1)
            
        else:
            pos_distances = torch.norm(orig_flat - edit_flat, dim=-1)
            neg_distances = torch.cdist(orig_flat.unsqueeze(0), edit_flat.unsqueeze(0)).squeeze(0)
        
        pos_scores = -pos_distances / self.temperature
        neg_scores = -neg_distances / self.temperature
        
        eye = torch.eye(B * T, device=neg_scores.device).bool()
        neg_scores = neg_scores.masked_fill(eye, float('-inf'))
        
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        labels = torch.zeros(B * T, dtype=torch.long, device=logits.device)
        
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        return loss
```
---
## File: `reality_stone/python/reality_stone/models/top_down_decoder.py`

```python
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict


class TopDownDecoder(nn.Module):
    def __init__(self, d_model: int, d_head: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.vocab_size = vocab_size
        self.sentence_proj = nn.Linear(d_model, d_model)
        self.token_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        paragraph_embedding: Tensor,
        num_sentences: int,
        max_length: int,
        paragraph_metric: Optional[Tensor] = None,
        replacement_mask: Optional[Tensor] = None,
        original_tokens: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size = paragraph_embedding.shape[0]
        sent = self.sentence_proj(paragraph_embedding)
        sentence_embeddings = sent.unsqueeze(1).expand(batch_size, num_sentences, self.d_model)
        token_logits = self.token_proj(sentence_embeddings.reshape(batch_size * num_sentences, self.d_model))
        token_ids = token_logits.argmax(dim=-1)
        tokens = token_ids.view(batch_size, num_sentences, -1)
        seq_len = tokens.shape[2]
        if seq_len < max_length:
            pad_len = max_length - seq_len
            pad = torch.zeros(batch_size, num_sentences, pad_len, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=2)
        elif seq_len > max_length:
            tokens = tokens[:, :, :max_length]
        return {"sentence_embeddings": sentence_embeddings, "tokens": tokens}
```
