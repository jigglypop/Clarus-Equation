# LLM Symbol Index

## Python classes/functions

### `reality_stone/python/reality_stone/data.py`
- L14: `class SentenceTopicDataset(Dataset):`
- L19: `def __init__(`
- L38: `def _load_data(self):`
- L68: `def __len__(self):`
- L71: `def __getitem__(self, idx):`
- L126: `def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:`
- L207: `class SimpleTextDataset(Dataset):`
- L212: `def __init__(self, texts: List[str], tokenizer, max_len: int = 128):`
- L221: `def __len__(self):`
- L224: `def __getitem__(self, idx):`
- L233: `class TextFileDataset(Dataset):`
- L238: `def __init__(self, path: str, tokenizer, max_len: int = 128):`
- L249: `def __len__(self) -> int:`
- L252: `def __getitem__(self, idx: int) -> Dict[str, Any]:`

### `reality_stone/python/reality_stone/losses.py`
- L6: `def laplacian_same_label(dists_sq: torch.Tensor, labels: torch.Tensor, tau: float = 0.5) -> torch.Tensor:`
- L23: `def poincare_kinetic_energy(x_hyp: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:`
- L31: `class HyperbolicSupConLoss(nn.Module):`
- L36: `def __init__(self, temperature: float = 0.1, curvature: float = 1.0):`
- L41: `def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:`
- L100: `class BellmanConsistencyLoss(nn.Module):`
- L101: `def __init__(self, lambda_bellman: float = 0.1, gamma: float = 0.99, label_smoothing: float = 0.0) -> None:`
- L107: `def forward(self, logits: torch.Tensor, labels: torch.Tensor, apply_bellman: bool = True) -> dict:`

### `reality_stone/python/reality_stone/metrikey.py`
- L9: `def spd_metric_from_key_weighted(`
- L19: `def spd_metric_from_key(`
- L28: `def metric_factor_cholesky(g) -> np.ndarray:`
- L35: `def metric_from_keys(`
- L55: `def mahalanobis_distance_sq_g(x, y, g) -> float:`
- L61: `def mahalanobis_distance_sq_l(x, y, l_factor) -> float:`

### `reality_stone/python/reality_stone/_fallback.py`
- L14: `def sigmoid(x: float) -> float:`
- L18: `def dynamic_curvature(kappa: float, c_min: float, c_max: float) -> float:`
- L22: `def mobius_add_torch(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:`
- L32: `def mobius_scalar_torch(x: torch.Tensor, r: float, c: float) -> torch.Tensor:`
- L49: `def lorentz_inner_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:`
- L53: `def lorentz_distance_torch(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:`
- L64: `def klein_distance_torch(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:`
- L76: `def euclidean_metric_np(x: np.ndarray, metric_type: str, curvature: float) -> np.ndarray:`
- L85: `def geodesic_distance_np(`
- L107: `def geodesic_interpolate_np(`
- L121: `class TorchUnifiedRiemannianLayer:`
- L122: `def __init__(`
- L134: `def forward(self, x, target=None):`
- L146: `def backward(self, grad_output, x):`
- L150: `def geodesic_path(self, x, y, num_steps: int = 10):`
- L157: `def compute_energy(self, x, v, x_next, reward):`
- L173: `def flow_step(self, x, num_steps: int = 1, learning_rate: float = 0.01):`
- L181: `def _hash_seed(parts: Iterable[object]) -> int:`
- L186: `def deterministic_spd(key: str, dim: int, min_lambda: float, max_lambda: float, mass: float = 1.0) -> np.ndarray:`

### `reality_stone/python/reality_stone/_rust.py`
- L16: `def _as_f32(x):`
- L20: `def _curvature_from_kappa(kappa, c_min: float, c_max: float) -> float:`
- L26: `def _project_ball(x, c: float, eps: float = 1e-6):`
- L36: `def poincare_ball_layer_cpu(u, v, c: float, t: float):`
- L41: `def poincare_ball_layer_backward_cpu(grad, u, v, c: float, t: float):`
- L47: `def poincare_ball_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):`
- L53: `def poincare_ball_layer_layerwise_backward_cpu(`
- L68: `def poincare_to_lorentz_cpu(x, c: float):`
- L78: `def poincare_to_klein_cpu(x, c: float):`
- L85: `def lorentz_inner(u, v):`
- L91: `def lorentz_distance(u, v, c: float):`
- L97: `def lorentz_layer_forward(u, v, c: float, t: float):`
- L102: `def lorentz_ball_layer_backward_cpu(grad, u, v, c: float, t: float):`
- L108: `def lorentz_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):`
- L114: `def lorentz_add(u, v, c: float):`
- L119: `def lorentz_scalar(x, r: float, c: float):`
- L124: `def lorentz_to_poincare(x, c: float):`
- L130: `def lorentz_to_klein(x, c: float):`
- L136: `def klein_layer_forward(u, v, c: float, t: float):`
- L141: `def klein_ball_layer_backward_cpu(grad, u, v, c: float, t: float):`
- L147: `def klein_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):`
- L153: `def klein_add(u, v, c: float):`
- L157: `def klein_scalar(x, r: float, c: float):`
- L161: `def klein_distance(x, y, c: float):`
- L173: `def klein_to_poincare(x, c: float):`
- L179: `def klein_to_lorentz(x, c: float):`
- L185: `def from_poincare_dynamic_cpu(x, kappa, c_min: float, c_max: float):`
- L190: `def from_poincare_dynamic_backward_cpu(grad, x, kappa, c_min: float, c_max: float):`
- L195: `def _svd_basis(wq_list, target_rank: int):`
- L205: `def extract_global_basis(wq_list, wk_list, target_rank: int):`
- L211: `def build_causal_laplacian(seq_len: int, window: int = 1):`
- L223: `def verify_metric_consistency(wq, wk, r: int):`
- L232: `def fold_metric_svd(wq, wk, r: int):`
- L237: `def fold_metric_optimized(wq, wk, r: int):`
- L241: `def nystrom_metric(wq, wk, r: int):`
- L245: `def fold_ffn(w, r: int):`
- L256: `def bellman_geodesic_forward(x, *args, **kwargs):`
- L261: `def bellman_geodesic_backward(grad, *args, **kwargs):`
- L266: `def extract_metric_cuda(w, calib, target_dim: int, num_steps: int, curvature: float, lr: float):`
- L273: `class PyHyperMetric:`
- L274: `def __init__(self, u_global, v_global, w1, b1, w2, b2):`
- L282: `def generate_core(self, layer_emb):`
- L289: `def project_forward(self, x, layer_emb):`
- L295: `class PySymplecticLayer:`
- L296: `def __init__(self, layer_idx, layer_emb, hyper_metric, dt=0.01):`
- L302: `def step(self, q, p, kick):`
- L311: `class PyRSULFLayer:`
- L312: `def __init__(self, wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window):`
- L331: `def new_fast(cls, wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window, calibration_samples=1024):`
- L336: `def new_with_metric(cls, wq, wk, w1, w2, g_diag, d_model, r, eta, alpha, beta, gamma, seq_len, window):`
- L343: `def new_with_basis(cls, wq, wk, w1, w2, u, rank, d_model, r, eta, alpha, beta, gamma, seq_len, window):`
- L347: `def forward(self, x, v=None):`
- L357: `def export_components(self):`
- L383: `def param_count(self):`
- L389: `class PyGeodesicMemory:`
- L393: `class SplineCache:`
- L397: `class PyRiemannianDiffusion:`
- L398: `def __init__(self, dim: int, alpha: float, dt: float):`
- L403: `def step(self, h, flow):`

### `reality_stone/python/reality_stone/__init__.py`
- L181: `def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float = None, t: float = 0.5, kappas: torch.Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> torch.Tensor:`
- L185: `def klein_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:`
- L189: `def lorentz_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:`

### `reality_stone/python/reality_stone/api/indexing.py`
- L8: `class DocumentIndexer:`
- L13: `def add(self, document: str) -> int:`
- L17: `def extend(self, documents: Iterable[str]) -> list[int]:`
- L20: `def __call__(self, documents):`
- L25: `def search(self, query: str, top_k: int = 3) -> list[dict]:`

### `reality_stone/python/reality_stone/api/inference.py`
- L4: `class TextGenerator:`
- L6: `def __init__(`
- L22: `def __call__(`
- L40: `def generate_batch(`
- L49: `class TextEditor:`
- L51: `def __init__(`
- L69: `def __call__(`
- L95: `def edit_batch(`

### `reality_stone/python/reality_stone/api/pipeline.py`
- L11: `class HierarchicalLLM:`
- L13: `def __init__(`
- L30: `def from_pretrained(`
- L61: `def from_config(`
- L72: `def generate(`
- L100: `def __call__(self, text: str, **kwargs):`
- L103: `def save_pretrained(self, save_path: Union[str, Path]):`
- L127: `def pipeline(`

### `reality_stone/python/reality_stone/api/qa.py`
- L11: `class QuestionAnswerer:`
- L13: `def __init__(`
- L31: `def __call__(`
- L74: `def batch(`

### `reality_stone/python/reality_stone/clarus/agent.py`
- L39: `class CriticResult:`
- L46: `def compute_critic(`
- L73: `def select_action_discrete(`
- L84: `def select_action_continuous(`
- L96: `def bootstrap_operator(`
- L107: `def agent_step(`
- L134: `class ConsciousnessMonitor:`
- L137: `def __init__(`
- L150: `def record_deviation(self, active_frac: float, target: float = ACTIVE_RATIO) -> None:`
- L156: `def d_tau(self) -> float:`
- L162: `def consciousness_depth(self) -> float:`
- L166: `def metacognition_step(self, deviation: float) -> list[float]:`
- L179: `class WorkingMemory:`
- L182: `def __init__(self, capacity: int = WM_CAPACITY) -> None:`
- L186: `def append(self, action: Any, observation: Any) -> None:`
- L189: `def contents(self) -> list[tuple[Any, Any]]:`
- L192: `def __len__(self) -> int:`
- L196: `class CerebellumPredictor:`
- L199: `def __init__(self, dim: int, alpha: float = CEREBELLUM_ALPHA, eta: float = CEREBELLUM_ETA) -> None:`
- L204: `def predict(self) -> torch.Tensor:`
- L207: `def update(self, observation: torch.Tensor) -> torch.Tensor:`
- L215: `class RuntimeAgentConfig:`
- L226: `def __post_init__(self) -> None:`
- L237: `class RuntimeAgentStep:`
- L249: `def default_action_embeddings(action_count: int, dim: int) -> torch.Tensor:`
- L260: `class RuntimeAgent:`
- L268: `def __init__(`
- L288: `def step(`
- L344: `class TextEnvironmentStep:`
- L355: `class TextEnvironment:`
- L358: `def __init__(`
- L372: `def encode(self, text: str) -> torch.Tensor:`
- L387: `def action_embeddings(self) -> torch.Tensor:`
- L390: `def reset(self, prompt: str) -> torch.Tensor:`
- L394: `def step(self, action_index: int, state: torch.Tensor | None = None) -> TextEnvironmentStep:`
- L410: `def _render_response(self, action: str, state: torch.Tensor | None) -> str:`
- L423: `class RuntimeTextAgentTurn:`
- L430: `class RuntimeTextAgent:`
- L433: `def __init__(`
- L449: `def ask(`

### `reality_stone/python/reality_stone/clarus/bitfield.py`
- L40: `def quantize_4bit(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:`
- L51: `def dequantize_4bit(q: torch.Tensor, scale: float, zero: float) -> torch.Tensor:`
- L55: `def quantize_8bit(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:`
- L65: `def dequantize_8bit(q: torch.Tensor, scale: float, zero: float) -> torch.Tensor:`
- L70: `class BitfieldLayout:`
- L76: `def active_mask_bytes(self) -> int:`
- L80: `def freeze_mask_bytes(self) -> int:`
- L84: `def mode_bytes(self) -> int:`
- L88: `def weight_bytes(self) -> int:`
- L93: `def csr_index_bytes(self) -> int:`
- L97: `def state_bytes(self) -> int:`
- L101: `def phi_bytes(self) -> int:`
- L105: `def trace_bytes(self) -> int:`
- L110: `def total_engine_bytes(self) -> int:`
- L115: `def summary(self) -> dict[str, str]:`
- L129: `class BitfieldRuntime:`
- L136: `def __init__(self, weight: torch.Tensor, *, active_ratio: float = ACTIVE_RATIO) -> None:`
- L164: `def step(self, external: torch.Tensor | None = None) -> dict[str, float]:`
- L208: `def get_activation(self) -> torch.Tensor:`
- L211: `def memory_bytes(self) -> int:`
- L221: `class Float32Runtime:`
- L224: `def __init__(self, weight: torch.Tensor, *, active_ratio: float = ACTIVE_RATIO) -> None:`
- L233: `def step(self, external: torch.Tensor | None = None) -> dict[str, float]:`
- L263: `def get_activation(self) -> torch.Tensor:`
- L266: `def memory_bytes(self) -> int:`
- L270: `def benchmark(dim: int = 768, steps: int = 200, seed: int = 42) -> dict:`

### `reality_stone/python/reality_stone/clarus/ce_euler.py`
- L52: `def ce_rotary_base(block: int, layer_idx: int = 0, n_layers: int = 1,`
- L76: `def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:`
- L98: `def _causal_softmax_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:`
- L105: `def _chunked_decay_sdpa(`
- L157: `class EulerRotaryAttention(nn.Module):`
- L168: `def __init__(`
- L205: `def bitfield(self) -> torch.Tensor:`
- L209: `def head_freq_scalars(self) -> torch.Tensor:`
- L214: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L241: `class EulerAttnBlock(nn.Module):`
- L244: `def __init__(self, d_model: int, n_heads: int, block: int,`
- L256: `def forward(self, x):`
- L302: `class EulerCEAttention(nn.Module):`
- L313: `def __init__(`
- L365: `def extend_to(self, new_block: int) -> None:`
- L381: `def _rotate(self, x, cos, sin):`
- L390: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L417: `class EulerCEBlock(nn.Module):`
- L418: `def __init__(self, d_model: int, n_heads: int, block: int,`
- L434: `def forward(self, x):`
- L467: `class RecursiveEulerCEBlock(nn.Module):`
- L478: `def __init__(`
- L507: `def _step(self, h: torch.Tensor) -> torch.Tensor:`
- L510: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L545: `def fixed_point_loss(block: RecursiveEulerCEBlock, h: torch.Tensor,`
- L592: `def head_types_from_spec(spec, n_heads: int) -> torch.Tensor:`
- L625: `class EulerCEMinimal(nn.Module):`
- L646: `def __init__(`
- L749: `def extend_to(self, new_block: int) -> None:`
- L764: `def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:`
- L779: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L796: `def _forward_uniform(self, q, k, v, n, H):`
- L822: `def _forward_mixed(self, q, k, v, n, H):`
- L871: `class EulerCEMinimalBlock(nn.Module):`
- L874: `def __init__(self, d_model: int, n_heads: int, block: int,`
- L894: `def forward(self, x: torch.Tensor) -> torch.Tensor:`

### `reality_stone/python/reality_stone/clarus/ce_ffn.py`
- L22: `class StdFFN(nn.Module):`
- L25: `def __init__(self, d: int, mult: int = 4):`
- L30: `def forward(self, x):`
- L34: `class SwiGLU_FFN(nn.Module):`
- L37: `def __init__(self, d: int, mult: int = 4):`
- L48: `def forward(self, x):`
- L52: `class EulerDecayFFN(nn.Module):`
- L65: `def __init__(self, d: int, mult: int = 4, xi_init: float = 3.0):`
- L71: `def forward(self, x):`
- L78: `class EulerPhaseFFN(nn.Module):`
- L85: `def __init__(self, d: int, mult: int = 4, tau_init: float = 2.0,`
- L93: `def forward(self, x):`
- L100: `class EulerFullFFN(nn.Module):`
- L106: `def __init__(self, d: int, mult: int = 4,`
- L117: `def forward(self, x):`
- L126: `def make_ffn(kind: str, d: int, mult: int = 4) -> nn.Module:`

### `reality_stone/python/reality_stone/clarus/ce_laplacian.py`
- L39: `def _cosine_adjacency(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:`
- L48: `def _rbf_adjacency(z: torch.Tensor, sigma) -> torch.Tensor:`
- L63: `def _row_stochastic_causal(A: torch.Tensor, causal_mask: Optional[torch.Tensor],`
- L78: `def _sym_normalized_laplacian(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:`
- L87: `class DualLaplacianBlock(nn.Module):`
- L101: `def __init__(`
- L145: `def current_gate(self) -> tuple[torch.Tensor, torch.Tensor]:`
- L149: `def current_sigma(self) -> torch.Tensor:`
- L152: `def forward(self, h: torch.Tensor,`
- L169: `def graph_spectrum(`

### `reality_stone/python/reality_stone/clarus/ce_mra.py`
- L56: `def bootstrap_sparse(`
- L77: `class MellinRiemannAttention(nn.Module):`
- L90: `def __init__(`
- L188: `def extend_to(self, new_block: int) -> None:`
- L216: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L283: `class MRABlock(nn.Module):`
- L286: `def __init__(`
- L318: `def forward(self, x: torch.Tensor) -> torch.Tensor:`

### `reality_stone/python/reality_stone/clarus/ce_ops.py`
- L53: `def has_rust() -> bool:`
- L57: `def has_cuda() -> bool:`
- L61: `def ce_backend(device: torch.device, requested: str = "auto") -> str:`
- L86: `def _as_cpu_numpy_flat(x: torch.Tensor):`
- L90: `def _hist_from_tensors(`
- L110: `def pack_sparse(`
- L136: `def build_metric_basis(`
- L200: `def codebook_pull(`
- L241: `def _spmv_torch(`
- L267: `def _natural_direction_torch(`
- L297: `def _fdt_noise_torch(`
- L330: `def _energy_parts_torch(`
- L356: `def _relax_packed_torch(`
- L527: `def _iss_from_tail(`
- L575: `def relax_packed(`
- L712: `def relax(`
- L749: `def pq_build_codebook(`
- L821: `def pq_reconstruct_tokens(`
- L842: `def pq_scores(`

### `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py`
- L65: `def riemann_zeros(n: int) -> torch.Tensor:`
- L105: `def has_rust_riemann() -> bool:`
- L109: `def has_cuda_riemann() -> bool:`
- L116: `def _build_phase_and_sheet(`
- L143: `def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:`
- L155: `def _sheet_bias(sheet: torch.Tensor, lambda_sigma: torch.Tensor) -> torch.Tensor:`
- L168: `def _attention_torch(`
- L184: `class RiemannRotaryAttention(nn.Module):`
- L198: `def __init__(`
- L248: `def _resolve_backend(self, x: torch.Tensor) -> str:`
- L271: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L306: `def _forward_native(`
- L339: `def _forward_cuda_devptr(`
- L366: `class RiemannAttnBlock(nn.Module):`
- L369: `def __init__(`
- L390: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L402: `def riemann_zero_init(linear: nn.Linear, axis: str = "in") -> None:`

### `reality_stone/python/reality_stone/clarus/ce_softmax.py`
- L38: `class ModeGate:`
- L49: `def as_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:`
- L54: `def mode_gate(mode: str) -> ModeGate:`
- L70: `def lang_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:`
- L76: `def lang_attention(q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:`
- L87: `def grav_scores(`
- L103: `def grav_attention(`
- L123: `def metric_family_attention(`
- L179: `class CESoftmaxAttention(nn.Module):`
- L186: `def __init__(`
- L211: `def set_mode(self, mode: str) -> None:`
- L215: `def forward(`

### `reality_stone/python/reality_stone/clarus/ce_zeta.py`
- L38: `def _eta_truncated(x: torch.Tensor, N: int = 24) -> tuple[torch.Tensor, torch.Tensor]:`
- L57: `def _zeta_critical(x: torch.Tensor, N: int = 24`
- L76: `def zeta_magnitude_sq(x: torch.Tensor, N: int = 24) -> torch.Tensor:`
- L82: `class ZetaActivation(nn.Module):`
- L91: `def __init__(self, N: int = 24, lam_init: float = 0.1):`
- L99: `def _init_stats(self, x: torch.Tensor) -> None:`
- L106: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L115: `class ZetaFFN(nn.Module):`
- L118: `def __init__(self, d: int, mult: int = 4, N: int = 24):`
- L124: `def forward(self, x):`

### `reality_stone/python/reality_stone/clarus/daemon.py`
- L55: `class DaemonConfig:`
- L68: `class DaemonStats:`
- L83: `class BrainDaemon:`
- L86: `def __init__(`
- L130: `def start(self) -> None:`
- L137: `def stop(self) -> None:`
- L144: `def query(self, prompt: str, max_tokens: int = 30, timeout: float = 10.0) -> str:`
- L152: `def teach(self, fact: str, repetitions: int = 3, timeout: float = 15.0) -> dict:`
- L161: `def think(self, topic: str, depth: int = 5, timeout: float = 15.0) -> list[str]:`
- L170: `def recall(self, cue: str, timeout: float = 10.0) -> str:`
- L178: `def _loop(self) -> None:`
- L206: `def _make_ce_args(self, steps: int = 20, noise: float = 0.003):`
- L214: `def _relax_and_generate(self, prompt: str, max_tokens: int = 30,`
- L238: `def _handle_query(self, prompt, event, result, max_tokens) -> None:`
- L259: `def _handle_teach(self, fact, event, result, repetitions) -> None:`
- L301: `def _handle_think(self, topic, event, result, depth) -> None:`
- L355: `def _handle_recall(self, cue, event, result, max_tokens) -> None:`
- L389: `def _idle_tick(self) -> None:`
- L406: `def _post_step(self, step: RuntimeStep, external: torch.Tensor) -> None:`
- L464: `def _encode_prompt(self, prompt: str) -> torch.Tensor:`
- L470: `def _save_checkpoint(self) -> None:`
- L481: `def status(self) -> dict:`

### `reality_stone/python/reality_stone/clarus/device.py`
- L8: `def auto_device(preference: str = "auto") -> torch.device:`
- L22: `def device_summary(device: torch.device) -> str:`

### `reality_stone/python/reality_stone/clarus/dimensionless.py`
- L21: `def _frac(x: int | float | Fraction) -> Fraction:`
- L25: `def dim(*exponents: int | float | Fraction) -> DimVector:`
- L42: `class Quantity:`
- L50: `def dimensionless(self) -> bool:`
- L55: `class GateResult(Generic[T]):`
- L62: `def ok(cls, value: T) -> "GateResult[T]":`
- L66: `def fail(cls, *errors: str) -> "GateResult[T]":`
- L70: `def passed(self) -> bool:`
- L73: `def map(self, transform: Callable[[T], U]) -> "GateResult[U]":`
- L78: `def bind(self, transform: Callable[[T], "GateResult[U]"]) -> "GateResult[U]":`
- L83: `def unwrap(self) -> T:`
- L91: `def is_dimensionless(dims: Sequence[Fraction]) -> bool:`
- L95: `def same_rank_dims(quantities: Iterable[Quantity]) -> int:`
- L102: `def require_dimensionless(quantity: Quantity, *, context: str = "") -> Quantity:`
- L111: `def check_dimensionless(quantity: Quantity, *, context: str = "") -> GateResult[Quantity]:`
- L120: `def audit_dimensionless(`
- L140: `def _rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:`
- L165: `def nullspace(matrix: Sequence[Sequence[int | float | Fraction]]) -> list[list[Fraction]]:`
- L187: `def buckingham_pi_groups(quantities: Sequence[Quantity]) -> list[dict[str, Fraction]]:`
- L200: `def evaluate_group(quantities: Sequence[Quantity], exponents: dict[str, Fraction]) -> float:`
- L208: `def group_dimension(quantities: Sequence[Quantity], exponents: dict[str, Fraction]) -> DimVector:`
- L219: `def nondimensionalize(quantity: Quantity, scales: Sequence[Quantity]) -> Quantity:`
- L240: `def exp_argument(quantity: Quantity) -> float:`
- L247: `def exp_arguments(quantities: Iterable[Quantity]) -> GateResult[tuple[float, ...]]:`

### `reality_stone/python/reality_stone/clarus/engine.py`
- L51: `def postprocess_output(text: str) -> str:`
- L72: `def _rounded_count(total: int, ratio: float) -> int:`
- L76: `def update_phi(phi: torch.Tensor, m_star: torch.Tensor, phi_var: torch.Tensor | None = None) -> torch.Tensor:`
- L86: `def _optional_float(value) -> float | None:`
- L98: `def _format_optional(value) -> str:`
- L103: `def state_partition_counts(dim: int, active_ratio: float, struct_ratio: float) -> tuple[int, int, int]:`
- L127: `class PromptContext:`
- L137: `class CEEngine:`
- L138: `def __init__(self, path: str, device: str = "cpu", backend: str = "torch"):`
- L293: `def _load_w_pack(self, data):`
- L305: `def _load_model(self):`
- L338: `def _build_state_graph_laplacian(self) -> torch.Tensor:`
- L347: `def _build_state_coords(self) -> torch.Tensor:`
- L355: `def state_coords(self) -> torch.Tensor:`
- L360: `def state_graph_laplacian(self) -> torch.Tensor:`
- L365: `def weight_density(self, w: torch.Tensor | None = None) -> float:`
- L374: `def resparsify_relax_matrix(`
- L411: `def apply_relax_matrix(self, w: torch.Tensor):`
- L435: `def build_brain_runtime(`
- L455: `def apply_state_partition(`
- L475: `def active_indices(self) -> torch.Tensor | None:`
- L481: `def struct_indices(self) -> torch.Tensor | None:`
- L487: `def _projection_indices(self) -> torch.Tensor | None:`
- L491: `def _compress_state_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:`
- L502: `def _compress_prev_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:`
- L513: `def _compress_token_state_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:`
- L524: `def _compress_runtime_projections(self):`
- L541: `def state_partition(`
- L571: `def masked_state(`
- L582: `def _project_state_query(self, state_hidden: torch.Tensor) -> torch.Tensor:`
- L600: `def _project_prev_query(self, prev_emb: torch.Tensor) -> torch.Tensor:`
- L617: `def _get_w_eigvecs(self, metric_rank: int) -> torch.Tensor | None:`
- L628: `def memory_usage(self) -> dict[str, float]:`
- L749: `def save_artifact(self, path: str):`
- L752: `def save_runtime_artifact(self, path: str):`
- L759: `def has_standalone_lexicon(self) -> bool:`
- L764: `def prompt_embeddings(self, prompt_ids: torch.Tensor) -> torch.Tensor:`
- L773: `def runtime_prompt_state(`
- L815: `def token_embedding(self, token_ids: int | list[int] | torch.Tensor) -> torch.Tensor:`
- L841: `def lexical_scores(self, query: torch.Tensor) -> torch.Tensor:`
- L854: `def _rescale_to_reference(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:`
- L868: `def decoder_query(self, state_hidden: torch.Tensor, prev_emb: torch.Tensor) -> torch.Tensor:`
- L892: `def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:`
- L900: `def _merge_candidate_ids(self, *groups: torch.Tensor | None) -> torch.Tensor:`
- L916: `def _sentence_terminal_ids(self) -> torch.Tensor:`
- L937: `def _sentence_close_bonus(self, candidate_ids: torch.Tensor, *, generated_len: int) -> torch.Tensor:`
- L953: `def _paper_candidate_count(self, vocab_size: int, top_k: int) -> int:`
- L959: `def ensure_vocab_head(self):`
- L976: `def vocab_logits(self, query: torch.Tensor) -> torch.Tensor:`
- L983: `def _ngram_repeat_scores(self, history_ids: list[int] | None, candidate_ids: torch.Tensor) -> torch.Tensor:`
- L1000: `def _curvature_adjust_logits(`
- L1090: `def build_runtime_codebook(self, m_ref: torch.Tensor, top_k: int) -> torch.Tensor:`
- L1100: `def ce_hidden(self, m_star: torch.Tensor) -> torch.Tensor:`
- L1103: `def teacher_embedding(self, token_ids: torch.Tensor | list[int]) -> torch.Tensor:`
- L1111: `def teacher_hidden_and_logits(self, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:`
- L1114: `def teacher_next_logits(self, prompt_ids: torch.Tensor) -> torch.Tensor:`
- L1117: `def decoder_token_correction(`
- L1144: `def standalone_logits(`
- L1237: `def apply_vocab_head(`
- L1254: `def apply_decoder_refine(`
- L1269: `def apply_token_head(`
- L1290: `def decoder_snapshot(self) -> dict[str, torch.Tensor | float | None]:`
- L1314: `def restore_decoder_snapshot(self, snapshot: dict[str, torch.Tensor | float | None]):`
- L1365: `def standalone_generate(`
- L1490: `def legacy_generate(`
- L1550: `def prompt_context(self, prompt: str) -> PromptContext:`
- L1554: `def _analyze_prompt_ids(`
- L1610: `def context_from_ids(`
- L1636: `def relax_context(self, ctx: PromptContext, args):`
- L1696: `def select_mode(self, phi_updated: torch.Tensor, args) -> str:`
- L1703: `def _copy_args(args, **updates):`
- L1708: `def decode_outputs(self, ctx: PromptContext, relax_result: dict, args):`
- L1755: `def reference_generate(self, prompt: str, max_new_tokens: int) -> str:`
- L1759: `def build_prompt_list(args) -> list[str]:`
- L1770: `def build_guard_list(args) -> list[str]:`
- L1781: `def load_microsleep_tools():`
- L1789: `def main():`

### `reality_stone/python/reality_stone/clarus/evidence.py`
- L72: `class EvidenceCheck:`
- L81: `def is_gate_ready(self) -> bool:`
- L85: `def is_reproducible(self) -> bool:`
- L90: `class ArtifactCheck:`
- L100: `def is_reproducible(self) -> bool:`
- L103: `def to_dict(self) -> dict[str, object]:`
- L115: `class LinearDecoderGate:`
- L126: `def passed(self) -> bool:`
- L129: `def to_dict(self) -> dict[str, object]:`
- L142: `class LocomotionGatePanel:`
- L148: `def recording_count(self) -> int:`
- L151: `def pass_count(self, target: str) -> int:`
- L154: `def pass_rate(self, target: str) -> float:`
- L159: `def summary(self, targets: tuple[str, ...] = ("velocity", "curvature")) -> dict[str, object]:`
- L184: `def to_dict(self) -> dict[str, object]:`
- L196: `def passed(`
- L206: `class LocomotionControlComparison:`
- L215: `def target_summary(self, target: str) -> dict[str, object]:`
- L227: `def passed(self) -> bool:`
- L230: `def to_dict(self) -> dict[str, object]:`
- L241: `def assess_manifest(manifest: Mapping[str, object]) -> EvidenceCheck:`
- L298: `def validate_locomotion_artifact(artifact: Mapping[str, object]) -> ArtifactCheck:`
- L337: `def validate_locomotion_artifact_file(path: str | Path) -> ArtifactCheck:`
- L352: `def celegans_elife_66135_manifest() -> dict[str, object]:`
- L401: `def linear_decoder_gate(`
- L455: `def celegans_locomotion_gate(`
- L486: `def celegans_locomotion_gate_from_pickle(`
- L511: `def build_locomotion_gate_artifact(`
- L541: `def build_locomotion_control_artifact(`
- L573: `def compare_locomotion_to_control(`
- L592: `def main(argv: Sequence[str] | None = None) -> int:`
- L687: `def _missing(manifest: Mapping[str, object], fields: tuple[str, ...]) -> tuple[str, ...]:`
- L691: `def _empty(value: object) -> bool:`
- L701: `def _clean_xy(features: object, target: object) -> tuple[np.ndarray, np.ndarray]:`
- L715: `def _celegans_features(`
- L733: `def _standardize_train_test(`
- L743: `def _fit_predict_ridge(`
- L757: `def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:`

### `reality_stone/python/reality_stone/clarus/neuromod.py`
- L29: `class NeuromodulatorState:`
- L36: `def as_tuple(self) -> tuple[float, float, float, float]:`
- L40: `def step_neuromodulators(`
- L61: `class ModulationEffect:`
- L69: `def apply_modulation(`

### `reality_stone/python/reality_stone/clarus/ops.py`
- L37: `def _as_flat_f32(t: torch.Tensor) -> np.ndarray:`
- L44: `def _from_flat(arr: np.ndarray, shape: Tuple[int, ...], device, dtype) -> torch.Tensor:`
- L54: `def topk_silu(x: torch.Tensor, k: int, ratio: float, threshold: float = 0.0) -> torch.Tensor:`
- L78: `def lbo_fused_fwd(`
- L128: `def power_iter_step(`
- L154: `def gauge_lattice_fwd(`
- L192: `def channel(x_part, up, down, k, hid):`
- L212: `def ops_backend() -> str:`

### `reality_stone/python/reality_stone/clarus/quantum.py`
- L17: `def quantum_phase_step(`
- L30: `def wick_rotate(`
- L41: `def quantum_to_real(psi: torch.Tensor) -> torch.Tensor:`
- L48: `def check_norm_conservation(psi_before: torch.Tensor, psi_after: torch.Tensor, tol: float = 1e-6) -> bool:`
- L55: `def convergence_inequality(`
- L69: `def time_curvature(m_history: Sequence[torch.Tensor]) -> float:`
- L80: `def estimate_mu(`
- L119: `def iss_ball_radius(`
- L148: `def pci_regression(`
- L182: `def iss_report(`

### `reality_stone/python/reality_stone/clarus/reality.py`
- L22: `def _ensure_local_source_path() -> None:`
- L30: `def load_reality_stone() -> ModuleType:`
- L45: `def has_reality_stone() -> bool:`
- L54: `class RealityStoneStatus:`
- L62: `def status() -> RealityStoneStatus:`
- L76: `def metric_attention(*args: Any, **kwargs: Any):`
- L81: `def unified_riemannian_layer(*args: Any, **kwargs: Any):`
- L89: `def convert_transformer_to_rsulf(model: Any, *args: Any, **kwargs: Any):`

### `reality_stone/python/reality_stone/clarus/research.py`
- L15: `class RuntimeProbeResult:`
- L24: `class PhaseLockProbeResult:`
- L33: `class PhaseNetworkProbeResult:`
- L41: `def pca_projection(hidden: torch.Tensor, max_dim: int) -> tuple[torch.Tensor, torch.Tensor]:`
- L57: `def apply_projection(hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:`
- L67: `def hopfield_from_hidden(hidden: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:`
- L83: `def normalized_drive(hidden: torch.Tensor, gain: float = 0.4) -> torch.Tensor:`
- L88: `def periodic_stimulus(`
- L110: `def phase_coherence(state: torch.Tensor, *, eps: float = 1e-8) -> float:`
- L120: `def phase_alignment(state: torch.Tensor, reference: torch.Tensor, *, eps: float = 1e-8) -> float:`
- L133: `def phase_grounding_risk(`
- L146: `def phase_grounding_suppression(`
- L167: `def phase_coupling_step(`
- L196: `def phase_network_probe(`
- L226: `def phase_lock_step(`
- L247: `def phase_lock_probe(`
- L280: `def _as_complex(state: torch.Tensor) -> torch.Tensor:`
- L286: `def build_runtime(`
- L316: `def train_runtime_stdp(runtime: BrainRuntime, train_drive: torch.Tensor, steps: int) -> BrainRuntime:`
- L326: `def evaluate_transition_probe(runtime: BrainRuntime, eval_drive: torch.Tensor) -> RuntimeProbeResult:`

### `reality_stone/python/reality_stone/clarus/runtime.py`
- L67: `class RuntimeMode(str, Enum):`
- L73: `class ModuleLifecycle(str, Enum):`
- L96: `class BrainRuntimeConfig:`
- L138: `def __post_init__(self) -> None:`
- L156: `def energy_budget(self, mode: RuntimeMode) -> int:`
- L164: `def activation_decay(self, mode: RuntimeMode) -> float:`
- L171: `def activation_gain(self, mode: RuntimeMode) -> float:`
- L178: `def refractory_decay(self, mode: RuntimeMode) -> float:`
- L185: `def refractory_gain(self, mode: RuntimeMode) -> float:`
- L192: `def replay_mix(self, mode: RuntimeMode) -> float:`
- L201: `class RuntimeStep:`
- L216: `class BrainRuntimeSnapshot:`
- L243: `class HippocampusMemory:`
- L252: `def __post_init__(self) -> None:`
- L257: `def __len__(self) -> int:`
- L260: `def encode(`
- L279: `def decay_priorities(self, steps: int = 1) -> None:`
- L286: `def recall(self, cue: torch.Tensor, *, topk: int = 4) -> torch.Tensor:`
- L304: `def replay(self, mode: RuntimeMode) -> torch.Tensor:`
- L314: `def state_dict(self) -> dict[str, object]:`
- L326: `def from_state_dict(`
- L344: `class BrainRuntime:`
- L353: `def __init__(`
- L458: `def _rebuild_sparse(self) -> None:`
- L479: `def _apply_dale_sign(self) -> None:`
- L484: `def _apply_runtime_stdp(self, active_count: int, energy: float) -> float:`
- L527: `def brainwave_observable(self) -> dict[str, float]:`
- L549: `def energy_full(self) -> float:`
- L556: `def compute_self_state(self) -> dict[str, float]:`
- L578: `def set_goal(self, goal: torch.Tensor | None) -> None:`
- L587: `def active_mask(self) -> torch.Tensor:`
- L590: `def lifecycle_counts(self) -> Dict[str, int]:`
- L596: `def mode_occupancy_kl(self, eps: float = 1e-9) -> Dict[str, float]:`
- L631: `def reset_mode_occupancy(self) -> None:`
- L636: `def _f1_effective_budget(self, mode: RuntimeMode) -> int:`
- L656: `def _f1_update_ema(self, active_count: int) -> None:`
- L663: `def bridge_gate_report(self) -> Dict[str, Dict[str, float]]:`
- L692: `def _matvec(self, x: torch.Tensor) -> torch.Tensor:`
- L695: `def _select_active(self, salience: torch.Tensor, budget: int) -> torch.Tensor:`
- L710: `def _auto_mode(self, external_norm: float) -> RuntimeMode:`
- L725: `def _update_sleep_state(self, mode: RuntimeMode, active_count: int, external_norm: float) -> None:`
- L748: `def nrem_target_length(self) -> float:`
- L753: `def _update_lifecycle(self, salience: torch.Tensor, active_mask: torch.Tensor) -> None:`
- L774: `def _energy(self, recurrent: torch.Tensor, replay: torch.Tensor) -> float:`
- L783: `def _use_rust(self) -> bool:`
- L792: `def _step_rust(`
- L863: `def _compute_salience(`
- L879: `def _step_torch(`
- L961: `def step(`
- L1018: `def snapshot(self) -> BrainRuntimeSnapshot:`
- L1047: `def from_snapshot(`

### `reality_stone/python/reality_stone/clarus/sleep.py`
- L29: `class SleepBatch:`
- L46: `class DecoderTokenHead:`
- L55: `class PromptReplayBuffer:`
- L59: `def add(self, prompt: str):`
- L66: `def extend(self, prompts: list[str]):`
- L70: `def items(self) -> list[str]:`
- L79: `def _split_corpus_documents(text: str) -> list[str]:`
- L98: `def _chunk_document(text: str, *, max_chars: int = 320, min_chars: int = 64) -> list[str]:`
- L150: `def load_corpus_documents(`
- L202: `def _content_terms(text: str) -> set[str]:`
- L206: `def prioritize_documents_for_prompts(`
- L249: `def ridge_solve(`
- L267: `def fit_linear_with_bias(`
- L279: `def batch_weights(batch: SleepBatch, rem_weight: float) -> torch.Tensor:`
- L290: `def fit_decoder_from_batch(`
- L320: `def fit_token_head_from_batch(`
- L379: `def finetune_vocab_head_from_batch(`
- L478: `def build_refresh_args(`
- L496: `def allocate_phase_sample_counts(`
- L543: `def _build_sleep_batch(`
- L575: `def _mean_phase_grounding_risk(step_meta: dict[str, object]) -> float:`
- L582: `def _context_slice(full_ids: torch.Tensor, end_pos: int, window_tokens: int) -> torch.Tensor:`
- L587: `def _target_distribution(`
- L645: `def collect_sleep_batch(`
- L830: `def batch_stats(batch: SleepBatch) -> dict[str, float]:`
- L852: `def classify_state_dimensions(`
- L880: `def _weighted_covariance(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:`
- L889: `def covariance_delta(batch: SleepBatch, *, emphasize_hard: float = 1.0) -> torch.Tensor:`
- L897: `def offdiag_density(mask: torch.Tensor) -> float:`
- L906: `def row_topk_mask(matrix: torch.Tensor, keep_ratio: float) -> torch.Tensor:`
- L927: `def normalize_update(matrix: torch.Tensor) -> torch.Tensor:`
- L934: `def smooth_weight_matrix(w: torch.Tensor, laplacian: torch.Tensor, eta: float) -> torch.Tensor:`
- L941: `def apply_nrem_weight_update(`
- L978: `def apply_rem_weight_update(`
- L1023: `def evaluate_guard_set(`
- L1139: `def should_accept_guard_update(`
- L1169: `def run_guarded_microsleep_step(`
- L1290: `def maybe_refresh_pq(`
- L1339: `def run_sleep_cycle(`
- L1631: `def run_guarded_microsleep_session(`
- L1749: `def build_prompts(args) -> list[str]:`
- L1762: `def main():`

### `reality_stone/python/reality_stone/clarus/stdp.py`
- L31: `class STDPConfig:`
- L43: `class EligibilityTracker:`
- L46: `def __init__(self, config: STDPConfig, device: str | torch.device = "cpu") -> None:`
- L53: `def update(self, activation: torch.Tensor) -> None:`
- L62: `def reset(self) -> None:`
- L67: `def state_dict(self) -> dict[str, torch.Tensor]:`
- L74: `def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:`
- L80: `def compute_learning_gate(`
- L102: `def structural_projection(`
- L129: `def apply_stdp_update(`

### `reality_stone/python/reality_stone/clarus/utils.py`
- L16: `def safe_print(text: object) -> None:`
- L26: `def normalize_vector(x: torch.Tensor) -> torch.Tensor:`
- L35: `def resolve_device(name: str) -> torch.device:`

### `reality_stone/python/reality_stone/core/mobius.py`
- L13: `class MobiusAdd(Function):`
- L15: `def forward(`
- L73: `def backward(ctx, grad_output: Tensor):`
- L119: `class MobiusScalarMul(Function):`
- L121: `def forward(ctx, x: Tensor, r: float, c: float) -> Tensor:`
- L142: `def backward(ctx, grad_output: Tensor):`

### `reality_stone/python/reality_stone/layers/diffusion.py`
- L6: `class RiemannianDiffusionStep(torch.autograd.Function):`
- L8: `def forward(ctx, h, flow, diffusion_engine, alpha, dt):`
- L34: `def backward(ctx, grad_output):`
- L46: `class RiemannianDiffusionModule(nn.Module):`
- L47: `def __init__(self, dim, alpha=0.9, dt=0.1, num_steps=5):`
- L65: `def forward(self, h):`

### `reality_stone/python/reality_stone/layers/klein.py`
- L9: `class KleinLayer(Function):`
- L11: `def forward(`
- L53: `def backward(ctx, grad_output: Tensor):`
- L99: `def klein_add(u: Tensor, v: Tensor, c: float) -> Tensor:`
- L105: `def klein_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:`
- L111: `def klein_distance(x: Tensor, y: Tensor, c: float) -> Tensor:`
- L134: `def klein_to_poincare(x: Tensor, c: float) -> Tensor:`
- L141: `def klein_to_lorentz(x: Tensor, c: float) -> Tensor:`
- L149: `class KleinFromPoincare(Function):`
- L151: `def forward(ctx, x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:`
- L172: `def backward(ctx, grad_output: Tensor):`
- L187: `def from_poincare(x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:`
- L191: `def project_to_klein(x: Tensor, c: float | Tensor, epsilon: float = 1e-5) -> Tensor:`

### `reality_stone/python/reality_stone/layers/lorentz.py`
- L8: `class LorentzDistance(Function):`
- L13: `def forward(ctx, u: Tensor, v: Tensor, c: float) -> Tensor:`
- L33: `def backward(ctx, grad_output: Tensor):`
- L68: `def lorentz_distance(x: Tensor, y: Tensor, c: float | Tensor) -> Tensor:`
- L81: `class LorentzLayer(Function):`
- L86: `def forward(ctx, u: Tensor, v: Tensor, c: float, t: float) -> Tensor:`
- L104: `def backward(ctx, grad_output: Tensor):`
- L126: `def lorentz_add(u: Tensor, v: Tensor, c: float) -> Tensor:`
- L135: `def lorentz_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:`
- L144: `def lorentz_inner(u: Tensor, v: Tensor) -> Tensor:`
- L153: `def lorentz_to_poincare(x: Tensor, c: float) -> Tensor:`
- L163: `def lorentz_to_klein(x: Tensor, c: float) -> Tensor:`
- L172: `def euclidean_to_lorentz(x: Tensor, c: float = 1.0, epsilon: float = 1e-6) -> Tensor:`
- L184: `class LorentzBallLayer(Function):`
- L191: `def forward(ctx, u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = 0.1, c_max: float = 5.0) -> Tensor:`
- L225: `def backward(ctx, grad_output: Tensor):`
- L321: `def lorentz_ball(u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = 0.1, c_max: float = 5.0) -> Tensor:`
- L327: `class LorentzFromPoincare(Function):`
- L332: `def forward(ctx, x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:`
- L353: `def backward(ctx, grad_output: Tensor):`
- L367: `def from_poincare(x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:`

### `reality_stone/python/reality_stone/layers/metric_attention.py`
- L18: `class SPDMetric(nn.Module):`
- L19: `def __init__(self, hidden_size: int, rank: int = 0, init_u_scale: float = 1e-3) -> None:`
- L30: `def scale_q(self, q: Tensor) -> Tensor:`
- L34: `def scale_k(self, k: Tensor) -> Tensor:`
- L38: `def lowrank_proj(self, x: Tensor) -> Optional[Tensor]:`
- L44: `def _sparsemax(logits: Tensor, dim: int = -1) -> Tensor:`
- L60: `def _sinkhorn(logits: Tensor, iters: int = 20, tau: float = 1.0, eps: float = 1e-9) -> Tensor:`
- L67: `def normalize(scores: Tensor, method: str = "softmax", tau: float = 1.0) -> Tensor:`
- L78: `def build_topo_topk(topo_idx: Dict[str, Tensor], topk_cfg: Dict[str, int]) -> Tensor:`
- L112: `def masked_gather(scores: Tensor, idx: Tensor) -> Tensor:`
- L120: `def aggregate(weights: Tensor, values: Tensor, idx: Tensor) -> Tensor:`
- L134: `def get_default_topk_cfg() -> Dict[str, int]:`
- L138: `class MetricAttention(nn.Module):`
- L139: `def __init__(`
- L159: `def _apply_metric_factor(self, x: Tensor, l_factor: Tensor) -> Tensor:`
- L165: `def _cholesky_from_keys(`
- L202: `def _geodesic_distance_pairs(self, q_pairs: Tensor, k_pairs: Tensor, c: float) -> Tensor:`
- L217: `def forward(`

### `reality_stone/python/reality_stone/layers/poincare.py`
- L11: `def project_to_ball(x: Tensor, epsilon: float = 1e-7) -> Tensor:`
- L25: `class PoincareBallLayer(Function):`
- L31: `def forward(ctx, u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:`
- L88: `def backward(ctx, grad_output: Tensor):`
- L167: `def poincare_add(`
- L181: `def poincare_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:`
- L187: `def poincare_distance(x: Tensor, y: Tensor, c: float | Tensor, eps: float = 1e-7) -> Tensor:`
- L216: `def poincare_to_lorentz(x: Tensor, c: float) -> Tensor:`
- L230: `def poincare_to_klein(x: Tensor, c: float) -> Tensor:`
- L243: `def exp_map_zero(v: Tensor, c: float, eps: float = 1e-7) -> Tensor:`
- L264: `def log_map_zero(y: Tensor, c: float, eps: float = 1e-7) -> Tensor:`
- L291: `class HyperbolicLinear(nn.Module):`
- L299: `def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True, mode: str = 'tangent'):`
- L316: `def reset_parameters(self) -> None:`
- L333: `def forward(self, x: Tensor) -> Tensor:`
- L385: `def extra_repr(self) -> str:`
- L389: `def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):`
- L404: `class PoincareWrapper(nn.Module):`
- L409: `def __init__(self, linear_layer: nn.Module):`
- L413: `def forward(self, x: Tensor) -> Tensor:`
- L430: `def __repr__(self):`
- L433: `class GeodesicLinear(nn.Module):`
- L439: `def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):`
- L454: `def reset_parameters(self) -> None:`
- L465: `def forward(self, x: Tensor) -> Tensor:`
- L492: `def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):`
- L502: `class EquivalentHyperbolicLinear(nn.Module):`
- L508: `def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):`
- L528: `def reset_parameters(self) -> None:`
- L541: `def forward(self, x: Tensor) -> Tensor:`
- L547: `def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):`
- L557: `class CompactEquivalentHyperbolicLinear(nn.Module):`
- L566: `def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):`
- L586: `def reset_parameters(self) -> None:`
- L595: `def forward(self, x: Tensor) -> Tensor:`
- L616: `def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):`
- L625: `def _extract_linear_like(linear_layer: nn.Module) -> tuple[int, int, torch.Tensor, bool]:`

### `reality_stone/python/reality_stone/layers/rsulf_cuda.py`
- L26: `class RSULFLayerCUDA(nn.Module):`
- L27: `def __init__(`
- L168: `def _get_graph_laplacian(self, seq_len: int, device: torch.device) -> torch.Tensor:`
- L186: `def _compute_riemannian_laplacian(self, x: torch.Tensor) -> torch.Tensor:`
- L192: `def _compute_graph_diffusion(self, x: torch.Tensor) -> torch.Tensor:`
- L198: `def _update_bellman_memory(self, phi: torch.Tensor) -> torch.Tensor:`
- L208: `def reset_bellman_memory(self):`
- L211: `def _compute_potential_and_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:`
- L241: `def _geodesic_update(self, x: torch.Tensor, v_mem: Optional[torch.Tensor] = None) -> torch.Tensor:`
- L259: `def set_ffn_gate(self, w_gate, b_gate=None):`
- L266: `def _norm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:`
- L279: `def _gelu_new(self, x: torch.Tensor) -> torch.Tensor:`
- L282: `def _pfc_cap_and_project(self, h: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:`
- L295: `def _pfc_gate(self, v: torch.Tensor) -> torch.Tensor:`
- L307: `def _pfc_bilinear(self, h_seq: torch.Tensor) -> torch.Tensor:`
- L341: `def _pfc_accel(self, h_seq: torch.Tensor) -> torch.Tensor:`
- L377: `def _apply_pfc(self, h_seq: torch.Tensor) -> torch.Tensor:`
- L388: `def set_ln1(self, weight, bias=None):`
- L393: `def set_ln2(self, weight, bias=None):`
- L398: `def set_attention_weights(self, wv, wo):`
- L402: `def set_biases(self, bq=None, bk=None, bv=None, bo=None, b1=None, b2=None):`
- L410: `def forward(`
- L536: `def forward_step(self, x_t: torch.Tensor, cache: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:`
- L662: `def param_count(self) -> Tuple[int, int, float]:`
- L666: `def curvature(self) -> float:`
- L670: `def eta(self) -> float:`
- L674: `def alpha(self) -> float:`
- L678: `def beta(self) -> float:`
- L682: `def gamma(self) -> float:`
- L686: `def g_diag(self) -> np.ndarray:`
- L690: `def g_inv(self) -> np.ndarray:`
- L694: `class RSULFWrapperCUDA(nn.Module):`
- L695: `def __init__(self, rsulf_layer: RSULFLayerCUDA):`
- L704: `def reset_memory(self):`
- L710: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L719: `def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:`
- L727: `def init_step_cache(self, batch: int, max_len: int, device: torch.device, dtype: torch.dtype):`
- L746: `class RSULFLMHeadCUDA(nn.Module):`
- L747: `def __init__(`
- L763: `def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:`

### `reality_stone/python/reality_stone/layers/spline.py`
- L7: `class SplineLinear(nn.Module):`
- L8: `def __init__(self, in_features: int, out_features: int, k: int = 8, bias: bool = True, use_residual: bool = True):`
- L28: `def forward(self, input: torch.Tensor) -> torch.Tensor:`
- L44: `def interpolate_weights_static(control_points, k, out_features):`
- L66: `def interpolate_weights_torch(self) -> torch.Tensor:`
- L69: `def precompute_weight(self) -> None:`
- L74: `def _refresh_blend_matrix(self) -> None:`
- L103: `def from_linear(cls, linear: nn.Linear, k: int = 8,`
- L132: `def extra_repr(self) -> str:`
- L136: `def get_compression_ratio(self) -> float:`

### `reality_stone/python/reality_stone/layers/suppression.py`
- L6: `class HyperbolicSuppressionField(nn.Module):`
- L7: `def __init__(self, base: float = 0.37, linear: float = 0.0, hyp: float = 0.1, scale: float = 1.0) -> None:`
- L14: `def compute_field(self, x: Tensor) -> Tensor:`
- L18: `def compute_effective_temperature(self, t0, x: Tensor) -> Tensor:`

### `reality_stone/python/reality_stone/models/bottom_up_encoder.py`
- L9: `class BottomUpEncoder(nn.Module):`
- L10: `def __init__(`
- L28: `def encode_tokens_to_sentences(`
- L52: `def encode_sentences_to_paragraph(`
- L66: `def forward(`

### `reality_stone/python/reality_stone/models/hierarchical_sentence_topic_llm.py`
- L26: `class HierarchicalLLMConfig:`
- L101: `class EditOperationHead(nn.Module):`
- L102: `def __init__(self, d_model: int, num_ops: int = 5, edit_budget: float = 0.25) -> None:`
- L112: `def forward(self, hidden: torch.Tensor) -> torch.Tensor:`
- L115: `def apply_edits(`
- L186: `class SentenceOrderHead(nn.Module):`
- L187: `def __init__(self, d_model: int) -> None:`
- L192: `def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:`
- L197: `class TreeNodeOperator(nn.Module):`
- L198: `def __init__(`
- L230: `def _curvatures(self, device: torch.device, dtype: torch.dtype):`
- L244: `def up_operator(`
- L267: `def down_operator(`
- L286: `class LevelInvariantTreeProcessor(nn.Module):`
- L287: `def __init__(self, d_model: int, enable_dynamic_manifold: bool = False) -> None:`
- L298: `def process_tree(`
- L352: `def _depth(self, tree: DocumentTree, node_id: int) -> int:`
- L367: `def compute_dynamic_lambda(`
- L393: `class RiemannianDiffusionStep(torch.autograd.Function):`
- L395: `def forward(ctx, h: torch.Tensor, flow: torch.Tensor, diffusion_engine, alpha: float, dt: float) -> torch.Tensor:`
- L418: `def backward(ctx, grad_output: torch.Tensor):`
- L428: `class SentenceTopicHead(nn.Module):`
- L429: `def __init__(`
- L472: `def forward(`
- L535: `class MetricContextRouter(nn.Module):`
- L536: `def __init__(`
- L568: `def _clamp_eigen(self, G: torch.Tensor) -> torch.Tensor:`
- L579: `def _make_metric(self, key: str, score_q: float, device: torch.device) -> torch.Tensor:`
- L613: `def forward(self, metric_keys: List[str], scores: torch.Tensor) -> torch.Tensor:`
- L643: `def _spd_log_euclidean_mean(`
- L688: `class SPDMetricMixer(nn.Module):`
- L689: `def __init__(`
- L712: `def mix_hierarchy(`
- L752: `class RCELexicalDecoder(nn.Module):`
- L753: `def __init__(`
- L768: `def forward(`
- L798: `class HierarchicalLMDecoder(nn.Module):`
- L799: `def __init__(`
- L827: `def _make_block(self) -> nn.Module:`
- L830: `def forward(`
- L854: `class _DecoderBlock(nn.Module):`
- L855: `def __init__(self, d_model: int, n_head: int, manifold: str, c: float) -> None:`
- L889: `def forward(`
- L951: `class HierarchicalSentenceTopicLLM(nn.Module):`
- L952: `def __init__(self, config: HierarchicalLLMConfig) -> None:`
- L1083: `def from_checkpoint(cls, checkpoint: Dict) -> "HierarchicalSentenceTopicLLM":`
- L1110: `def encode_tokens_to_sentences(`
- L1181: `def encode_sentences_to_paragraph(`
- L1210: `def encode_sentences(`
- L1232: `def _encode_with_tree_processor(`
- L1312: `def forward(`
- L1722: `def train_hierarchical_llm_from_text(`
- L1889: `def _apply_top_down_decoding(`
- L1961: `def infer_hierarchical_llm_on_text(`
- L2168: `def build_sentence_index_from_corpus(`
- L2204: `def answer_question_from_corpus(`
- L2282: `def answer_question_with_llm(`

### `reality_stone/python/reality_stone/models/manifold_learner.py`
- L9: `class TinyMLP(nn.Module):`
- L10: `def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):`
- L16: `def forward(self, x):`
- L19: `class GlobalManifoldLearner:`
- L20: `def __init__(`
- L44: `def collect_weights(self):`
- L71: `def extract_global_basis(self):`
- L87: `def train_hypernet(self, epochs=1000, batch_size=32, lr=1e-3, device=None):`
- L144: `def create_rust_hyper_metric(self):`
- L158: `def get_layer_embedding(self, idx: int):`
- L163: `def replace_layers(self):`
- L167: `def save_rsu_v2(self, path):`
- L203: `def from_rsu_v2(`
- L244: `class SymplecticModelWrapper(nn.Module):`
- L245: `def __init__(self, original_model, layer_indices, rust_hyper_metric, layer_embeddings):`
- L266: `def _get_layers(self):`
- L275: `def forward(self, x):`

### `reality_stone/python/reality_stone/models/pretrained_backbone.py`
- L5: `class PretrainedBackbone(nn.Module):`
- L6: `def __init__(`
- L41: `def forward(self, input_ids: torch.Tensor) -> torch.Tensor:`
- L50: `def get_vocab_size(self) -> int:`

### `reality_stone/python/reality_stone/models/riemannian_aggregation.py`
- L11: `class RiemannianAggregation(nn.Module):`
- L12: `def __init__(`
- L25: `def forward(`
- L72: `def _poincare_agg(`
- L102: `def _lorentz_agg(`
- L134: `def _klein_agg(`

### `reality_stone/python/reality_stone/models/semantic_preservation.py`
- L6: `class SemanticPreservationLoss(nn.Module):`
- L7: `def __init__(`
- L18: `def forward(`
- L57: `class ContrastiveSemanticLoss(nn.Module):`
- L58: `def __init__(`
- L69: `def forward(`

### `reality_stone/python/reality_stone/models/top_down_decoder.py`
- L7: `class TopDownDecoder(nn.Module):`
- L8: `def __init__(self, d_model: int, d_head: int, vocab_size: int) -> None:`
- L16: `def forward(`

### `reality_stone/python/reality_stone/models/transformer_converter.py`
- L26: `class ConversionStats:`
- L36: `class RSULFTransformerConverter:`
- L37: `def __init__(`
- L89: `def extract_weights(self, layer) -> Dict[str, np.ndarray]:`
- L208: `def verify_weights(self, weights: Dict[str, np.ndarray], idx: int) -> Tuple[bool, Dict]:`
- L221: `def convert_layer(self, layer, idx: int) -> Tuple[Optional[RSULFLayerCUDA], Dict[str, Any]]:`
- L303: `def convert_model(self, model) -> "RSULFModel":`
- L564: `def _save_checkpoint(self, layers: List[RSULFLayerCUDA], count: int):`
- L592: `def analyze_errors(self) -> Dict[str, Any]:`
- L602: `def verify_conversion(self, wq: np.ndarray, wk: np.ndarray) -> Dict:`
- L610: `class RSULFModel(torch.nn.Module):`
- L611: `def __init__(self, layers: List[RSULFLayerCUDA], stats: Optional[ConversionStats] = None):`
- L619: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L624: `def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:`
- L629: `def reset_memory(self):`
- L633: `def init_step_cache(self, batch: int, max_len: int, device: torch.device, dtype: torch.dtype):`
- L639: `class TorchRiemannianDecoder(nn.Module):`
- L640: `def __init__(self, u: np.ndarray, a: np.ndarray, bt: np.ndarray, bias: np.ndarray):`
- L647: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L666: `class SyntaxHead(nn.Module):`
- L667: `def __init__(self, d_model: int):`
- L676: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L681: `class RSULFCausalLM(nn.Module):`
- L682: `def __init__(`
- L702: `def forward(self, input_ids: torch.Tensor) -> torch.Tensor:`
- L719: `def _decode_hidden(self, h: torch.Tensor) -> torch.Tensor:`
- L730: `def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:`
- L740: `def generate_sample(`
- L806: `def save_rsulf_causal_lm(path: str, rs_lm: RSULFCausalLM, decoder_state: dict | None = None) -> None:`
- L857: `def load_rsulf_causal_lm(path: str, device: str | torch.device | None = None) -> RSULFCausalLM:`
- L942: `def build_rsulf_causal_lm(model: nn.Module, converter: RSULFTransformerConverter) -> RSULFCausalLM:`
- L946: `def wrap_rsulf_as_causal_lm(model: nn.Module, rsulf: RSULFModel) -> RSULFCausalLM:`
- L1004: `def convert_transformer_to_rsulf(`
- L1035: `class FFNPotential(nn.Module):`
- L1036: `def __init__(self, d_model: int, hidden_dim: int):`
- L1046: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L1054: `def gradient(self, x: torch.Tensor) -> torch.Tensor:`
- L1062: `class LowRankFFN(nn.Module):`
- L1063: `def __init__(self, mlp: nn.Module, rank: int):`
- L1095: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L1105: `def param_count(self):`
- L1111: `class StructuralRSULFLayer(nn.Module):`
- L1112: `def __init__(self, block: nn.Module, d_model: int, rank: int):`
- L1120: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L1130: `class StructuralRSULFModel(nn.Module):`
- L1131: `def __init__(`
- L1148: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L1154: `def wrappers(self) -> nn.ModuleList:`

### `reality_stone/python/reality_stone/optim/riemannian_adam.py`
- L15: `class PoincareRiemannianAdam(Optimizer):`
- L16: `def __init__(`
- L47: `def step(self, closure: Optional[Callable] = None):`
- L117: `def zero_grad(self, set_to_none: bool = False):`

### `reality_stone/python/reality_stone/utils/misc.py`
- L5: `def get_device() -> str:`
- L8: `def set_seed(seed: int = 42):`
- L18: `def load_mnist_dataloaders(data_dir="./data", batch_size=256, test_batch_size=1000, download=True):`
- L34: `def evaluate_accuracy(model, test_loader, device):`

### `reality_stone/python/reality_stone/utils/pre_segmenter.py`
- L13: `class TreeNode:`
- L21: `class DocumentTree:`
- L25: `def children(self, node_id: int) -> List[int]:`
- L29: `class LevelSegmenter:`
- L30: `def __init__(self, level: str, parent: "PreSegmenter"):`
- L35: `def segment(self, text: str) -> List[str]:`
- L57: `class PreSegmenter:`
- L58: `def __init__(`
- L77: `def recursive_segment(self, text: str, levels: List[str] = ['document', 'paragraph', 'sentence', 'token']) -> DocumentTree:`
- L121: `def __call__(self, paragraph: str) -> Dict:`
- L147: `def _segment_sentences(self, paragraph: str) -> List[str]:`
- L188: `def _tokenize_sentences(self, sentences: List[str]) -> Tuple[torch.Tensor, List[List[str]]]:`
- L244: `def _generate_replacement_mask(`
- L280: `def _is_replaceable(self, token: str) -> bool:`
- L318: `def _build_topology(self, num_sentences: int, k: int = 3) -> torch.Tensor:`
- L349: `def _build_document_tree(`

### `reality_stone/python/reality_stone/utils/sampling.py`
- L3: `def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:`
- L14: `def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:`
- L34: `def sample_next_token(`

### `reality_stone/python/reality_stone/utils/text_corpus.py`
- L8: `class CorpusDoc:`
- L13: `def iter_text_files(`
- L35: `def read_text_file(path: str | Path) -> Optional[str]:`
- L50: `def load_corpus(`
- L73: `def chunk_text(`

### `reality_stone/tests/llm/test_bindings_cuda_symbols.py`
- L7: `def test_rust_extension_loaded_when_cuda_available():`
- L20: `def test_required_cuda_symbols_exist_on_rust_module():`
- L58: `def test_has_cuda_flag_consistent_with_symbols_and_torch():`

### `reality_stone/tests/llm/test_dynamic_manifold.py`
- L10: `def test_dynamic_manifold_basic():`
- L20: `def test_dynamic_manifold_up_operator():`
- L34: `def test_dynamic_manifold_down_operator():`
- L48: `def test_dynamic_manifold_selection():`
- L64: `def test_dynamic_manifold_gradient_flow():`
- L79: `def test_tree_processor_dynamic_manifold():`
- L88: `def test_dynamic_manifold_different_manifolds():`
- L107: `def test_dynamic_manifold_weighted_combination():`
- L133: `def test_dynamic_manifold_batch_consistency():`
- L152: `def test_dynamic_manifold_deterministic():`

### `reality_stone/tests/llm/test_edit_operations.py`
- L8: `def edit_head():`
- L12: `def test_edit_head_forward(edit_head):`
- L22: `def test_edit_head_apply_edits_disabled(edit_head):`
- L39: `def test_edit_head_apply_edits_enabled(edit_head):`
- L58: `def test_edit_head_apply_edits_with_replacement_mask(edit_head):`
- L78: `def test_edit_head_budget_constraint(edit_head):`
- L100: `def test_edit_head_keep_operation(edit_head):`
- L120: `def test_edit_head_replace_operation(edit_head):`
- L146: `def test_edit_head_delete_operation(edit_head):`

### `reality_stone/tests/llm/test_gpt2_last_layer.py`
- L10: `def test_gpt2_last_layer_rsulf_forward():`

### `reality_stone/tests/llm/test_gpt2_manifold_learner.py`
- L7: `def test_gpt2_manifold_learner_collect_weights():`
- L42: `def test_gpt2_manifold_learner_full_pipeline():`

### `reality_stone/tests/llm/test_hierarchical_integration.py`
- L13: `def sample_config():`
- L30: `def sample_model(sample_config):`
- L34: `def test_hierarchical_llm_forward_basic(sample_model):`
- L55: `def test_hierarchical_llm_forward_with_tree(sample_model):`
- L75: `def test_hierarchical_llm_forward_loss_components(sample_model):`
- L96: `def test_hierarchical_llm_encode_decode_cycle(sample_model):`
- L111: `def test_hierarchical_llm_metric_context_generation(sample_model):`
- L130: `def test_hierarchical_llm_backward_pass(sample_model):`
- L154: `def test_infer_hierarchical_llm_basic():`
- L185: `def test_infer_hierarchical_llm_with_top_down():`
- L213: `def test_hierarchical_llm_structural_edit():`
- L241: `def test_hierarchical_llm_dynamic_manifold():`
- L267: `def test_hierarchical_llm_empty_input():`
- L292: `def test_hierarchical_llm_long_input():`
- L318: `def test_hierarchical_llm_gradient_accumulation(sample_model):`

### `reality_stone/tests/llm/test_hierarchical_llm.py`
- L15: `def sample_config():`
- L30: `def sample_model(sample_config):`
- L35: `def test_recursive_segment():`
- L55: `def test_full_edit_ops(sample_model):`
- L75: `def test_pretrain_loading():`
- L92: `def test_model_initialization(sample_config):`
- L109: `def test_forward_pass_shape(sample_model):`
- L133: `def test_encode_tokens_to_sentences(sample_model):`
- L147: `def test_encode_sentences_to_paragraph(sample_model):`
- L171: `def test_full_pipeline_config_grid(d_model, n_layer_decoder):`
- L212: `def test_bottom_up_encoder_shapes(sample_config):`
- L244: `def test_top_down_decoder_shapes(sample_config):`

### `reality_stone/tests/llm/test_manifold_symplectic_pipeline.py`
- L8: `class ToyBlock(nn.Module):`
- L9: `def __init__(self, d_model: int):`
- L14: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L18: `class ToyModel(nn.Module):`
- L19: `def __init__(self, d_model: int, num_layers: int):`
- L23: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L29: `def _init_toy_qk_weights(model: ToyModel, d_model: int) -> None:`
- L39: `def test_global_manifold_learner_creates_hypermetric_toy():`

### `reality_stone/tests/llm/test_metric_attention.py`
- L5: `def test_metric_attention_dot_product_basic():`
- L29: `def test_metric_attention_geodesic_with_topology():`

### `reality_stone/tests/llm/test_metric_router.py`
- L14: `def sample_router():`
- L19: `def test_metric_router_shape_and_spd(sample_router):`
- L43: `def test_metric_router_identity_fallback_without_metrikey():`
- L62: `def test_metric_router_cache_functionality(sample_router):`
- L82: `def test_metric_router_score_quantization(sample_router):`

### `reality_stone/tests/llm/test_poincare_cuda.py`
- L8: `def test_has_cuda_flag_matches_torch():`
- L24: `def test_poincare_ball_layer_cpu_cuda_consistency():`
- L41: `def project_to_ball(x, epsilon=1e-5):`
- L64: `def test_lorentz_layer_cpu_cuda_consistency_forward_backward():`
- L76: `def sample_lorentz(batch: int, d: int, device: torch.device) -> torch.Tensor:`
- L108: `def test_klein_layer_cpu_cuda_consistency_forward_backward():`

### `reality_stone/tests/llm/test_rce_lexical_decoder.py`
- L8: `def sample_decoder():`
- L18: `def test_rce_decoder_shapes_and_mask_preservation(sample_decoder):`
- L68: `def test_rce_decoder_respects_lexical_candidates(sample_decoder):`
- L106: `def test_rce_decoder_no_candidates_fallback(sample_decoder):`
- L130: `def test_rce_decoder_all_masked(sample_decoder):`

### `reality_stone/tests/llm/test_rsu_v2_symplectic_pipeline.py`
- L8: `class ToyBlock(nn.Module):`
- L9: `def __init__(self, d_model: int):`
- L14: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L18: `class ToyModel(nn.Module):`
- L19: `def __init__(self, d_model: int, num_layers: int):`
- L23: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
- L29: `def _init_toy_qk_weights(model: ToyModel, d_model: int) -> None:`
- L39: `def test_rsu_v2_symplectic_end_to_end(tmp_path):`

### `reality_stone/tests/llm/test_sentence_topic_head.py`
- L8: `def sample_topic_head():`
- L19: `def test_sentence_topic_head_output_shapes_and_probs(sample_topic_head):`
- L44: `def test_sentence_topic_head_metric_keys_format(sample_topic_head):`
- L71: `def test_sentence_topic_head_poincare_projection(sample_topic_head):`
- L90: `def test_sentence_topic_head_topic_names(sample_topic_head):`
- L109: `def test_sentence_topic_head_gradient_flow(sample_topic_head):`

### `reality_stone/tests/llm/test_spd_performance.py`
- L11: `def test_spd_fast_mixing_vs_log_euclidean():`
- L32: `def test_spd_fast_mixing_performance():`
- L52: `def test_spd_log_euclidean_mean_batch():`
- L67: `def test_spd_mixer_gradient_flow():`
- L85: `def test_spd_mixer_with_children():`
- L103: `def test_spd_mixer_weights_normalized():`
- L119: `def test_spd_fast_mixing_cuda():`

### `reality_stone/tests/llm/test_top_down_decoder.py`
- L13: `def sample_model():`
- L28: `def sample_tree():`
- L37: `def test_top_down_decoding_basic(sample_model, sample_tree):`
- L61: `def test_top_down_decoding_with_tree_processor(sample_model, sample_tree):`
- L86: `def test_top_down_decoding_preserves_structure(sample_model, sample_tree):`

### `reality_stone/tests/llm/test_tree_processor.py`
- L12: `def sample_tree():`
- L24: `def test_tree_node_operator_up():`
- L37: `def test_tree_node_operator_down():`
- L51: `def test_tree_node_operator_dynamic_manifold():`
- L64: `def test_tree_processor_up(sample_tree):`
- L84: `def test_tree_processor_down(sample_tree):`
- L99: `def test_tree_processor_dynamic_manifold(sample_tree):`
- L117: `def test_pre_segmenter_tree_output():`

### `reality_stone/tests/api/test_pipeline_api.py`
- L11: `def small_config():`
- L25: `def sample_model(small_config):`
- L29: `class TestHierarchicalLLM:`
- L31: `def test_from_config(self, small_config):`
- L37: `def test_from_config_dict(self):`
- L51: `def test_call_inference(self, sample_model):`
- L61: `def test_save_and_load(self, sample_model):`
- L74: `class TestPipeline:`
- L76: `def test_pipeline_text_generation(self, small_config):`
- L85: `def test_pipeline_text_editing(self, small_config):`
- L95: `def test_pipeline_with_model_instance(self, sample_model):`
- L103: `def test_pipeline_invalid_task(self, small_config):`
- L107: `def test_pipeline_no_model_or_config(self):`
- L112: `class TestTextGenerator:`
- L114: `def test_single_generation(self, small_config):`
- L123: `def test_batch_generation(self, small_config):`
- L133: `class TestTextEditor:`
- L135: `def test_edit_with_structural_edit(self, small_config):`
- L145: `def test_edit_without_structural_edit(self, small_config):`
- L154: `def test_batch_editing(self, small_config):`

## Rust public-ish symbols

### `reality_stone/src/bindings/bellman.rs`
- L15: `#[pyfunction]`
- L17: `pub fn bellman_geodesic_forward<'py>(`
- L31: `#[pyfunction]`
- L33: `pub fn bellman_geodesic_backward<'py>(`
- L48: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/diffusion.rs`
- L7: `fn riemannian_diffusion_step_cuda(`
- L19: `#[pyclass]`
- L20: `pub struct PyRiemannianDiffusion {`
- L24: `#[pymethods]`
- L25: `impl PyRiemannianDiffusion {`
- L27: `pub fn new(dim: usize, alpha: f32, dt: f32) -> Self {`
- L36: `pub fn step_cuda(`
- L60: `pub fn step_cpu<'py>(`
- L74: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/extraction.rs`
- L5: `#[pyfunction]`
- L7: `pub fn extract_metric_cuda_py<'py>(`
- L26: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/geodesic_attention.rs`
- L10: `fn geodesic_topk_attention_cuda(`
- L30: `fn batched_cholesky_cuda_ffi(A: *const f32, L: *mut f32, batch_count: i32, d: i32);`
- L48: `#[pyfunction]`
- L50: `pub fn geodesic_topk_attention(`
- L155: `#[pyfunction]`
- L156: `pub fn batched_cholesky_cuda(`
- L201: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/hyper_metric.rs`
- L6: `#[pyclass]`
- L7: `pub struct PyHyperMetric {`
- L11: `#[pymethods]`
- L12: `impl PyHyperMetric {`
- L15: `pub fn new(`
- L39: `pub fn generate_core<'py>(`
- L48: `pub fn project_forward<'py>(`
- L61: `#[pyclass]`
- L62: `pub struct PySymplecticLayer {`
- L66: `#[pymethods]`
- L67: `impl PySymplecticLayer {`
- L69: `pub fn new(`
- L84: `pub fn step<'py>(`
- L102: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/klein.rs`
- L6: `#[pyfunction]`
- L7: `pub fn klein_add<'py>(`
- L19: `#[pyfunction]`
- L20: `pub fn klein_scalar<'py>(`
- L31: `#[pyfunction]`
- L32: `pub fn klein_distance<'py>(`
- L44: `#[pyfunction]`
- L45: `pub fn klein_to_poincare<'py>(`
- L55: `#[pyfunction]`
- L56: `pub fn klein_to_lorentz<'py>(`
- L66: `#[pyfunction]`
- L67: `pub fn klein_layer_forward<'py>(`
- L80: `#[pyfunction]`
- L81: `pub fn klein_ball_layer_backward_cpu<'py>(`
- L96: `#[pyfunction]`
- L97: `fn from_poincare_dynamic_cpu<'py>(`
- L111: `#[pyfunction]`
- L112: `fn from_poincare_dynamic_backward_cpu<'py>(`
- L136: `#[pyfunction]`
- L137: `pub fn klein_distance_cuda(`
- L157: `#[pyfunction]`
- L158: `pub fn klein_layer_forward_cuda(`
- L180: `#[pyfunction]`
- L181: `pub fn klein_ball_layer_backward_cuda(`
- L206: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/lorentz.rs`
- L7: `#[pyfunction]`
- L8: `pub fn lorentz_add<'py>(`
- L20: `#[pyfunction]`
- L21: `pub fn lorentz_scalar<'py>(`
- L32: `#[pyfunction]`
- L33: `pub fn lorentz_distance<'py>(`
- L45: `#[pyfunction]`
- L46: `pub fn lorentz_inner<'py>(`
- L57: `#[pyfunction]`
- L58: `pub fn lorentz_to_poincare<'py>(`
- L68: `#[pyfunction]`
- L69: `pub fn lorentz_to_klein<'py>(`
- L79: `#[pyfunction]`
- L80: `pub fn lorentz_layer_forward<'py>(`
- L93: `#[pyfunction]`
- L94: `pub fn lorentz_ball_layer_backward_cpu<'py>(`
- L109: `#[pyfunction]`
- L110: `pub fn lorentz_layer_dynamic_cpu<'py>(`
- L126: `#[pyfunction]`
- L127: `pub fn lorentz_layer_dynamic_backward_cpu<'py>(`
- L146: `#[pyfunction]`
- L147: `pub fn lorentz_layer_layerwise_cpu<'py>(`
- L164: `#[pyfunction]`
- L165: `pub fn lorentz_layer_layerwise_backward_cpu<'py>(`
- L192: `#[pyfunction]`
- L193: `pub fn lorentz_distance_cuda(`
- L213: `#[pyfunction]`
- L214: `pub fn lorentz_layer_forward_cuda(`
- L236: `#[pyfunction]`
- L237: `pub fn lorentz_ball_layer_backward_cuda(`
- L262: `#[pyfunction]`
- L263: `fn from_poincare_dynamic_cpu<'py>(`
- L277: `#[pyfunction]`
- L278: `fn from_poincare_dynamic_backward_cpu<'py>(`
- L301: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/memory.rs`
- L5: `#[pyclass]`
- L6: `pub struct PyGeodesicMemory {`
- L10: `#[pymethods]`
- L11: `impl PyGeodesicMemory {`
- L13: `pub fn new(d_model: usize, threshold: f32) -> Self {`
- L19: `pub fn push(&mut self, t: usize, x: PyReadonlyArray1<f32>) -> bool {`
- L23: `pub fn query<'py>(&self, py: Python<'py>, t: f32) -> &'py PyArray1<f32> {`
- L28: `pub fn get_stats(&self) -> (usize, usize, f32) {`
- L32: `pub fn reset(&mut self) {`

### `reality_stone/src/bindings/metrikey.rs`
- L6: `#[pyfunction]`
- L7: `pub fn householder_chain_apply_from_key(`
- L20: `#[pyfunction]`
- L21: `pub fn householder_chain_apply_transpose_from_key(`
- L34: `#[pyfunction]`
- L35: `pub fn givens_chain_apply_from_key(`
- L48: `#[pyfunction]`
- L49: `pub fn lowrank_plus_diag_apply_from_key(`
- L63: `#[pyfunction]`
- L64: `pub fn rotate_metric_factor_block<'py>(`
- L75: `#[pyfunction]`
- L76: `pub fn spd_metric_from_key_weighted<'py>(`
- L88: `#[pyfunction]`
- L89: `pub fn compose_layers_gravity<'py>(`
- L102: `#[pyfunction]`
- L103: `pub fn compose_layers_gravity_f64<'py>(`
- L115: `#[pyfunction]`
- L116: `pub fn apply_linear_f64<'py>(`
- L127: `#[pyfunction]`
- L128: `pub fn effective_metric_from_transform_f64<'py>(`
- L137: `#[pyfunction]`
- L138: `pub fn metric_factor_cholesky_f64<'py>(`
- L147: `#[pyfunction]`
- L148: `pub fn compose_layers_gravity_compact_f64<'py>(`
- L165: `#[pyclass]`
- L166: `pub struct CollapsedTransformF32 {`
- L172: `#[pyclass]`
- L173: `pub struct CollapsedTransformF64 {`
- L178: `#[pymethods]`
- L179: `impl CollapsedTransformF64 {`
- L181: `fn new(t: PyReadonlyArray2<f64>) -> Self {`
- L188: `fn from_keys(`
- L200: `fn from_master_key_compact(`
- L215: `fn apply<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> &'py PyArray2<f64> {`
- L221: `fn matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {`
- L226: `fn dim(&self) -> usize {`
- L231: `#[pymethods]`
- L232: `impl CollapsedTransformF32 {`
- L234: `fn new(t: PyReadonlyArray2<f32>) -> Self {`
- L241: `fn from_keys(`
- L253: `fn from_master_key_compact(`
- L271: `fn apply<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>) -> &'py PyArray2<f32> {`
- L278: `fn matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {`
- L283: `fn dim(&self) -> usize {`
- L288: `#[pyfunction]`
- L289: `pub fn spd_metric_from_key<'py>(`
- L300: `#[pyfunction]`
- L301: `pub fn metric_factor_cholesky<'py>(`
- L310: `#[pyfunction]`
- L311: `pub fn mahalanobis_distance_sq_g(`
- L322: `#[pyfunction]`
- L323: `pub fn mahalanobis_distance_sq_l(`
- L334: `#[pyfunction]`
- L335: `pub fn block_orthogonal_from_key<'py>(`
- L345: `#[pyfunction]`
- L346: `pub fn spd_block_metric_from_key<'py>(`
- L359: `#[pyfunction]`
- L360: `pub fn compose_layers_order_preserving<'py>(`
- L372: `#[pyfunction]`
- L373: `pub fn apply_linear<'py>(`
- L384: `pub fn init_module(_py: Python, m: &PyModule) -> PyResult<()> {`
- L399: `#[pyfunction]`
- L400: `fn layer_norm_forward_exact_f32_py<'py>(`
- L417: `#[pyfunction]`
- L418: `fn gelu_new_f32_py<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>) -> &'py PyArray2<f32> {`
- L423: `#[pyfunction]`
- L424: `fn softmax_lastdim_f32_py<'py>(`
- L437: `#[pyfunction]`
- L438: `fn layer_norm_forward_exact_f64_py<'py>(`
- L455: `#[pyfunction]`
- L456: `fn gelu_new_f64_py<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {`
- L461: `#[pyfunction]`
- L462: `fn softmax_lastdim_f64_py<'py>(`
- L496: `#[pyclass]`
- L497: `pub struct CollapsedRunnerF32 {`
- L505: `#[pyclass]`
- L506: `pub struct CollapsedRunnerF64 {`
- L513: `#[pymethods]`
- L514: `impl CollapsedRunnerF64 {`
- L516: `fn new(`
- L535: `fn step<'py>(&self, py: Python<'py>, ids: PyReadonlyArray1<'py, i64>) -> &'py PyArray2<f64> {`
- L560: `#[pymethods]`
- L561: `impl CollapsedRunnerF32 {`
- L563: `fn new(`
- L582: `fn step<'py>(&self, py: Python<'py>, ids: PyReadonlyArray1<'py, i64>) -> &'py PyArray2<f32> {`

### `reality_stone/src/bindings/mobius.rs`
- L5: `#[pyfunction]`
- L6: `pub fn mobius_add_cpu<'py>(`
- L19: `#[pyfunction]`
- L20: `pub fn mobius_add_cuda(`
- L36: `#[pyfunction]`
- L37: `pub fn mobius_scalar_cpu<'py>(`
- L49: `#[pyfunction]`
- L50: `pub fn mobius_scalar_cuda(`
- L66: `#[pyfunction]`
- L67: `pub fn mobius_add_dynamic_cpu<'py>(`
- L83: `#[pyfunction]`
- L84: `pub fn mobius_add_dynamic_backward_cpu<'py>(`
- L102: `#[pyfunction]`
- L103: `pub fn mobius_add_layerwise_cpu<'py>(`
- L119: `#[pyfunction]`
- L120: `pub fn mobius_add_layerwise_backward_cpu<'py>(`
- L144: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/mod.rs`
- L24: `pub fn _rust(py: Python, m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/poincare.rs`
- L28: `#[pyfunction]`
- L29: `pub fn poincare_ball_layer_cpu<'py>(`
- L42: `#[pyfunction]`
- L43: `pub fn poincare_exp_at_cpu<'py>(`
- L56: `#[pyfunction]`
- L57: `pub fn poincare_log_at_cpu<'py>(`
- L69: `#[pyfunction]`
- L70: `pub fn poincare_ball_layer_backward_cpu<'py>(`
- L90: `#[pyfunction]`
- L91: `pub fn poincare_ball_layer_dynamic_cpu<'py>(`
- L107: `#[pyfunction]`
- L108: `pub fn poincare_ball_layer_dynamic_backward_cpu<'py>(`
- L132: `#[pyfunction]`
- L133: `pub fn poincare_ball_layer_layerwise_cpu<'py>(`
- L150: `#[pyfunction]`
- L151: `pub fn poincare_ball_layer_layerwise_backward_cpu<'py>(`
- L177: `#[pyfunction]`
- L178: `pub fn mobius_add_vjp_cpu<'py>(`
- L194: `#[pyfunction]`
- L195: `pub fn mobius_scalar_vjp_cpu<'py>(`
- L206: `#[pyfunction]`
- L207: `pub fn project_to_ball_cpu<'py>(`
- L215: `#[pyfunction]`
- L216: `pub fn poincare_riemannian_adam_step_cpu<'py>(`
- L257: `#[pyfunction]`
- L258: `pub fn poincare_distance_cuda(`
- L280: `#[pyfunction]`
- L281: `pub fn poincare_ball_layer_cuda(`
- L303: `#[pyfunction]`
- L304: `pub fn poincare_ball_layer_backward_cuda(`
- L331: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/riemann.rs`
- L4: `#[pyfunction]`
- L5: `pub fn riemann_lowrank_forward_cpu<'py>(`
- L27: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/rsulf.rs`
- L17: `pub fn rsulf_forward_cuda(`
- L38: `pub fn rsulf_batch_forward_cuda(`
- L60: `pub fn rsulf_unified_forward_cuda(`
- L86: `pub fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;`
- L87: `pub fn cudaFree(ptr: *mut c_void) -> i32;`
- L88: `pub fn cudaDeviceSynchronize() -> i32;`
- L92: `#[pyclass]`
- L93: `pub struct PyRSULFLayer {`
- L97: `#[pymethods]`
- L98: `impl PyRSULFLayer {`
- L101: `pub fn new(`
- L140: `pub fn new_with_metric(`
- L181: `pub fn new_with_basis(`
- L226: `pub fn forward<'py>(`
- L237: `pub fn param_count(&self) -> (usize, usize, f32) {`
- L242: `pub fn curvature(&self) -> f32 {`
- L247: `pub fn d_model(&self) -> usize {`
- L252: `pub fn r(&self) -> usize {`
- L257: `pub fn eta(&self) -> f32 {`
- L262: `pub fn alpha(&self) -> f32 {`
- L267: `pub fn beta(&self) -> f32 {`
- L272: `pub fn gamma(&self) -> f32 {`
- L277: `pub fn g_inv<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {`
- L282: `pub fn g_diag<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {`
- L286: `pub fn export_components<'py>(&self, py: Python<'py>) -> &'py PyDict {`
- L328: `pub fn from_components(`
- L384: `pub fn new_fast(`
- L423: `#[pyclass]`
- L424: `pub struct PyRiemannianDecoder {`
- L428: `#[pymethods]`
- L429: `impl PyRiemannianDecoder {`
- L431: `pub fn new(`
- L446: `pub fn from_lm_head(`
- L466: `pub fn forward<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> &'py PyArray2<f32> {`
- L472: `pub fn d_model(&self) -> usize {`
- L477: `pub fn rank(&self) -> usize {`
- L482: `pub fn vocab_size(&self) -> usize {`
- L487: `#[pyclass]`
- L488: `pub struct PyHumanDecoder {`
- L492: `#[pymethods]`
- L493: `impl PyHumanDecoder {`
- L507: `pub fn new(`
- L541: `pub fn decode(`
- L564: `pub fn rsulf_forward_cuda_py<'py>(`
- L752: `pub fn rsulf_batch_forward_cuda_py<'py>(`
- L945: `pub fn rsulf_unified_forward_cuda_py<'py>(`
- L1159: `#[pyfunction]`
- L1160: `pub fn fold_metric_svd<'py>(`
- L1181: `#[pyfunction]`
- L1182: `pub fn build_causal_laplacian<'py>(`
- L1191: `#[pyfunction]`
- L1192: `pub fn fold_ffn<'py>(`
- L1216: `#[pyfunction]`
- L1217: `pub fn verify_metric_consistency<'py>(`
- L1241: `#[pyfunction]`
- L1242: `pub fn fold_metric_optimized<'py>(`
- L1319: `#[pyfunction]`
- L1320: `pub fn nystrom_metric<'py>(`
- L1351: `pub fn analyze_layer_py<'py>(`
- L1383: `pub fn extract_global_basis_py<'py>(`
- L1401: `pub fn create_compression_plan_py<'py>(`
- L1485: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/spline.rs`
- L8: `#[pyfunction]`
- L9: `pub fn spline_interpolate_cuda(`
- L27: `#[pyfunction]`
- L28: `pub fn spline_forward_cuda(`
- L50: `#[pyfunction]`
- L51: `pub fn spline_backward_cuda(`
- L73: `pub fn register_spline_module(py: Python, parent_module: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/spline_cache.rs`
- L6: `pub struct PySplineCache {`
- L10: `#[pymethods]`
- L11: `impl PySplineCache {`
- L13: `pub fn new(curvature: f32, dimension: usize) -> Self {`
- L19: `pub fn add_point(`
- L29: `pub fn reconstruct<'py>(&self, py: Python<'py>, t: f32) -> Option<&'py PyArray1<f32>> {`
- L33: `pub fn batch_reconstruct<'py>(`
- L42: `pub fn clear(&mut self) {`
- L47: `pub fn register_spline_cache_module(py: Python, parent_module: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/suppression.rs`
- L5: `#[pyfunction]`
- L6: `pub fn compute_suppression_field<'py>(`
- L19: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/bindings/unified_riemannian.rs`
- L13: `#[pyclass]`
- L14: `pub struct PyUnifiedRiemannianLayer {`
- L18: `#[pymethods]`
- L19: `impl PyUnifiedRiemannianLayer {`
- L22: `fn new(`
- L35: `fn forward<'py>(`
- L63: `fn backward<'py>(`
- L83: `fn geodesic_path<'py>(`
- L98: `fn compute_energy<'py>(`
- L121: `fn flow_step<'py>(`
- L134: `fn update_value_function(`
- L150: `fn update_metric(`
- L163: `#[pyfunction]`
- L164: `pub fn compute_metric<'py>(`
- L190: `#[pyfunction]`
- L191: `pub fn geodesic_distance<'py>(`
- L218: `#[pyfunction]`
- L220: `pub fn geodesic_interpolate<'py>(`
- L253: `pub fn laplace_beltrami_matrix_py<'py>(`
- L283: `pub fn register(m: &PyModule) -> PyResult<()> {`

### `reality_stone/src/layers/bellman.rs`
- L15: `pub fn compute_diagonal_geodesic_update(input: &ArrayView2<f64>, dt: f64) -> Array2<f64> {`
- L49: `pub fn compute_diagonal_geodesic_backward(`

### `reality_stone/src/layers/bellman_lagrangian.rs`
- L10: `pub struct ValueFunction {`
- L16: `impl ValueFunction {`
- L17: `pub fn new(input_dim: usize, hidden_dim: usize) -> Self {`
- L29: `pub fn compute(&self, x: &ArrayView2<f32>) -> Array1<f32> {`
- L45: `pub fn gradient(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L71: `pub struct LagrangianParams {`
- L79: `pub struct RegularizationConfig {`
- L84: `impl Default for LagrangianParams {`
- L85: `fn default() -> Self {`
- L100: `pub fn bellman_potential(`
- L118: `pub fn kinetic_energy(`
- L132: `pub fn lagrangian(`
- L148: `pub fn representation_flow(`
- L170: `pub fn metric_flow(`
- L189: `pub fn bellman_update(`
- L252: `pub struct EnergyComponents {`
- L259: `impl EnergyComponents {`
- L260: `pub fn new(batch_size: usize) -> Self {`
- L271: `pub fn compute_energy_components(`
- L303: `fn test_value_function() {`
- L317: `fn test_kinetic_energy() {`

### `reality_stone/src/layers/decoder.rs`
- L4: `pub struct RiemannianDecoder {`
- L15: `impl RiemannianDecoder {`
- L16: `pub fn new(u: Array2<f32>, a: Array2<f32>, bt: Array2<f32>, bias: Array1<f32>) -> Self {`
- L33: `pub fn from_lm_head(`
- L66: `pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {`

### `reality_stone/src/layers/diffusion.rs`
- L11: `pub struct RiemannianDiffusion {`
- L17: `impl RiemannianDiffusion {`
- L18: `pub fn new(dim: usize, alpha: f32, dt: f32) -> Self {`
- L30: `pub fn step(`
- L63: `pub fn compute_flow(`

### `reality_stone/src/layers/geodesic.rs`
- L14: `pub fn exponential_map(`
- L46: `pub fn logarithmic_map(`
- L71: `fn exponential_map_generic(`
- L112: `fn logarithmic_map_generic(`
- L138: `pub fn geodesic_interpolation(`
- L160: `pub fn geodesic_path(`
- L179: `pub fn parallel_transport(`
- L214: `pub fn geodesic_distance(`
- L223: `fn is_at_origin(x: &ArrayView2<f32>) -> bool {`
- L234: `fn test_geodesic_interpolation_euclidean() {`
- L245: `fn test_geodesic_path() {`

### `reality_stone/src/layers/human_decoder.rs`
- L7: `pub struct StageWeights {`
- L13: `pub struct HumanStyleDecoder {`
- L24: `impl HumanStyleDecoder {`
- L25: `pub fn new(`
- L52: `fn masked_argmax(&self, logits: &ArrayView1<f32>, pool: &[usize]) -> Option<usize> {`
- L68: `fn select_topk(&self, logits: &ArrayView1<f32>, pool: &[usize], k: usize) -> Vec<usize> {`
- L81: `fn cosine_with_context(&self, idx: usize, context: &ArrayView1<f32>, ctx_norm: f32) -> f32 {`
- L87: `fn poincare_distance_single(&self, idx: usize, context: &ArrayView2<f32>) -> f32 {`
- L96: `fn euclidean_distance(&self, idx: usize, context: &ArrayView2<f32>) -> f32 {`
- L103: `fn select_relation(`
- L129: `fn select_object(`
- L158: `pub fn decode_batch(`

### `reality_stone/src/layers/hyper_metric.rs`
- L4: `pub struct TinyMLP {`
- L11: `impl TinyMLP {`
- L12: `pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {`
- L21: `pub fn from_weights(`
- L30: `pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {`
- L39: `pub struct HyperMetric {`
- L47: `impl HyperMetric {`
- L48: `pub fn new(d_model: usize, r: usize, hyper_hidden: usize) -> Self {`
- L61: `pub fn from_components(`
- L77: `pub fn generate_core(&self, layer_emb: &Array1<f32>) -> Array2<f32> {`
- L82: `pub fn project_forward(&self, x: &Array2<f32>, layer_emb: &Array1<f32>) -> Array2<f32> {`

### `reality_stone/src/layers/klein.rs`
- L5: `fn safe_sqrt(x: f32) -> f32 {`
- L10: `fn safe_acosh(x: f32) -> f32 {`
- L19: `pub fn klein_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {`
- L35: `pub fn klein_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L54: `pub fn klein_add_vjp(`
- L122: `pub fn klein_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {`
- L132: `pub fn klein_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L139: `pub fn klein_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L149: `pub fn klein_scalar_vjp(`
- L176: `pub fn klein_layer_forward(`
- L188: `pub fn klein_layer_backward(`
- L208: `pub fn klein_distance_cuda(`
- L216: `pub fn klein_layer_forward_cuda(`
- L225: `pub fn klein_layer_backward_cuda(`
- L239: `pub fn klein_distance_cuda(`
- L252: `pub fn klein_layer_forward_cuda(`
- L266: `pub fn klein_layer_backward_cuda(`
- L294: `pub fn to_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L306: `pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L314: `pub fn from_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`

### `reality_stone/src/layers/lorentz.rs`
- L8: `fn safe_sqrt(x: f32) -> f32 {`
- L13: `fn safe_acosh(x: f32) -> f32 {`
- L24: `pub fn lorentz_inner(u: &ArrayView2<f32>, v: &ArrayView2<f32>) -> Array1<f32> {`
- L50: `pub fn lorentz_exp0_space(u: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L86: `pub fn lorentz_exp0_space_backward(`
- L149: `pub fn lorentz_log0_space(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L174: `pub fn lorentz_log0_space_backward(`
- L218: `pub fn lorentz_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {`
- L227: `pub fn lorentz_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L270: `pub fn lorentz_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {`
- L310: `pub fn lorentz_scalar_backward(`
- L416: `pub fn lorentz_to_klein(x: &ArrayView2<f32>, _: f32) -> Array2<f32> {`
- L438: `pub fn lorentz_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L446: `pub fn lorentz_layer_forward(`
- L493: `pub fn lorentz_layer_backward(`
- L576: `fn acosh_derivative(z: f32) -> f32 {`
- L584: `pub fn lorentz_layer_dynamic(`
- L596: `pub fn lorentz_layer_dynamic_backward(`
- L660: `pub fn lorentz_layer_layerwise(`
- L673: `pub fn lorentz_layer_layerwise_backward(`
- L736: `pub fn lorentz_distance_cuda(`
- L744: `pub fn lorentz_layer_forward_cuda(`
- L753: `pub fn lorentz_layer_backward_cuda(`
- L767: `pub fn lorentz_distance_cuda(`
- L780: `pub fn lorentz_layer_forward_cuda(`
- L794: `pub fn lorentz_layer_backward_cuda(`
- L822: `pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L837: `pub fn from_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`

### `reality_stone/src/layers/memory.rs`
- L5: `pub struct ControlPoint {`
- L11: `pub struct GeodesicMemory {`
- L19: `impl GeodesicMemory {`
- L20: `pub fn new(d_model: usize, threshold: f32) -> Self {`
- L30: `pub fn push(&mut self, t: usize, x: ArrayView1<f32>) -> bool {`
- L62: `fn add_control_point(&mut self, t: usize) {`
- L83: `pub fn query(&self, t: f32) -> Array1<f32> {`
- L162: `pub fn get_compression_stats(&self) -> (usize, usize, f32) {`
- L184: `fn test_spline_compression_sine_wave() {`
- L235: `fn test_linear_trajectory_compression() {`

### `reality_stone/src/layers/metric.rs`
- L11: `pub trait MetricTensor: Send + Sync {`
- L13: `fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32>;`
- L16: `fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32>;`
- L19: `fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>>;`
- L22: `fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32>;`
- L25: `fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32>;`
- L28: `fn curvature(&self) -> f32;`
- L34: `pub struct DiagonalMetric {`
- L40: `impl DiagonalMetric {`
- L41: `pub fn new(dim: usize) -> Self {`
- L49: `fn compute_weights(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L61: `impl MetricTensor for DiagonalMetric {`
- L62: `fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L66: `fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L71: `fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {`
- L91: `fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {`
- L99: `fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {`
- L107: `fn curvature(&self) -> f32 {`
- L115: `pub struct PoincareMetric {`
- L119: `impl PoincareMetric {`
- L120: `pub fn new(curvature: f32) -> Self {`
- L124: `fn conformal_factor(&self, x: &ArrayView2<f32>) -> Array1<f32> {`
- L131: `impl MetricTensor for PoincareMetric {`
- L132: `fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L146: `fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L159: `fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {`
- L180: `fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {`
- L184: `fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {`
- L190: `fn curvature(&self) -> f32 {`
- L198: `pub struct LorentzMetric {`
- L202: `impl LorentzMetric {`
- L203: `pub fn new(curvature: f32) -> Self {`
- L208: `impl MetricTensor for LorentzMetric {`
- L209: `fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L224: `fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L229: `fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {`
- L236: `fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {`
- L240: `fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {`
- L244: `fn curvature(&self) -> f32 {`
- L251: `pub struct KleinMetric {`
- L255: `impl KleinMetric {`
- L256: `pub fn new(curvature: f32) -> Self {`
- L261: `impl MetricTensor for KleinMetric {`
- L262: `fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L281: `fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {`
- L286: `fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {`
- L307: `fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {`
- L311: `fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {`
- L319: `fn curvature(&self) -> f32 {`
- L326: `fn softplus(x: f32) -> f32 {`
- L335: `fn sigmoid(x: f32) -> f32 {`
- L340: `pub enum MetricType {`
- L347: `impl MetricType {`
- L348: `pub fn as_trait(&self) -> &dyn MetricTensor {`
- L357: `pub fn as_trait_mut(&mut self) -> &mut dyn MetricTensor {`

### `reality_stone/src/layers/poincare.rs`
- L17: `pub fn poincare_ball_layer_backward(`
- L41: `pub fn poincare_distance(`
- L72: `pub fn poincare_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L89: `pub fn poincare_to_klein(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L100: `pub fn poincare_exp_at(x: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, _: f32) -> Array2<f32> {`
- L129: `pub fn poincare_log_at(`
- L165: `pub fn poincare_ball_layer(`
- L177: `pub fn poincare_ball_layer_dynamic(`
- L191: `pub fn poincare_ball_layer_dynamic_backward(`
- L216: `pub fn poincare_ball_layer_layerwise(`
- L236: `pub fn poincare_ball_layer_layerwise_backward(`
- L279: `pub fn poincare_riemannian_adam_step(`
- L383: `pub fn poincare_distance_cuda(`
- L392: `pub fn poincare_ball_layer_cuda(`
- L401: `pub fn poincare_ball_layer_backward_cuda(`
- L415: `pub fn poincare_distance_cuda(`
- L429: `pub fn poincare_ball_layer_cuda(`
- L443: `pub fn poincare_ball_layer_backward_cuda(`
- L480: `fn test_mobius_add_identity() {`
- L489: `fn test_poincare_to_lorentz_and_back() {`
- L500: `fn test_poincare_ball_layer_interpolation() {`
- L520: `fn test_distance_is_zero_for_same_point() {`
- L532: `fn test_poincare_to_klein_then_back_shape_and_finiteness() {`
- L542: `fn test_riemannian_adam_matches_euclidean_when_c_zero() {`
- L576: `fn test_riemannian_adam_poincare_stays_inside_ball() {`

### `reality_stone/src/layers/riemann.rs`
- L6: `fn zeros_like(x: &ArrayView2<f32>) -> Array2<f32> {`
- L12: `pub fn riemann_lowrank_forward(`

### `reality_stone/src/layers/rsulf.rs`
- L5: `pub struct RSULFConfig {`
- L17: `impl Default for RSULFConfig {`
- L18: `fn default() -> Self {`
- L34: `pub struct GlobalBasis {`
- L39: `pub fn extract_global_basis(`
- L97: `pub struct FoldedMetric {`
- L106: `fn dense_svd(a: &Array2<f32>, k: usize) -> (Array2<f32>, Array1<f32>, Array2<f32>) {`
- L135: `fn orthonormalize_columns(mut y: Array2<f32>) -> Array2<f32> {`
- L185: `pub fn randomized_svd(`
- L234: `fn qr_decomposition(a: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {`
- L265: `pub fn fold_dimension_svd(`
- L310: `pub fn fold_dimension_diagonal(`
- L353: `pub fn fold_with_global_basis(`
- L401: `pub fn compute_curvature(s_residual: &Array1<f32>) -> f32 {`
- L407: `pub enum LayerType {`
- L417: `pub enum CompressionStrategy {`
- L431: `pub struct LayerAnalysis {`
- L445: `pub struct CompressionPlan {`
- L453: `pub fn analyze_weight_matrix(w: ArrayView2<f32>, max_rank: usize) -> (f32, f32, usize, f32) {`
- L507: `pub fn analyze_layer(`
- L558: `pub fn create_compression_plan(layer_analyses: Vec<LayerAnalysis>, _: f32) -> CompressionPlan {`
- L595: `pub fn verify_compression_plan(plan: &CompressionPlan, min_accuracy: f32) -> Result<(), String> {`
- L622: `pub fn create_causal_laplacian(seq_len: usize, window: usize) -> Array2<f32> {`
- L646: `pub struct FoldedFFN {`
- L655: `pub fn ffn_force_and_grad_row(`
- L679: `pub fn fold_ffn_svd(w1: ArrayView2<f32>, w2: ArrayView2<f32>, target_dim: usize) -> FoldedFFN {`
- L698: `pub fn fold_ffn_random_projection(`
- L774: `fn calibrate_eta_alpha(`
- L885: `pub struct RSULFLayer {`
- L900: `impl RSULFLayer {`
- L901: `pub fn from_transformer(`
- L970: `pub fn from_transformer_with_basis(`
- L1050: `pub fn from_transformer_with_metric(`
- L1124: `pub fn from_transformer_fast(`
- L1188: `pub fn forward(`
- L1452: `pub fn param_count(&self) -> (usize, usize, f32) {`
- L1471: `pub fn export_components(&self) -> RSULFComponents {`
- L1499: `pub fn from_components(comp: RSULFComponents) -> Self {`
- L1537: `pub struct RSULFComponents {`
- L1563: `pub struct FoldConsistencyResult {`
- L1572: `pub fn verify_fold_consistency(`
- L1658: `pub fn block_lanczos_svd(`
- L1721: `pub fn nystrom_approximation(`
- L1793: `pub fn adaptive_rank_svd(`

### `reality_stone/src/layers/spline.rs`
- L11: `#[pyclass]`
- L12: `pub struct SplineLayer {`
- L19: `#[pymethods]`
- L20: `impl SplineLayer {`
- L22: `pub fn new(k: usize, in_features: usize, out_features: usize) -> Self {`
- L35: `pub fn from_weight_py(`
- L47: `pub fn get_control_points<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {`
- L52: `pub fn set_control_points(&mut self, control_points: PyReadonlyArray2<f32>) -> PyResult<()> {`
- L57: `pub fn forward<'py>(`
- L68: `pub fn interpolate<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {`
- L72: `pub fn get_compression_ratio(&self) -> f32 {`
- L79: `impl SplineLayer {`
- L80: `pub fn from_weight(weight: &Array2<f32>, k: usize, learning_rate: f32, steps: usize) -> Self {`
- L117: `pub fn interpolate_internal(&self) -> Array2<f32> {`
- L144: `fn interpolate_with_grad(`
- L188: `struct CatmullRomGradients {`
- L199: `pub fn spline_interpolate_cuda(`
- L206: `pub fn spline_forward_cuda(`
- L215: `pub fn spline_backward_cuda(`
- L227: `pub fn spline_interpolate_cuda(`
- L239: `pub fn spline_forward_cuda(`
- L261: `pub fn spline_backward_cuda(`

### `reality_stone/src/layers/spline_cache.rs`
- L5: `pub struct ControlPoint {`
- L12: `pub struct SplineCache {`
- L18: `impl SplineCache {`
- L19: `pub fn new(curvature: f32, dimension: usize) -> Self {`
- L27: `pub fn add_point(&mut self, time: f32, state: ArrayView1<f32>, velocity: ArrayView1<f32>) {`
- L47: `pub fn reconstruct(&self, t: f32) -> Option<Array1<f32>> {`
- L131: `pub fn batch_reconstruct(&self, timestamps: ArrayView1<f32>) -> Array2<f32> {`
- L143: `pub fn clear(&mut self) {`
- L154: `fn test_spline_reconstruction() {`
- L176: `fn test_curvature_effect() {`

### `reality_stone/src/layers/suppression.rs`
- L5: `pub fn compute_dynamic_suppression(`

### `reality_stone/src/layers/symplectic.rs`
- L5: `pub struct SymplecticState {`
- L10: `impl SymplecticState {`
- L11: `pub fn new(batch_size: usize, d: usize) -> Self {`
- L19: `pub struct SymplecticLayer {`
- L26: `impl SymplecticLayer {`
- L27: `pub fn new(`
- L41: `pub fn step(&self, state: &mut SymplecticState, x_input: &Array2<f32>) -> Array2<f32> {`

### `reality_stone/src/layers/unified_riemannian.rs`
- L12: `pub struct UnifiedRiemannianLayer {`
- L20: `impl UnifiedRiemannianLayer {`
- L28: `pub fn new(metric_type: &str, curvature: f32, input_dim: usize, enable_bellman: bool) -> Self {`
- L62: `pub fn forward(&self, x: &ArrayView2<f32>, target: Option<&ArrayView2<f32>>) -> LayerOutput {`
- L121: `pub fn backward(`
- L141: `pub fn update_metric(&mut self, x: &ArrayView2<f32>, v: &ArrayView2<f32>, learning_rate: f32) {`
- L178: `pub fn compute_energy(`
- L209: `pub fn geodesic_path(`
- L227: `pub fn flow_step(`
- L251: `pub fn update_value_function(`
- L271: `pub fn laplace_beltrami_matrix(`
- L369: `pub struct LayerOutput {`
- L376: `pub struct LayerCache {`
- L383: `pub struct LayerGradients {`
- L390: `pub struct ValueFunctionGrad {`
- L401: `fn test_unified_layer_creation() {`
- L411: `fn test_forward_poincare() {`
- L421: `fn test_forward_with_target() {`
- L432: `fn test_geodesic_path() {`
- L444: `fn test_energy_computation() {`

### `reality_stone/src/ops/batch.rs`
- L6: `pub fn norm_sq_batched(x: &ArrayView2<f32>) -> Array1<f32> {`
- L10: `pub fn dot_batched(x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {`
- L14: `pub fn norm_sq_batched_f64(x: &ArrayView2<f64>) -> Array1<f64> {`
- L18: `pub fn dot_batched_f64(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array1<f64> {`

### `reality_stone/src/ops/curvature.rs`
- L3: `pub struct DynamicCurvature {`
- L9: `impl DynamicCurvature {`
- L10: `pub fn new(kappa: f32, c_min: f32, c_max: f32) -> Self {`
- L18: `pub fn compute_c(&self) -> f32 {`
- L23: `pub fn compute_dc_dkappa(&self) -> f32 {`
- L30: `pub struct LayerWiseDynamicCurvature {`
- L36: `impl LayerWiseDynamicCurvature {`
- L37: `pub fn new(num_layers: usize, c_min: f32, c_max: f32) -> Self {`
- L45: `pub fn from_kappas(kappas: Vec<f32>, c_min: f32, c_max: f32) -> Self {`
- L53: `pub fn compute_c(&self, layer_idx: usize) -> f32 {`
- L59: `pub fn compute_dc_dkappa(&self, layer_idx: usize) -> f32 {`

### `reality_stone/src/ops/extraction.rs`
- L5: `fn fast_extract_metric_cuda(`
- L18: `pub fn extract_metric_cuda(`
- L51: `pub fn extract_metric_cuda(`

### `reality_stone/src/ops/metrikey.rs`
- L8: `fn seed_from_key(key: &str) -> u64 {`
- L19: `fn box_muller_pair<R: Rng>(rng: &mut R) -> (f32, f32) {`
- L28: `fn box_muller_pair64<R: Rng>(rng: &mut R) -> (f64, f64) {`
- L36: `fn random_normal_matrix(dim: usize, rng: &mut SmallRng) -> Array2<f32> {`
- L55: `fn random_normal_matrix64(dim: usize, rng: &mut SmallRng) -> ndarray::Array2<f64> {`
- L74: `fn modified_gram_schmidt(a: &Array2<f32>, reorth_passes: usize) -> Array2<f32> {`
- L111: `fn modified_gram_schmidt64(a: &ndarray::Array2<f64>, reorth_passes: usize) -> ndarray::Array2<f64> {`
- L143: `pub fn deterministic_orthogonal_from_key(key: &str, dim: usize) -> Array2<f32> {`
- L151: `pub fn deterministic_orthogonal_from_key_f64(key: &str, dim: usize) -> ndarray::Array2<f64> {`
- L158: `pub fn spd_metric_from_key(key: &str, dim: usize, min_lambda: f32, max_lambda: f32) -> Array2<f32> {`
- L176: `pub fn spd_metric_from_key_f64(`
- L199: `pub fn spd_metric_from_key_weighted(`
- L222: `pub fn compose_layers_gravity(`
- L250: `pub fn compose_layers_gravity_f64(`
- L277: `pub fn apply_linear_f64(`
- L290: `pub fn compose_layers_gravity_compact_f64(`
- L320: `pub fn metric_factor_cholesky(g: &Array2<f32>) -> Array2<f32> {`
- L341: `pub fn mahalanobis_distance_sq_g(x: &Array1<f32>, y: &Array1<f32>, g: &Array2<f32>) -> f32 {`
- L350: `pub fn mahalanobis_distance_sq_l(x: &Array1<f32>, y: &Array1<f32>, l: &Array2<f32>) -> f32 {`
- L360: `pub fn block_orthogonal_from_key(key: &str, global_dim: usize, dept_dim: usize) -> Array2<f32> {`
- L370: `pub fn spd_block_metric_from_key(`
- L386: `pub fn compose_layers_order_preserving(layers: &[Array2<f32>]) -> Array2<f32> {`
- L399: `pub fn compose_layers_order_preserving_f64(`
- L413: `pub fn apply_linear(matrix: &Array2<f32>, vecs: &Array2<f32>) -> Array2<f32> {`
- L424: `pub fn layer_norm_forward_exact_f32(`
- L458: `pub fn gelu_new_f32(x: &Array2<f32>) -> Array2<f32> {`
- L475: `pub fn softmax_lastdim_f32(x: &Array2<f32>) -> Array2<f32> {`
- L502: `pub fn apply_causal_mask_inplace_f32(scores: &mut Array2<f32>, neg_large: f32) {`
- L513: `pub fn layer_norm_forward_exact_f64(`
- L549: `pub fn gelu_new_f64(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {`
- L564: `pub fn softmax_lastdim_f64(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {`
- L589: `pub fn apply_causal_mask_inplace_f64(scores: &mut ndarray::Array2<f64>, neg_large: f64) {`
- L601: `pub fn linear_f64(`
- L618: `pub fn attention_forward_f64(`
- L677: `pub fn ffn_gelu_forward_f64(`
- L689: `pub fn transformer_block_forward_f64(`
- L737: `pub fn effective_metric_from_transform_f64(t: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {`
- L742: `pub fn metric_factor_cholesky_f64(g: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {`
- L766: `pub fn rotate_metric_factor_block(key: &str, l: &Array2<f32>, global_dim: usize) -> Array2<f32> {`
- L777: `fn random_unit_vector_f32(dim: usize, rng: &mut SmallRng) -> Array1<f32> {`
- L786: `fn householder_vectors_from_key(key: &str, dim: usize, num: usize) -> Vec<Array1<f32>> {`
- L795: `fn apply_householder_chain(vecs: &[Array1<f32>], x: &Array1<f32>, reverse: bool) -> Array1<f32> {`
- L811: `pub fn householder_chain_apply_from_key(`
- L821: `pub fn householder_chain_apply_transpose_from_key(`
- L832: `pub fn lowrank_plus_diag_apply_from_key(`
- L853: `pub fn givens_chain_apply_from_key(`

### `reality_stone/src/ops/mobius.rs`
- L17: `pub fn mobius_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L31: `pub fn mobius_add_f64(u: &ArrayView2<f64>, v: &ArrayView2<f64>, c: f64) -> Array2<f64> {`
- L46: `pub fn mobius_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {`
- L79: `pub fn mobius_scalar_f64(u: &ArrayView2<f64>, c: f64, r: f64) -> Array2<f64> {`
- L103: `pub fn mobius_scalar_grad_c(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {`
- L154: `pub fn mobius_scalar_grad_c_f64(u: &ArrayView2<f64>, c: f64, r: f64) -> Array2<f64> {`
- L189: `pub fn mobius_add_grad_c(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {`
- L203: `pub fn mobius_add_grad_c_f64(u: &ArrayView2<f64>, v: &ArrayView2<f64>, c: f64) -> Array2<f64> {`
- L219: `pub fn mobius_scalar_vjp(`
- L265: `pub fn mobius_add_vjp(`
- L309: `pub fn mobius_add_dynamic(`
- L320: `pub fn mobius_add_dynamic_backward(`
- L336: `pub fn mobius_add_layerwise(`
- L348: `pub fn mobius_add_layerwise_backward(`
- L368: `pub fn mobius_add_cuda(`
- L376: `pub fn mobius_scalar_cuda(`
- L387: `pub fn mobius_add_cuda(`
- L400: `pub fn mobius_scalar_cuda(`

### `reality_stone/src/ops/mod.rs`
- L49: `pub fn mse_loss_grad(pred: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {`

### `reality_stone/src/ops/project.rs`
- L5: `pub fn project_to_ball(x: &ArrayView2<f32>, epsilon: f32) -> Array2<f32> {`

### `reality_stone/python/reality_stone/clarus/core/src/lib.rs`
- L22: `#[pyfunction]`
- L23: `fn topk_sparse(data: Vec<f64>, ratio: f64) -> (Vec<f64>, usize) {`
- L40: `#[pyfunction]`
- L41: `fn topk_sparse_batch(data: Vec<f64>, row_len: usize, ratio: f64) -> Vec<f64> {`
- L63: `#[pyfunction]`
- L64: `fn nn_topk_silu_fwd<'py>(`
- L75: `#[pyfunction]`
- L76: `fn nn_topk_silu_bwd<'py>(`
- L89: `#[pyfunction]`
- L90: `fn nn_lbo_fused_fwd<'py>(`
- L114: `#[pyfunction]`
- L115: `fn nn_power_iter<'py>(`
- L131: `#[pyfunction]`
- L133: `fn nn_gauge_lattice_fwd<'py>(`
- L164: `#[pyfunction]`
- L165: `fn nn_ce_pack_sparse<'py>(`
- L176: `#[pyfunction]`
- L177: `fn nn_ce_metric_basis_fwd<'py>(`
- L190: `#[pyfunction]`
- L191: `fn nn_ce_codebook_pull<'py>(`
- L206: `#[pyfunction]`
- L208: `fn nn_ce_relax_fwd<'py>(`
- L274: `#[pyfunction]`
- L276: `fn nn_brain_step<'py>(`
- L394: `#[pyfunction]`
- L396: `fn nn_ce_mfa_fwd<'py>(`
- L417: `#[pyfunction]`
- L419: `fn nn_ce_euler_fwd<'py>(`
- L445: `#[pyfunction]`
- L447: `fn nn_ce_riemann_fwd<'py>(`
- L475: `#[pyfunction]`
- L477: `fn nn_ce_riemann_fwd_cuda<'py>(`
- L509: `#[pyfunction]`
- L511: `fn nn_ce_riemann_fwd_cuda_devptr(`
- L534: `#[pyfunction]`
- L536: `fn nn_ce_dual_attn_fwd<'py>(`
- L560: `fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {`

### `reality_stone/python/reality_stone/clarus/core/src/cuda/mod.rs`
- L29: `fn ctx() -> Result<Arc<CudaContext>, String> {`
- L38: `fn module() -> Result<Arc<CudaModule>, String> {`
- L52: `fn shape_check(bh: usize, n: usize, d_head: usize) -> Result<(), String> {`
- L71: `fn launch_cfg(bh: usize, n: usize, d_head: usize) -> LaunchConfig {`
- L84: `pub fn ce_riemann_fwd_cuda(`

### `reality_stone/python/reality_stone/clarus/core/src/engine/ce_riemann.rs`
- L11: `pub struct RelaxOutput {`
- L23: `pub fn pack_sparse_csr(`
- L44: `fn csr_spmv(`
- L64: `pub fn codebook_pull(`
- L94: `pub fn metric_basis_from_codebook(`
- L154: `fn natural_direction(`
- L198: `fn fdt_noise(`
- L244: `fn symmetric_eigen(a: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {`
- L324: `fn solve_small_system(a: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {`
- L372: `fn normalize(v: &Array1<f32>) -> Array1<f32> {`
- L377: `fn norm(v: &Array1<f32>) -> f32 {`
- L381: `pub fn relax_forward(`

### `reality_stone/python/reality_stone/clarus/core/src/engine/config.rs`
- L2: `pub struct NoiseConfig {`
- L11: `impl Default for NoiseConfig {`
- L12: `fn default() -> Self {`
- L24: `impl NoiseConfig {`
- L25: `pub fn from_env_with_noise(noise_amp: f64) -> Self {`
- L38: `pub struct SuppressionConfig {`
- L46: `impl Default for SuppressionConfig {`
- L47: `fn default() -> Self {`
- L58: `impl SuppressionConfig {`
- L59: `pub fn from_env() -> Self {`
- L77: `pub fn has_any(&self) -> bool {`
- L81: `pub fn apply_to_trace(&self, trace: &mut [f64]) {`
- L94: `pub fn cancel_from_sample(&self, val: f64, t_abs: usize) -> f64 {`
- L107: `pub struct QecConfig {`
- L113: `impl Default for QecConfig {`
- L114: `fn default() -> Self {`
- L123: `impl QecConfig {`
- L124: `pub fn from_env() -> Self {`
- L133: `fn env_f64(key: &str, default: f64) -> f64 {`
- L140: `fn env_usize(key: &str, default: usize) -> usize {`
- L147: `fn env_i32(key: &str, default: i32) -> i32 {`

### `reality_stone/python/reality_stone/clarus/core/src/engine/constants.rs`
- L6: `fn lambert_w0(x: f64) -> f64 {`
- L95: `fn leptonic_running() -> f64 {`
- L108: `fn solve_alpha_s() -> f64 {`
- L153: `pub struct CeConstants {`
- L220: `impl CeConstants {`
- L221: `pub fn derive() -> Self {`
- L381: `pub fn print_all(&self) {`
- L444: `pub struct Discrepancy {`
- L451: `impl Discrepancy {`
- L452: `fn new(name: &'static str, predicted: f64, observed: f64) -> Self {`
- L467: `impl CeConstants {`
- L468: `pub fn verify(&self) -> Vec<Discrepancy> {`
- L497: `pub fn print_verification(&self) {`
- L529: `fn c() -> CeConstants {`
- L533: `fn assert_pct(name: &str, pred: f64, obs: f64, tol_pct: f64) {`
- L542: `fn layer1_alpha_s() {`
- L547: `fn layer1_sin2_theta_w() {`
- L552: `fn layer1_alpha_inv_0() {`
- L557: `fn layer1_sum() {`
- L564: `fn layer2_delta() {`
- L571: `fn layer2_d_eff() {`
- L576: `fn layer3_omega_b() {`
- L581: `fn layer3_omega_lambda() {`
- L586: `fn layer3_omega_dm() {`
- L591: `fn layer3_energy_conservation() {`
- L598: `fn layer4_higgs_mass() {`
- L603: `fn layer4_v_cb() {`
- L608: `fn layer4_jarlskog() {`
- L613: `fn layer5_theta13() {`
- L618: `fn layer5_theta12() {`
- L623: `fn layer5_theta23() {`
- L628: `fn layer6_m_p_over_m_e() {`
- L633: `fn layer6_m_d_over_m_u() {`
- L638: `fn layer6_koide() {`
- L643: `fn layer7_v_ew_over_m_pl() {`
- L649: `fn layer7_n_s() {`
- L654: `fn layer7_a_e() {`
- L659: `fn layer7_h0_t0() {`
- L664: `fn lambert_w0_basic() {`
- L670: `fn lambert_w0_small() {`

### `reality_stone/python/reality_stone/clarus/core/src/engine/field.rs`
- L8: `pub enum BoundaryMode {`
- L14: `pub struct FieldConfig {`
- L24: `impl Default for FieldConfig {`
- L25: `fn default() -> Self {`
- L38: `impl FieldConfig {`
- L39: `pub fn vacuum_vev(&self) -> f64 {`
- L45: `pub struct FieldState {`
- L51: `impl FieldState {`
- L52: `pub fn new_uniform(size: usize, vacuum_vev: f64) -> Self {`
- L60: `pub fn with_localized_source(`
- L78: `pub fn validate(&self) -> Result<(), String> {`
- L88: `pub struct FieldStepOutput {`
- L94: `pub struct FieldEngine {`
- L102: `impl FieldEngine {`
- L103: `pub fn new(config: FieldConfig, state: FieldState) -> Result<Self, String> {`
- L115: `pub fn with_size(size: usize, config: FieldConfig) -> Self {`
- L120: `pub fn state(&self) -> FieldState {`
- L129: `fn potential_force(phi_val: f64, mu: f64, lam: f64) -> f64 {`
- L134: `fn sample(phi_slice: &[f64], idx: isize, boundary: BoundaryMode) -> f64 {`
- L148: `pub fn step(&mut self) -> FieldStepOutput {`
- L230: `pub fn get_center_value(&self) -> f64 {`

### `reality_stone/python/reality_stone/clarus/core/src/engine/filter.rs`
- L4: `pub struct FilterFunction {`
- L9: `impl FilterFunction {`
- L10: `pub fn compute(pulse_times: &[f64], duration: f64, n_omega: usize) -> Self {`
- L59: `pub fn integrate_with_spectrum<F>(&self, spectrum: F) -> f64`
- L82: `pub fn compute_moment(&self, order: usize) -> f64 {`
- L100: `pub fn compute_gain_function(`
- L123: `pub fn generate_cpmg_sequence(n_pulses: usize) -> Vec<f64> {`
- L129: `pub fn generate_udd_sequence(n_pulses: usize) -> Vec<f64> {`

### `reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs`
- L9: `pub struct ModeParams {`
- L23: `impl ModeParams {`
- L25: `pub fn wake() -> Self {`
- L40: `pub fn nrem() -> Self {`
- L55: `pub fn rem() -> Self {`
- L70: `pub fn from_mode(mode: super::runtime_types::Mode) -> Self {`
- L81: `pub struct StpParams {`
- L87: `impl Default for StpParams {`
- L88: `fn default() -> Self {`
- L98: `pub struct StepConfig {`
- L114: `impl Default for StepConfig {`
- L115: `fn default() -> Self {`
- L135: `pub fn apply_dale_sign(`
- L160: `pub struct StepOutput {`
- L166: `fn spmv_masked(`
- L198: `pub fn brain_step(`
- L363: `fn make_identity_csr(dim: usize) -> (Vec<f32>, Vec<i32>, Vec<i32>) {`
- L377: `fn basic_step_runs() {`
- L407: `fn energy_decreases_nrem() {`
- L438: `fn adaptation_accumulates() {`
- L468: `fn stp_depletes_with_activity() {`

### `reality_stone/python/reality_stone/clarus/core/src/engine/manifold.rs`
- L6: `pub trait Manifold {`
- L8: `fn metric(&self, x: &[f64]) -> Array2<f64>;`
- L11: `fn christoffel(&self, x: &[f64]) -> Vec<Array2<f64>>;`
- L14: `fn ricci_scalar(&self, x: &[f64]) -> f64;`
- L18: `fn exp_map(&self, x: &[f64], v: &[f64], dt: f64) -> Vec<f64>;`
- L24: `pub struct SuppressionManifold {`
- L30: `impl SuppressionManifold {`
- L31: `pub fn new(phi_field: Box<PhiField>, dim: usize) -> Self {`
- L39: `fn gradient_phi(&self, x: &[f64]) -> Vec<f64> {`
- L54: `fn laplacian_phi(&self, x: &[f64]) -> f64 {`
- L71: `impl Manifold for SuppressionManifold {`
- L72: `fn metric(&self, x: &[f64]) -> Array2<f64> {`
- L83: `fn christoffel(&self, x: &[f64]) -> Vec<Array2<f64>> {`
- L107: `fn ricci_scalar(&self, x: &[f64]) -> f64 {`
- L125: `fn exp_map(&self, x: &[f64], v: &[f64], dt: f64) -> Vec<f64> {`

### `reality_stone/python/reality_stone/clarus/core/src/engine/nn_ops.rs`
- L14: `fn silu_f32(x: f32) -> f32 {`
- L19: `fn sigmoid_f32(x: f32) -> f32 {`
- L29: `pub fn topk_silu_fwd(input: &[f32], dim: usize, ratio: f32) -> (Vec<f32>, Vec<u8>) {`
- L76: `pub fn topk_silu_bwd(grad: &[f32], input: &[f32], mask: &[u8], dim: usize) -> Vec<f32> {`
- L101: `pub fn lbo_fused_fwd(`
- L148: `pub fn power_iter_step(`
- L177: `fn channel_fwd(`
- L209: `pub fn gauge_lattice_fwd(`
- L288: `fn dot_f32(a: &[f32], b: &[f32]) -> f32 {`
- L297: `fn sq_dist_f32(a: &[f32], b: &[f32]) -> f32 {`
- L309: `pub fn ce_mfa_fwd(`
- L406: `pub fn ce_dual_attn_fwd(`
- L552: `pub fn ce_riemann_fwd(`
- L659: `pub fn ce_euler_fwd(`

### `reality_stone/python/reality_stone/clarus/core/src/engine/runtime_types.rs`
- L6: `pub enum Mode {`
- L13: `pub struct CellState {`
- L23: `impl Default for CellState {`
- L24: `fn default() -> Self {`
- L38: `pub struct RelaxInput {`
- L53: `pub struct RelaxOutput {`
- L62: `pub struct SnapshotMeta {`
