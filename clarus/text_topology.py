"""Topology-first text engine built on Euler-style local propagation.

This module treats text as a layered cell complex:

  - words      -> 0-cells   (V)
  - sentences  -> 1-cells   (E) connecting word-level paths and co-occurrences
  - paragraphs -> 2-cells   (F) filling triangles in the sentence graph

Euler characteristic is therefore the standard chi = V - E + F.

The five-constant basis (1, pi, e, pi*e, pi/e) acts as masked role weights
for local propagation rather than as a fixed normalised vector, so the
dominant role responds to the input rather than collapsing to a constant.

On top of that, the Boolean-Spectral Carrier Theorem (graph.md sec 10.6)
adds a P-axis carrier U_P(theta) = exp(i theta f(L_G)) on the cell states,
realised here as a RoPE-style rotation on adjacent dim pairs of the
sentence/paragraph state vectors. The pi/i constants of the Euler basis
become the rotation generator; the e constant becomes the additive
distance decay applied downstream by HippocampusMemory.

The implementation intentionally avoids zeta-specific machinery and runs
without third-party dependencies; the same operators map onto the rest of
the Clarus Euler stack when it is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import random
import re


_WORD_RE = re.compile(r"[A-Za-z0-9가-힣_]+(?:'[A-Za-z0-9가-힣_]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")

# Five-constant role basis. Each weight is paired with a propagation role:
#   1     -> base self          (always active)
#   pi    -> previous neighbour (active when idx > 0)
#   e     -> next neighbour     (active when idx + 1 < length)
#   pi*e  -> position drift     (active when sentence has >= 2 tokens)
#   pi/e  -> sentence anchor    (always active)
_EULER_BASIS = (1.0, math.pi, math.e, math.pi * math.e, math.pi / math.e)
_EULER_BASIS_NAMES = ("self", "prev", "next", "drift", "anchor")

# RoPE-style P-carrier base. Mirrors `clarus.ce_euler.ce_rotary_base` with
# block=1 so the slowest mode stays near DC for short cell sequences.
# D_eff = 3 + sin^2 theta_W (1 - sin^2 theta_W) ~ 3.178 (CE physics layer).
_ALPHA_S = 0.11789
_SIN2 = 4.0 * _ALPHA_S ** (4.0 / 3.0)
_D_EFF = 3.0 + _SIN2 * (1.0 - _SIN2)
_P_CARRIER_BASE = math.pi ** _D_EFF


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in _PARA_SPLIT_RE.split(text) if part.strip()]
    return parts if parts else [text.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    parts = [part.strip() for part in _SENT_SPLIT_RE.split(paragraph) if part.strip()]
    return parts if parts else [paragraph.strip()]


def _tokenize(sentence: str) -> list[str]:
    tokens = _WORD_RE.findall(sentence.lower())
    return tokens if tokens else [sentence.strip().lower()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _scale(a: list[float], value: float) -> list[float]:
    return [value * x for x in a]


def _add(*rows: list[float]) -> list[float]:
    if not rows:
        return []
    out = [0.0] * len(rows[0])
    for row in rows:
        for idx, value in enumerate(row):
            out[idx] += value
    return out


def _mean(rows: list[list[float]], dim: int) -> list[float]:
    if not rows:
        return [0.0] * dim
    inv = 1.0 / len(rows)
    return [inv * sum(row[idx] for row in rows) for idx in range(dim)]


def _tanh_vec(a: list[float]) -> list[float]:
    return [math.tanh(x) for x in a]


def _unit_from_text(text: str, dim: int) -> list[float]:
    rng = random.Random(_stable_seed(text))
    row = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    nrm = _norm(row)
    if nrm <= 1e-12:
        row[0] = 1.0
        return row
    return [x / nrm for x in row]


def _cosine_adjacency(
    rows: list[list[float]], *, signed: bool = False
) -> list[list[float]]:
    n = len(rows)
    adj = [[0.0] * n for _ in range(n)]
    norms = [_norm(row) for row in rows]
    for i in range(n):
        for j in range(i + 1, n):
            denom = max(norms[i] * norms[j], 1e-8)
            cos = _dot(rows[i], rows[j]) / denom
            if not signed and cos < 0.0:
                cos = 0.0
            adj[i][j] = cos
            adj[j][i] = cos
    return adj


def _phase_carrier_states(
    states: list[list[float]], dim: int, *, base: float = _P_CARRIER_BASE
) -> list[list[float]]:
    """Apply U_P(theta) = exp(i theta) on adjacent dim pairs of each state.

    theta(pos, k) = pos * base ** (-2k / dim). The pi^D_eff base reproduces
    the CE-faithful rotary frequency from `clarus.ce_euler.ce_rotary_base`
    with block = 1, so a single carrier step encodes the position channel
    (rho axis in graph.md sec 10.6, case (a)) without requiring a transformer.
    """
    if not states or dim < 2:
        return [list(row) for row in states]
    half = dim // 2
    inv_freq = [base ** (-(2.0 * k) / dim) for k in range(half)]
    rotated: list[list[float]] = []
    for pos, row in enumerate(states):
        out = list(row)
        for k in range(half):
            theta = pos * inv_freq[k]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            i_re = 2 * k
            i_im = i_re + 1
            if i_im >= len(out):
                break
            x_re = out[i_re]
            x_im = out[i_im]
            out[i_re] = x_re * cos_t - x_im * sin_t
            out[i_im] = x_re * sin_t + x_im * cos_t
        rotated.append(out)
    return rotated


def _phase_carrier_metrics(
    raw_states: list[list[float]], rotated_states: list[list[float]], dim: int
) -> tuple[float, float]:
    """Quantify the action of U_P on the cell complex.

    Returns (alignment, decay) where:
      * alignment = mean cosine between consecutive rotated states. Captures
        how well the P-carrier preserves local coherence (high = phase locks).
      * decay     = mean drop in cosine between (raw_i, raw_j) and
        (rotated_i, rotated_j) for adjacent (i, j). This is the empirical
        ALiBi-style attenuation induced purely by the rotation -- the same
        Tier 1 signature the EulerCEMinimal paper observed for length-OOD.
    """
    n = len(rotated_states)
    if n < 2:
        return 0.0, 0.0
    align_total = 0.0
    decay_total = 0.0
    for idx in range(n - 1):
        rot_a = rotated_states[idx]
        rot_b = rotated_states[idx + 1]
        denom_rot = max(_norm(rot_a) * _norm(rot_b), 1e-8)
        rot_cos = _dot(rot_a, rot_b) / denom_rot
        align_total += rot_cos

        raw_a = raw_states[idx]
        raw_b = raw_states[idx + 1]
        denom_raw = max(_norm(raw_a) * _norm(raw_b), 1e-8)
        raw_cos = _dot(raw_a, raw_b) / denom_raw
        decay_total += max(raw_cos - rot_cos, 0.0)
    pairs = float(n - 1)
    return align_total / pairs, decay_total / pairs


def _edge_count(adj: list[list[float]], threshold: float) -> int:
    n = len(adj)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] > threshold:
                total += 1
    return total


def _edge_density(node_count: int, edge_count: int) -> float:
    if node_count <= 1:
        return 0.0
    return edge_count / max(node_count * (node_count - 1) / 2.0, 1.0)


def _triangle_count(adj: list[list[float]], threshold: float) -> int:
    """Count 2-cells (triangles) above the connection threshold."""
    n = len(adj)
    if n < 3:
        return 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] <= threshold:
                continue
            for k in range(j + 1, n):
                if adj[i][k] > threshold and adj[j][k] > threshold:
                    total += 1
    return total


def _algebraic_connectivity(adj: list[list[float]], threshold: float) -> float:
    """Power-iteration estimate of the Fiedler value of L = D - A.

    Returns 0.0 when the graph is disconnected (or trivial). The estimate
    deflates the constant eigenvector at every step so we recover the
    second-smallest eigenvalue of the symmetric Laplacian.
    """
    n = len(adj)
    if n < 2:
        return 0.0
    weighted = [[w if w > threshold else 0.0 for w in row] for row in adj]
    degrees = [sum(row) for row in weighted]
    if max(degrees) <= 0.0:
        return 0.0
    lam_max = 2.0 * max(degrees) + 1e-6

    rng = random.Random(0xA17BA)
    vec = [rng.gauss(0.0, 1.0) for _ in range(n)]
    mean_v = sum(vec) / n
    vec = [x - mean_v for x in vec]
    norm_v = _norm(vec) or 1.0
    vec = [x / norm_v for x in vec]

    for _ in range(96):
        # Multiply by (lam_max * I - L) to push the smallest non-trivial
        # eigenvalue to the top of the spectrum.
        out = [0.0] * n
        for i in range(n):
            row = weighted[i]
            acc = (lam_max - degrees[i]) * vec[i]
            for j in range(n):
                if row[j] != 0.0:
                    acc += row[j] * vec[j]
            out[i] = acc
        mean_o = sum(out) / n
        out = [x - mean_o for x in out]
        norm_o = _norm(out)
        if norm_o <= 1e-12:
            return 0.0
        vec = [x / norm_o for x in out]

    # Rayleigh quotient v^T L v / v^T v with v already unit-norm.
    rayleigh = 0.0
    for i in range(n):
        acc = degrees[i] * vec[i]
        row = weighted[i]
        for j in range(n):
            if row[j] != 0.0:
                acc -= row[j] * vec[j]
        rayleigh += vec[i] * acc
    return max(rayleigh, 0.0)


def _layer_alignment(left: list[list[float]], right: list[list[float]], dim: int) -> float:
    if not left or not right:
        return 0.0
    a = _mean(left, dim)
    b = _mean(right, dim)
    denom = max(_norm(a) * _norm(b), 1e-8)
    return _dot(a, b) / denom


def _cross_layer_bridge(
    fine: list[list[float]],
    coarse: list[list[float]],
    membership: list[int],
) -> float:
    """Mean cosine between each fine cell and its enclosing coarse cell."""
    if not fine or not coarse or len(fine) != len(membership):
        return 0.0
    fine_norms = [_norm(row) for row in fine]
    coarse_norms = [_norm(row) for row in coarse]
    total = 0.0
    counted = 0
    for idx, parent in enumerate(membership):
        if parent < 0 or parent >= len(coarse):
            continue
        denom = max(fine_norms[idx] * coarse_norms[parent], 1e-8)
        total += _dot(fine[idx], coarse[parent]) / denom
        counted += 1
    if counted == 0:
        return 0.0
    return total / counted


def _top_links(adj: list[list[float]], threshold: float, limit: int = 5) -> list[LinkSummary]:
    links: list[LinkSummary] = []
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            weight = adj[i][j]
            if weight > threshold:
                links.append(LinkSummary(source=i, target=j, weight=weight))
    links.sort(key=lambda item: item.weight, reverse=True)
    return links[:limit]


def _connected_components(adj: list[list[float]], threshold: float) -> int:
    n = len(adj)
    if n == 0:
        return 0
    seen = [False] * n
    comps = 0
    for start in range(n):
        if seen[start]:
            continue
        comps += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nxt, weight in enumerate(adj[node]):
                if weight > threshold and not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
    return comps


@dataclass
class TopologyLevelSummary:
    count: int
    edges: int
    components: int
    euler_characteristic: int
    edge_density: float
    faces: int = 0
    algebraic_connectivity: float = 0.0
    phase_carrier_alignment: float = 0.0
    phase_carrier_decay: float = 0.0


@dataclass
class LinkSummary:
    source: int
    target: int
    weight: float


@dataclass
class TextTopologyResult:
    token_summary: TopologyLevelSummary
    sentence_summary: TopologyLevelSummary
    paragraph_summary: TopologyLevelSummary
    token_state_norm: float
    sentence_state_norm: float
    paragraph_state_norm: float
    bridge_energy: float
    token_sentence_alignment: float
    sentence_paragraph_alignment: float
    dominant_euler_basis: str
    sentence_links: list[LinkSummary]
    paragraph_links: list[LinkSummary]
    paragraphs: list[str]
    sentences: list[str]
    token_sentence_bridge: float = 0.0
    sentence_paragraph_bridge: float = 0.0
    euler_basis_activation: dict[str, float] = field(default_factory=dict)
    phase_carrier_alignment: float = 0.0
    phase_carrier_decay: float = 0.0


class TextTopologyEngine:
    """Prototype topology engine over word/sentence/paragraph layers."""

    def __init__(
        self,
        dim: int = 32,
        sentence_threshold: float = 0.30,
        paragraph_threshold: float = 0.22,
    ) -> None:
        self.dim = int(dim)
        self.sentence_threshold = float(sentence_threshold)
        self.paragraph_threshold = float(paragraph_threshold)
        # Raw five-constant weights; masking + per-input renormalisation
        # happens inside `_token_states`, so the weights stay meaningful as
        # role amplitudes rather than as a fixed constant vector.
        self.euler_basis = list(_EULER_BASIS)

    def _collect_hierarchy(
        self, text: str
    ) -> tuple[list[str], list[list[str]], list[list[list[str]]]]:
        paragraphs = _split_paragraphs(text)
        para_sentences: list[list[str]] = []
        para_tokens: list[list[list[str]]] = []
        for paragraph in paragraphs:
            sentences = _split_sentences(paragraph)
            para_sentences.append(sentences)
            para_tokens.append([_tokenize(sentence) for sentence in sentences])
        return paragraphs, para_sentences, para_tokens

    def _token_states(
        self, token_lists: list[list[str]]
    ) -> tuple[list[str], list[list[float]], list[float]]:
        """Compute token states and accumulate per-role activation mass.

        Returns the activation totals so we can name the dominant role for
        the input rather than collapsing onto a fixed constant.
        """
        all_tokens: list[str] = []
        token_states: list[list[float]] = []
        activation = [0.0] * len(self.euler_basis)
        for tokens in token_lists:
            length = len(tokens)
            for idx, token in enumerate(tokens):
                base = _unit_from_text(token, self.dim)
                prev_vec = _unit_from_text(f"{token}:prev", self.dim)
                next_vec = _unit_from_text(f"{token}:next", self.dim)
                pos_vec = _unit_from_text(f"{token}:pos", self.dim)
                sent_vec = _unit_from_text(f"{token}:sent", self.dim)

                drift_amp = idx / max(length - 1, 1) if length > 1 else 0.0
                anchor_amp = 1.0 / max(length, 1)
                mask = (
                    1.0,
                    1.0 if idx > 0 else 0.0,
                    1.0 if idx + 1 < length else 0.0,
                    drift_amp,
                    anchor_amp,
                )
                raw = [w * m for w, m in zip(self.euler_basis, mask)]
                total = sum(raw)
                if total <= 0.0:
                    weights = [0.0] * len(raw)
                    weights[0] = 1.0
                else:
                    weights = [r / total for r in raw]
                for i, w in enumerate(weights):
                    activation[i] += w

                local = _add(
                    _scale(base, weights[0]),
                    _scale(prev_vec, weights[1]),
                    _scale(next_vec, weights[2]),
                    _scale(pos_vec, weights[3]),
                    _scale(sent_vec, weights[4]),
                )
                state = _tanh_vec(_add(base, local))
                all_tokens.append(token)
                token_states.append(state)
        return all_tokens, token_states, activation

    def analyze(self, text: str) -> TextTopologyResult:
        paragraphs, para_sentences, para_tokens = self._collect_hierarchy(text)
        flat_token_lists = [tokens for sentence_tokens in para_tokens for tokens in sentence_tokens]
        flat_sentences = [sentence for sentences in para_sentences for sentence in sentences]
        sentence_to_paragraph: list[int] = []
        for para_idx, sentences in enumerate(para_sentences):
            sentence_to_paragraph.extend([para_idx] * len(sentences))

        token_texts, token_states, activation = self._token_states(flat_token_lists)

        sentence_states: list[list[float]] = []
        token_to_sentence: list[int] = []
        cursor = 0
        for sent_idx, (token_list, sentence) in enumerate(zip(flat_token_lists, flat_sentences)):
            rows = token_states[cursor: cursor + len(token_list)]
            token_to_sentence.extend([sent_idx] * len(token_list))
            cursor += len(token_list)
            sent_embed = _mean(rows, self.dim)
            sentence_states.append(
                _tanh_vec(_add(sent_embed, _scale(_unit_from_text(sentence, self.dim), 0.25)))
            )

        paragraph_states: list[list[float]] = []
        sent_cursor = 0
        for sentences, paragraph in zip(para_sentences, paragraphs):
            rows = sentence_states[sent_cursor: sent_cursor + len(sentences)]
            sent_cursor += len(sentences)
            para_embed = _mean(rows, self.dim)
            paragraph_states.append(
                _tanh_vec(_add(para_embed, _scale(_unit_from_text(paragraph, self.dim), 0.15)))
            )

        token_count = len(token_texts)
        # Token graph = path edges within each sentence + co-occurrence edges
        # for tokens that repeat across the corpus. This breaks the trivial
        # forest identity chi = #components and lets chi report cycles.
        path_edges = sum(max(len(tokens) - 1, 0) for tokens in flat_token_lists)
        positions: dict[str, list[int]] = {}
        for idx, token in enumerate(token_texts):
            positions.setdefault(token, []).append(idx)
        cooccurrence_edges = sum(len(p) - 1 for p in positions.values() if len(p) > 1)
        token_edges = path_edges + cooccurrence_edges
        token_components = self._token_component_count(token_count, token_to_sentence, positions)
        token_summary = TopologyLevelSummary(
            count=token_count,
            edges=token_edges,
            components=token_components,
            euler_characteristic=token_count - token_edges,
            edge_density=_edge_density(token_count, token_edges),
        )

        sentence_rotated = _phase_carrier_states(sentence_states, self.dim)
        paragraph_rotated = _phase_carrier_states(paragraph_states, self.dim)
        sentence_phase_align, sentence_phase_decay = _phase_carrier_metrics(
            sentence_states, sentence_rotated, self.dim
        )
        paragraph_phase_align, paragraph_phase_decay = _phase_carrier_metrics(
            paragraph_states, paragraph_rotated, self.dim
        )

        sentence_adj = _cosine_adjacency(sentence_states)
        paragraph_adj = _cosine_adjacency(paragraph_states)
        sentence_edges = _edge_count(sentence_adj, self.sentence_threshold)
        paragraph_edges = _edge_count(paragraph_adj, self.paragraph_threshold)

        # Paragraphs act as 2-cells filling triangles within their own
        # sentence subgraph, so chi_sentence = V - E + F is well defined.
        sentence_faces = self._faces_within_paragraphs(
            sentence_adj, sentence_to_paragraph, self.sentence_threshold
        )
        paragraph_faces = _triangle_count(paragraph_adj, self.paragraph_threshold)

        sentence_components = _connected_components(sentence_adj, self.sentence_threshold)
        paragraph_components = _connected_components(paragraph_adj, self.paragraph_threshold)

        sentence_summary = TopologyLevelSummary(
            count=len(flat_sentences),
            edges=sentence_edges,
            components=sentence_components,
            euler_characteristic=len(flat_sentences) - sentence_edges + sentence_faces,
            edge_density=_edge_density(len(flat_sentences), sentence_edges),
            faces=sentence_faces,
            algebraic_connectivity=_algebraic_connectivity(sentence_adj, self.sentence_threshold),
            phase_carrier_alignment=float(sentence_phase_align),
            phase_carrier_decay=float(sentence_phase_decay),
        )
        paragraph_summary = TopologyLevelSummary(
            count=len(paragraphs),
            edges=paragraph_edges,
            components=paragraph_components,
            euler_characteristic=len(paragraphs) - paragraph_edges + paragraph_faces,
            edge_density=_edge_density(len(paragraphs), paragraph_edges),
            faces=paragraph_faces,
            algebraic_connectivity=_algebraic_connectivity(paragraph_adj, self.paragraph_threshold),
            phase_carrier_alignment=float(paragraph_phase_align),
            phase_carrier_decay=float(paragraph_phase_decay),
        )

        token_state_norm = 0.0 if not token_states else sum(_norm(row) for row in token_states) / len(token_states)
        sentence_state_norm = 0.0 if not sentence_states else sum(_norm(row) for row in sentence_states) / len(sentence_states)
        paragraph_state_norm = 0.0 if not paragraph_states else sum(_norm(row) for row in paragraph_states) / len(paragraph_states)

        token_sentence_alignment = _layer_alignment(token_states, sentence_states, self.dim)
        sentence_paragraph_alignment = _layer_alignment(sentence_states, paragraph_states, self.dim)

        token_sentence_bridge = _cross_layer_bridge(token_states, sentence_states, token_to_sentence)
        sentence_paragraph_bridge = _cross_layer_bridge(
            sentence_states, paragraph_states, sentence_to_paragraph
        )
        # bridge_energy is the geometric mean of the two cross-layer bridges
        # (so it is genuinely distinct from sentence_paragraph_alignment).
        bridge_energy = math.sqrt(
            max(token_sentence_bridge, 0.0) * max(sentence_paragraph_bridge, 0.0)
        )

        activation_total = sum(activation)
        if activation_total > 0.0:
            normalised_activation = [a / activation_total for a in activation]
        else:
            normalised_activation = [0.0] * len(self.euler_basis)
            normalised_activation[0] = 1.0
        dominant_idx = max(range(len(normalised_activation)), key=normalised_activation.__getitem__)
        activation_map = {
            name: float(value)
            for name, value in zip(_EULER_BASIS_NAMES, normalised_activation)
        }

        return TextTopologyResult(
            token_summary=token_summary,
            sentence_summary=sentence_summary,
            paragraph_summary=paragraph_summary,
            token_state_norm=float(token_state_norm),
            sentence_state_norm=float(sentence_state_norm),
            paragraph_state_norm=float(paragraph_state_norm),
            bridge_energy=float(bridge_energy),
            token_sentence_alignment=float(token_sentence_alignment),
            sentence_paragraph_alignment=float(sentence_paragraph_alignment),
            dominant_euler_basis=_EULER_BASIS_NAMES[dominant_idx],
            sentence_links=_top_links(sentence_adj, self.sentence_threshold),
            paragraph_links=_top_links(paragraph_adj, self.paragraph_threshold),
            paragraphs=paragraphs,
            sentences=flat_sentences,
            token_sentence_bridge=float(token_sentence_bridge),
            sentence_paragraph_bridge=float(sentence_paragraph_bridge),
            euler_basis_activation=activation_map,
            phase_carrier_alignment=float(
                0.5 * (sentence_phase_align + paragraph_phase_align)
            ),
            phase_carrier_decay=float(
                0.5 * (sentence_phase_decay + paragraph_phase_decay)
            ),
        )

    @staticmethod
    def _faces_within_paragraphs(
        sentence_adj: list[list[float]],
        sentence_to_paragraph: list[int],
        threshold: float,
    ) -> int:
        """Count triangles in the sentence graph that lie inside one paragraph."""
        if not sentence_adj:
            return 0
        groups: dict[int, list[int]] = {}
        for idx, parent in enumerate(sentence_to_paragraph):
            groups.setdefault(parent, []).append(idx)
        total = 0
        for indices in groups.values():
            n = len(indices)
            if n < 3:
                continue
            for a in range(n):
                ia = indices[a]
                for b in range(a + 1, n):
                    ib = indices[b]
                    if sentence_adj[ia][ib] <= threshold:
                        continue
                    for c in range(b + 1, n):
                        ic = indices[c]
                        if (
                            sentence_adj[ia][ic] > threshold
                            and sentence_adj[ib][ic] > threshold
                        ):
                            total += 1
        return total

    @staticmethod
    def _token_component_count(
        token_count: int,
        token_to_sentence: list[int],
        positions: dict[str, list[int]],
    ) -> int:
        """Union-find over per-sentence path edges + co-occurrence edges."""
        if token_count == 0:
            return 0
        parent = list(range(token_count))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        prev_idx = -1
        prev_sentence = -1
        for idx, sentence in enumerate(token_to_sentence):
            if prev_idx >= 0 and sentence == prev_sentence:
                union(prev_idx, idx)
            prev_idx = idx
            prev_sentence = sentence

        for indices in positions.values():
            if len(indices) < 2:
                continue
            anchor = indices[0]
            for other in indices[1:]:
                union(anchor, other)

        roots = {find(i) for i in range(token_count)}
        return len(roots)


__all__ = [
    "TextTopologyEngine",
    "TextTopologyResult",
    "TopologyLevelSummary",
]
