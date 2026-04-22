"""Topology-first text engine built on Euler-style local propagation.

This module treats text as a layered cell complex:

  - words      -> vertices
  - sentences  -> paths / local graphs
  - paragraphs -> higher-order cells

The implementation intentionally avoids zeta-specific machinery and can
run without third-party dependencies. When the rest of the project is
available, its Euler constants and graph ideas still map directly onto
this prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import re


_WORD_RE = re.compile(r"[A-Za-z0-9가-힣_]+(?:'[A-Za-z0-9가-힣_]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
_EULER_BASIS = (1.0, math.pi, math.e, math.pi * math.e, math.pi / math.e)
_EULER_BASIS_NAMES = ("1", "pi", "e", "pi*e", "pi/e")


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


def _cosine_adjacency(rows: list[list[float]]) -> list[list[float]]:
    n = len(rows)
    adj = [[0.0] * n for _ in range(n)]
    norms = [_norm(row) for row in rows]
    for i in range(n):
        for j in range(i + 1, n):
            denom = max(norms[i] * norms[j], 1e-8)
            cos = max(_dot(rows[i], rows[j]) / denom, 0.0)
            adj[i][j] = cos
            adj[j][i] = cos
    return adj


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
        total = sum(_EULER_BASIS)
        self.euler_basis = [value / total for value in _EULER_BASIS]

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
    ) -> tuple[list[str], list[list[float]]]:
        all_tokens: list[str] = []
        token_states: list[list[float]] = []
        for tokens in token_lists:
            length = len(tokens)
            for idx, token in enumerate(tokens):
                base = _unit_from_text(token, self.dim)
                prev_vec = _unit_from_text(f"{token}:prev", self.dim)
                next_vec = _unit_from_text(f"{token}:next", self.dim)
                pos_vec = _unit_from_text(f"{token}:pos", self.dim)
                sent_vec = _unit_from_text(f"{token}:sent", self.dim)
                local = _add(
                    _scale(base, self.euler_basis[0]),
                    _scale(prev_vec, self.euler_basis[1] * (1.0 if idx > 0 else 0.0)),
                    _scale(next_vec, self.euler_basis[2] * (1.0 if idx + 1 < length else 0.0)),
                    _scale(pos_vec, self.euler_basis[3] * (idx / max(length - 1, 1))),
                    _scale(sent_vec, self.euler_basis[4] / max(length, 1)),
                )
                state = _tanh_vec(_add(base, local))
                all_tokens.append(token)
                token_states.append(state)
        return all_tokens, token_states

    def analyze(self, text: str) -> TextTopologyResult:
        paragraphs, para_sentences, para_tokens = self._collect_hierarchy(text)
        flat_token_lists = [tokens for sentence_tokens in para_tokens for tokens in sentence_tokens]
        flat_sentences = [sentence for sentences in para_sentences for sentence in sentences]

        token_texts, token_states = self._token_states(flat_token_lists)

        sentence_states: list[list[float]] = []
        cursor = 0
        for token_list, sentence in zip(flat_token_lists, flat_sentences):
            rows = token_states[cursor: cursor + len(token_list)]
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
        token_edges = max(token_count - len(flat_sentences), 0)
        token_components = len(flat_sentences)
        token_summary = TopologyLevelSummary(
            count=token_count,
            edges=token_edges,
            components=token_components,
            euler_characteristic=token_count - token_edges,
            edge_density=_edge_density(token_count, token_edges),
        )

        sentence_adj = _cosine_adjacency(sentence_states)
        paragraph_adj = _cosine_adjacency(paragraph_states)
        sentence_edges = _edge_count(sentence_adj, self.sentence_threshold)
        paragraph_edges = _edge_count(paragraph_adj, self.paragraph_threshold)

        sentence_summary = TopologyLevelSummary(
            count=len(flat_sentences),
            edges=sentence_edges,
            components=_connected_components(sentence_adj, self.sentence_threshold),
            euler_characteristic=len(flat_sentences) - sentence_edges,
            edge_density=_edge_density(len(flat_sentences), sentence_edges),
        )
        paragraph_summary = TopologyLevelSummary(
            count=len(paragraphs),
            edges=paragraph_edges,
            components=_connected_components(paragraph_adj, self.paragraph_threshold),
            euler_characteristic=len(paragraphs) - paragraph_edges,
            edge_density=_edge_density(len(paragraphs), paragraph_edges),
        )

        token_state_norm = 0.0 if not token_states else sum(_norm(row) for row in token_states) / len(token_states)
        sentence_state_norm = 0.0 if not sentence_states else sum(_norm(row) for row in sentence_states) / len(sentence_states)
        paragraph_state_norm = 0.0 if not paragraph_states else sum(_norm(row) for row in paragraph_states) / len(paragraph_states)

        bridge_energy = 0.0
        if sentence_states and paragraph_states:
            bridge_energy = _dot(_mean(sentence_states, self.dim), _mean(paragraph_states, self.dim))
        token_sentence_alignment = 0.0
        if token_states and sentence_states:
            token_sentence_alignment = _dot(_mean(token_states, self.dim), _mean(sentence_states, self.dim))
        sentence_paragraph_alignment = 0.0
        if sentence_states and paragraph_states:
            sentence_paragraph_alignment = _dot(_mean(sentence_states, self.dim), _mean(paragraph_states, self.dim))

        dominant_idx = max(range(len(self.euler_basis)), key=self.euler_basis.__getitem__)

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
        )


__all__ = [
    "TextTopologyEngine",
    "TextTopologyResult",
    "TopologyLevelSummary",
]
