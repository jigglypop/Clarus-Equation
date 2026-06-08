from __future__ import annotations

import re


DEFAULT_WIKI_DATASET = "lcw99/wikipedia-korean-20221001"
DEFAULT_TRAIN_CORPUS = "mixed-ko-train"
WIKI_ALPACA_TRAIN_CORPUS = "mixed-ko-wiki-alpaca"


def content_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"[0-9A-Za-z\uac00-\ud7a3]{2,}", text)}


def build_prompt_weights(prompts: list[str] | tuple[str, ...] | None) -> dict[str, int]:
    weights: dict[str, int] = {}
    for prompt in prompts or []:
        for token in content_terms(prompt):
            weights[token] = weights.get(token, 0) + 1
    return weights


def topical_document_score(text: str, prompt_weights: dict[str, int]) -> float:
    if not prompt_weights:
        return 0.0
    tokens = content_terms(text)
    if not tokens:
        return 0.0
    overlaps = [prompt_weights[token] for token in tokens if token in prompt_weights]
    if not overlaps:
        return 0.0
    overlap_mass = float(sum(overlaps))
    coverage = float(len(overlaps)) / max(len(prompt_weights), 1)
    density = float(len(overlaps)) / max(len(tokens), 1)
    return overlap_mass + 2.0 * coverage + density


def select_topical_chunks(chunks: list[str], prompt_weights: dict[str, int], keep_limit: int) -> list[str]:
    if not chunks:
        return []
    if not prompt_weights:
        return chunks[:keep_limit]
    scored = [
        (topical_document_score(chunk, prompt_weights), idx, chunk)
        for idx, chunk in enumerate(chunks)
    ]
    positive = [item for item in scored if item[0] > 0.0]
    chosen = positive if positive else scored
    chosen.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [chunk for _, _, chunk in chosen[:keep_limit]]


def sleep_curriculum_stage(cycle_idx: int) -> dict[str, str]:
    if cycle_idx <= 2:
        return {"name": "wiki", "dataset_name": DEFAULT_WIKI_DATASET}
    if cycle_idx <= 4:
        return {"name": "wiki+alpaca", "dataset_name": WIKI_ALPACA_TRAIN_CORPUS}
    return {"name": "wiki+alpaca+squad", "dataset_name": DEFAULT_TRAIN_CORPUS}
