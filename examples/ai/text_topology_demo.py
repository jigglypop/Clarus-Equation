"""Demo for the topology-first Euler text engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarus.text_topology import TextTopologyEngine


SAMPLE_TEXT = """
Language is not only a sequence of tokens. A sentence bends local meaning.
Another sentence can reconnect that meaning from a different angle.

A paragraph then becomes a small field of linked statements.
When the field stays coherent, the topology remains connected.
"""


def _load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    data = sys.stdin.read().strip()
    if data:
        return data
    return SAMPLE_TEXT


def _print_links(title: str, links) -> None:
    print(title)
    if not links:
        print("  none")
        return
    for link in links:
        print(
            " ",
            f"{link.source}->{link.target}",
            f"weight={link.weight:.6f}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Topology-first Euler text demo")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    engine = TextTopologyEngine(dim=args.dim)
    result = engine.analyze(_load_text(args))

    print("dominant_euler_basis", result.dominant_euler_basis)
    print(
        "euler_basis_activation",
        " ".join(f"{k}={v:.6f}" for k, v in sorted(result.euler_basis_activation.items())),
    )
    print("bridge_energy", f"{result.bridge_energy:.6f}")
    print("token_sentence_bridge", f"{result.token_sentence_bridge:.6f}")
    print("sentence_paragraph_bridge", f"{result.sentence_paragraph_bridge:.6f}")
    print("token_sentence_alignment", f"{result.token_sentence_alignment:.6f}")
    print("sentence_paragraph_alignment", f"{result.sentence_paragraph_alignment:.6f}")
    print("")
    print(
        "tokens",
        result.token_summary.count,
        "edges",
        result.token_summary.edges,
        "faces",
        result.token_summary.faces,
        "components",
        result.token_summary.components,
        "euler",
        result.token_summary.euler_characteristic,
        "density",
        f"{result.token_summary.edge_density:.6f}",
    )
    print(
        "sentences",
        result.sentence_summary.count,
        "edges",
        result.sentence_summary.edges,
        "faces",
        result.sentence_summary.faces,
        "components",
        result.sentence_summary.components,
        "euler",
        result.sentence_summary.euler_characteristic,
        "density",
        f"{result.sentence_summary.edge_density:.6f}",
        "fiedler",
        f"{result.sentence_summary.algebraic_connectivity:.6f}",
    )
    print(
        "paragraphs",
        result.paragraph_summary.count,
        "edges",
        result.paragraph_summary.edges,
        "faces",
        result.paragraph_summary.faces,
        "components",
        result.paragraph_summary.components,
        "euler",
        result.paragraph_summary.euler_characteristic,
        "density",
        f"{result.paragraph_summary.edge_density:.6f}",
        "fiedler",
        f"{result.paragraph_summary.algebraic_connectivity:.6f}",
    )
    print("")
    print("token_state_norm", f"{result.token_state_norm:.6f}")
    print("sentence_state_norm", f"{result.sentence_state_norm:.6f}")
    print("paragraph_state_norm", f"{result.paragraph_state_norm:.6f}")
    print("")
    print("sentences_raw", len(result.sentences))
    for idx, sentence in enumerate(result.sentences):
        print(" ", idx, sentence)
    print("")
    print("paragraphs_raw", len(result.paragraphs))
    for idx, paragraph in enumerate(result.paragraphs):
        print(" ", idx, paragraph.replace("\n", " "))
    print("")
    _print_links("top_sentence_links", result.sentence_links)
    _print_links("top_paragraph_links", result.paragraph_links)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
