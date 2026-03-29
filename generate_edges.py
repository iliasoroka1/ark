"""Auto-generate graph edges from embedding similarity.

Iterates all document pairs in the embedding cache, computes cosine
similarity, and creates 'related_to' edges for pairs above threshold.
This bootstraps the graph for existing documents so graph expansion
and spectral analysis have real data to work with.

Usage: uv run python generate_edges.py [--threshold 0.5] [--corpus agent:ark-local]
"""

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate graph edges from embedding similarity")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cosine similarity threshold (default 0.5)")
    parser.add_argument("--corpus", default="agent:ark-local", help="Corpus to process")
    parser.add_argument("--dry-run", action="store_true", help="Print edges without writing")
    args = parser.parse_args()

    ark_home = os.environ.get("ARK_HOME", os.path.expanduser("~/.ark"))
    embed_path = os.path.join(ark_home, "embeddings.db")
    graph_path = os.path.join(ark_home, "graph.db")

    if not os.path.exists(embed_path):
        print(f"Embedding cache not found: {embed_path}")
        sys.exit(1)

    from ark.engine.embedding_cache import EmbeddingCache
    from ark.engine.graph_store import GraphStore

    cache = EmbeddingCache(embed_path)
    rows = cache.get_corpus(args.corpus)
    print(f"Loaded {len(rows)} embeddings from corpus '{args.corpus}'")

    if len(rows) < 2:
        print("Need at least 2 documents to generate edges")
        sys.exit(0)

    # Build matrix and compute pairwise cosine
    ids = [doc_id for doc_id, _ in rows]
    mat = np.array([vec for _, vec in rows], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    mat_norm = mat / norms

    sims = mat_norm @ mat_norm.T

    # Extract pairs above threshold
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = float(sims[i, j])
            if sim >= args.threshold:
                edges.append((ids[i], ids[j], "related_to", args.corpus, round(sim, 4)))
                edges.append((ids[j], ids[i], "related_to", args.corpus, round(sim, 4)))

    print(f"Found {len(edges) // 2} bidirectional pairs above threshold {args.threshold}")

    if args.dry_run:
        for f, t, et, c, w in edges[:20]:
            print(f"  {f[:12]}.. → {t[:12]}.. ({et}, weight={w})")
        if len(edges) > 20:
            print(f"  ... and {len(edges) - 20} more")
        return

    graph = GraphStore(graph_path)
    graph.add_edges_batch(edges)
    print(f"Wrote {len(edges)} edges to {graph_path}")

    # Summary stats
    unique_nodes = set()
    for f, t, *_ in edges:
        unique_nodes.add(f)
        unique_nodes.add(t)
    print(f"Connected {len(unique_nodes)} documents")


if __name__ == "__main__":
    main()
