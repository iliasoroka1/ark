"""Surprisal scoring via k-NN graph random walk.

Identifies geometrically anomalous observations in embedding space.
Observations in sparse/peripheral regions of the graph get high
surprisal scores — these are facts that don't cluster with anything
else, making them good candidates for consolidation review.

Algorithm:
  1. L2-normalize embeddings, compute cosine similarity via dot product
  2. Build k-NN graph from the similarity matrix
  3. Convert adjacency to row-stochastic transition matrix
  4. Power-iterate to find stationary distribution π
  5. surprisal(i) = -log(π[i])

High surprisal = low stationary probability = peripheral/isolated fact.
Low surprisal = well-connected hub = mainstream knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SurprisalScore:
    """A scored observation."""

    doc_id: str
    surprisal: float  # higher = more anomalous
    normalized: float  # 0-1 scaled (0=common, 1=most anomalous)


def compute_surprisal(
    observations: list[tuple[str, list[float]]],
    k: int = 5,
    max_iter: int = 200,
    tol: float = 1e-10,
    top_percent: float = 0.10,
    reference: list[tuple[str, list[float]]] | None = None,
) -> list[SurprisalScore]:
    """Compute surprisal scores for a set of observations.

    Args:
        observations: List of (doc_id, embedding_vector) pairs to score.
        k: Number of nearest neighbors for the graph.
        max_iter: Max power iteration steps.
        tol: Convergence tolerance for stationary distribution.
        top_percent: Fraction of most surprising to return (0.10 = top 10%).
        reference: If provided, build the k-NN graph from reference corpus
            but only return scores for observations. This enables incremental
            dreaming: score new docs against the full corpus without re-scoring
            everything.

    Returns:
        List of SurprisalScore sorted by surprisal descending (most anomalous first).
        Only returns the top_percent fraction.
    """
    if reference is not None:
        return _compute_incremental(observations, reference, k, max_iter, tol, top_percent)

    n = len(observations)
    if n < 3:
        return []

    k_actual = min(k, n - 1)

    doc_ids = [obs[0] for obs in observations]
    vecs = np.array([obs[1] for obs in observations], dtype=np.float32)

    # Step 1: L2-normalize → cosine similarity = dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vecs = vecs / norms

    # Step 2: Full similarity matrix (n x n) via single matmul
    # For n=5000, dims=1024 this is ~100MB and takes <1s
    sim = vecs @ vecs.T

    # Zero out self-similarity
    np.fill_diagonal(sim, -np.inf)

    # Step 3: k-NN via argpartition (O(n²) but vectorized, much faster than Python loops)
    # For each row, find indices of k largest similarities
    knn_indices = np.argpartition(sim, -k_actual, axis=1)[:, -k_actual:]

    # Step 4: Build sparse transition matrix from k-NN graph
    transition = _build_transition(n, knn_indices, k_actual)

    # Step 5: Power iteration for stationary distribution
    stationary = _power_iterate(transition, max_iter, tol)

    # Step 6: Surprisal = -log(π)
    raw = -np.log(stationary + 1e-15)

    # Normalize to [0, 1]
    rmin, rmax = raw.min(), raw.max()
    spread = rmax - rmin
    normed = (raw - rmin) / spread if spread > 1e-12 else np.full(n, 0.5)

    # Sort by surprisal descending, return top percent
    order = np.argsort(raw)[::-1]
    count = max(1, int(n * top_percent))

    return [
        SurprisalScore(
            doc_id=doc_ids[i],
            surprisal=float(raw[i]),
            normalized=float(normed[i]),
        )
        for i in order[:count]
    ]


def _compute_incremental(
    new_obs: list[tuple[str, list[float]]],
    full_corpus: list[tuple[str, list[float]]],
    k: int,
    max_iter: int,
    tol: float,
    top_percent: float,
) -> list[SurprisalScore]:
    """Score only new observations against the full corpus.

    Builds the k-NN graph on the full corpus (including new docs),
    runs power iteration, but only returns scores for new doc_ids.
    Much cheaper than full re-score when new_obs << full_corpus.
    """
    if not new_obs:
        return []

    # Merge new into full (they should already be there, but ensure)
    new_ids = {obs[0] for obs in new_obs}
    full_ids = [obs[0] for obs in full_corpus]
    full_vecs = np.array([obs[1] for obs in full_corpus], dtype=np.float32)
    n = len(full_corpus)

    if n < 3:
        return []

    k_actual = min(k, n - 1)

    # L2-normalize
    norms = np.linalg.norm(full_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    full_vecs = full_vecs / norms

    # Similarity + k-NN on full corpus
    sim = full_vecs @ full_vecs.T
    np.fill_diagonal(sim, -np.inf)
    knn_indices = np.argpartition(sim, -k_actual, axis=1)[:, -k_actual:]

    transition = _build_transition(n, knn_indices, k_actual)
    stationary = _power_iterate(transition, max_iter, tol)

    raw = -np.log(stationary + 1e-15)
    rmin, rmax = raw.min(), raw.max()
    spread = rmax - rmin
    normed = (raw - rmin) / spread if spread > 1e-12 else np.full(n, 0.5)

    # Filter to only new doc scores
    scores = []
    for i, did in enumerate(full_ids):
        if did in new_ids:
            scores.append(SurprisalScore(
                doc_id=did,
                surprisal=float(raw[i]),
                normalized=float(normed[i]),
            ))

    scores.sort(key=lambda s: s.surprisal, reverse=True)
    count = max(1, int(len(scores) * top_percent))
    return scores[:count]


def _build_transition(
    n: int,
    knn_indices: NDArray[np.intp],
    k: int,
) -> NDArray[np.float32]:
    """Build row-stochastic transition matrix from k-NN indices.

    Each node transitions uniformly to its k nearest neighbors.
    """
    T = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    cols = knn_indices.ravel()
    T[rows, cols] = 1.0 / k
    return T


def _power_iterate(
    T: NDArray[np.float32],
    max_iter: int,
    tol: float,
) -> NDArray[np.float64]:
    """Find stationary distribution π such that π·T = π.

    Computes π_{t+1} = T^T · π_t and normalizes each step.
    Uses float64 for numerical stability during iteration.
    """
    n = T.shape[0]
    T_t = T.T.astype(np.float64)
    pi = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(max_iter):
        pi_new = T_t @ pi
        # Normalize to stay on the probability simplex
        total = pi_new.sum()
        if total > 0:
            pi_new /= total
        diff = np.abs(pi_new - pi).sum()
        pi = pi_new
        if diff < tol:
            break

    return pi
