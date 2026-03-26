"""Spectral analysis of the observation graph + embedding space.

Random matrix theory, PageRank, betweenness centrality, graph Laplacian,
entropy production, local entropy. All numpy. No external deps.

These give the agent (and dreamer) signals that flat search can't:
- Which memories are structurally novel (RMT)
- Which are foundational (PageRank)
- Which are bridges between knowledge domains (betweenness)
- Where the natural knowledge boundaries are (Fiedler)
- How much a new fact disrupted existing structure (entropy production)
- Which memories are diverse hubs (local entropy)
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ark.engine.embedding_cache import EmbeddingCache
    from ark.engine.graph_store import GraphStore


# ── 1. Random Matrix Theory — Marchenko-Pastur anomaly detection ──


def rmt_anomalies(
    embed_cache: EmbeddingCache,
    corpus: str,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Find structurally novel observations via RMT.

    Builds the correlation matrix of embeddings, compares eigenvalue
    spectrum to the Marchenko-Pastur distribution (null model for
    random matrices). Observations aligned with eigenvalues that
    exceed the MP upper bound carry genuine structure — they represent
    knowledge directions the corpus has very little of.

    Returns [(doc_id, novelty_score)] sorted descending.
    """
    rows = embed_cache.get_corpus(corpus)
    if len(rows) < 10:
        return []

    ids = [r[0] for r in rows]
    M = np.array([r[1] for r in rows], dtype=np.float32)
    N, d = M.shape

    # Center but do NOT L2-normalize. RMT needs the variance structure.
    # Normalizing puts everything on a unit sphere, washing out the signal.
    M = M - M.mean(axis=0)

    # Use the smaller covariance matrix for efficiency + correct MP bounds.
    # Standard formulation: C = (1/N) X^T X (d×d), gamma = d/N.
    # MP bounds apply to eigenvalues of C.
    C = (M.T @ M) / N  # d × d
    gamma = d / N
    lambda_plus = (1 + np.sqrt(gamma)) ** 2
    lambda_minus = max(0, (1 - np.sqrt(gamma)) ** 2)

    # Eigendecompose the d×d matrix
    eigenvalues, eigenvectors_d = np.linalg.eigh(C)

    # Outlier eigenvalues: above MP upper bound
    outlier_mask = eigenvalues > lambda_plus
    if not outlier_mask.any():
        return []

    # Map back to observation space: project each observation onto
    # the outlier principal components.
    # eigenvectors_d is d × k_outliers, M is N × d
    # scores_per_obs = M @ eigenvectors_d → N × k_outliers (projection)
    outlier_pcs = eigenvectors_d[:, outlier_mask]  # d × k_outliers
    outlier_vals = eigenvalues[outlier_mask]
    projections = M @ outlier_pcs  # N × k_outliers

    # Weight by how far each eigenvalue exceeds the MP bound
    excess = outlier_vals - lambda_plus
    # Squared projection magnitude, weighted by excess
    scores = (projections ** 2) @ excess

    # Normalize to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    spread = s_max - s_min
    if spread > 1e-12:
        scores = (scores - s_min) / spread

    top_idx = np.argsort(-scores)[:top_k]
    return [(ids[i], float(scores[i])) for i in top_idx]


# ── 2. PageRank — recursive importance on derivation graph ────────


def pagerank(
    graph_store: GraphStore,
    corpus: str,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    edge_types: set[str] | None = None,
) -> dict[str, float]:
    """PageRank on the observation graph.

    A memory is important if important memories derive from it.
    The derives_from edges create a natural citation graph.

    By default uses all edge types. Pass edge_types={"derives_from"}
    for pure citation importance.

    Returns {doc_id: rank} normalized to sum=1.
    """
    et = edge_types or {"derives_from", "contradicts", "related_to", "same_tag", "co_session"}
    edges = []
    for etype in et:
        edges.extend(graph_store.get_edges_by_type(corpus, etype))

    if not edges:
        return {}

    # Build node index
    nodes: set[str] = set()
    for f, t, _ in edges:
        nodes.add(f)
        nodes.add(t)

    node_list = sorted(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    # Build adjacency matrix (weighted, directed)
    A = np.zeros((N, N), dtype=np.float64)
    for f, t, w in edges:
        A[idx[t], idx[f]] += w  # column-stochastic: A[to, from]

    # Normalize columns (out-degree normalization)
    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    A = A / col_sums

    # Power iteration: r = d * A @ r + (1-d)/N
    r = np.full(N, 1.0 / N, dtype=np.float64)
    teleport = (1 - damping) / N

    for _ in range(max_iter):
        r_new = damping * (A @ r) + teleport
        r_new /= r_new.sum()  # stay on simplex
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new

    return {node_list[i]: float(r[i]) for i in range(N)}


# ── 3. Betweenness centrality — Brandes' algorithm ───────────────


def betweenness_centrality(
    graph_store: GraphStore,
    corpus: str,
    edge_types: set[str] | None = None,
    sample: int | None = None,
) -> dict[str, float]:
    """Betweenness centrality via Brandes' algorithm.

    Nodes that lie on many shortest paths between other pairs are
    structural bridges — they connect otherwise separate knowledge.

    O(V*E) for exact computation. Pass sample=N to approximate
    by sampling N source nodes.

    Returns {doc_id: centrality} normalized by 1/((V-1)(V-2)).
    """
    # Build adjacency list (undirected)
    adj: dict[str, list[str]] = defaultdict(list)

    et = edge_types or {"derives_from", "contradicts", "related_to", "same_tag", "co_session"}
    for etype in et:
        for f, t, _ in graph_store.get_edges_by_type(corpus, etype):
            adj[f].append(t)
            adj[t].append(f)

    if not adj:
        return {}

    nodes = list(adj.keys())
    N = len(nodes)
    CB: dict[str, float] = {n: 0.0 for n in nodes}

    # Optional sampling for large graphs
    sources = nodes
    if sample and sample < N:
        rng = np.random.default_rng(42)
        sources = list(rng.choice(nodes, size=sample, replace=False))

    for s in sources:
        # BFS from s
        S: list[str] = []  # stack of nodes in order of distance
        P: dict[str, list[str]] = {n: [] for n in nodes}  # predecessors
        sigma: dict[str, int] = {n: 0 for n in nodes}  # number of shortest paths
        sigma[s] = 1
        d: dict[str, int] = {n: -1 for n in nodes}  # distance
        d[s] = 0
        Q: deque[str] = deque([s])

        while Q:
            v = Q.popleft()
            S.append(v)
            for w in adj[v]:
                if d[w] < 0:  # first visit
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:  # shortest path via v
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # Accumulate dependencies
        delta: dict[str, float] = {n: 0.0 for n in nodes}
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                CB[w] += delta[w]

    # Normalize
    if N > 2:
        norm = 1.0 / ((N - 1) * (N - 2))
        # If sampling, scale up
        if sample and sample < N:
            norm *= N / sample
        for n in CB:
            CB[n] *= norm

    return CB


# ── 4. Graph Laplacian spectrum — Fiedler vector + spectral gap ───


def laplacian_analysis(
    graph_store: GraphStore,
    corpus: str,
    k: int = 6,
    edge_types: set[str] | None = None,
) -> dict:
    """Spectral analysis of the graph Laplacian.

    Returns:
        {
            "spectral_gap": float,          # λ₂ — connectivity measure
            "num_components": int,           # count of zero eigenvalues
            "eigenvalues": [float, ...],     # smallest k eigenvalues
            "boundary_nodes": [(id, score)], # nodes near Fiedler cut
            "fiedler_communities": {         # sign-based bisection
                "positive": [id, ...],
                "negative": [id, ...],
            }
        }
    """
    et = edge_types or {"derives_from", "contradicts", "related_to", "same_tag", "co_session"}
    edges = []
    for etype in et:
        edges.extend(graph_store.get_edges_by_type(corpus, etype))

    if not edges:
        return {"spectral_gap": 0.0, "num_components": 0, "eigenvalues": [], "boundary_nodes": [], "fiedler_communities": {"positive": [], "negative": []}}

    # Build node index
    nodes: set[str] = set()
    for f, t, _ in edges:
        nodes.add(f)
        nodes.add(t)

    node_list = sorted(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    if N < 3:
        return {"spectral_gap": 0.0, "num_components": N, "eigenvalues": [], "boundary_nodes": [], "fiedler_communities": {"positive": [], "negative": []}}

    # Adjacency matrix (undirected, weighted)
    A = np.zeros((N, N), dtype=np.float64)
    for f, t, w in edges:
        i, j = idx[f], idx[t]
        A[i, j] += w
        A[j, i] += w

    # Degree matrix
    D = np.diag(A.sum(axis=1))

    # Laplacian L = D - A
    L = D - A

    # Eigendecompose (smallest k eigenvalues)
    k_actual = min(k, N)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = eigenvalues[:k_actual]
    eigenvectors = eigenvectors[:, :k_actual]

    # Count zero eigenvalues (connected components)
    num_components = int(np.sum(eigenvalues < 1e-8))

    # Spectral gap = λ₂ (algebraic connectivity)
    spectral_gap = float(eigenvalues[1]) if N > 1 else 0.0

    # Fiedler vector = eigenvector for λ₂
    fiedler = eigenvectors[:, 1] if N > 1 else np.zeros(N)

    # Boundary nodes: small absolute Fiedler component = on the cut
    abs_fiedler = np.abs(fiedler)
    boundary_idx = np.argsort(abs_fiedler)[:min(10, N)]
    boundary_nodes = [(node_list[i], float(abs_fiedler[i])) for i in boundary_idx]

    # Fiedler bisection
    positive = [node_list[i] for i in range(N) if fiedler[i] >= 0]
    negative = [node_list[i] for i in range(N) if fiedler[i] < 0]

    return {
        "spectral_gap": spectral_gap,
        "num_components": num_components,
        "eigenvalues": [float(e) for e in eigenvalues],
        "boundary_nodes": boundary_nodes,
        "fiedler_communities": {"positive": positive, "negative": negative},
    }


# ── 5. Entropy production — disruption score at add time ──────────


def entropy_production(
    graph_store: GraphStore,
    doc_id: str,
    corpus: str,
) -> float:
    """Measure how much a new observation disrupted the knowledge graph.

    Computes the change in Shannon entropy of the degree distribution
    in the 1-hop neighborhood before and after the new node's edges.
    High entropy production = the observation significantly changed
    the local structure.

    O(degree) — fast enough to run at every add.
    """
    edges = graph_store.get_all_edges(doc_id, current_only=True)
    if not edges:
        return 0.0

    # Collect the 1-hop neighborhood (excluding the new node)
    neighbor_ids = list({e[0] for e in edges})

    # Degree distribution of neighbors WITHOUT the new node's contribution
    degrees_before: list[int] = []
    degrees_after: list[int] = []

    for nid in neighbor_ids:
        all_edges = graph_store.get_all_edges(nid, current_only=True)
        degree_total = len(all_edges)
        # Count edges to/from doc_id
        edges_to_new = sum(1 for e in all_edges if e[0] == doc_id)
        degrees_before.append(max(0, degree_total - edges_to_new))
        degrees_after.append(degree_total)

    if not degrees_before:
        return 0.0

    H_before = _shannon_entropy(degrees_before)
    H_after = _shannon_entropy(degrees_after)

    return abs(H_after - H_before)


def _shannon_entropy(values: list[int]) -> float:
    """Shannon entropy of a discrete distribution."""
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    probs = np.array(values, dtype=np.float64) / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


# ── 6. Local edge-type entropy — diversity per node ───────────────


def local_entropy(
    graph_store: GraphStore,
    doc_id: str,
) -> float:
    """Shannon entropy of the edge-type distribution for a node.

    High entropy = connects to diverse types of knowledge.
    Low entropy = deeply embedded in one cluster.

    Example:
        3 derives_from + 2 contradicts + 1 related_to → H ≈ 1.46
        5 same_tag + 0 others → H = 0.0
    """
    edges = graph_store.get_all_edges(doc_id, current_only=True)
    if not edges:
        return 0.0

    type_counts: dict[str, int] = defaultdict(int)
    for _, edge_type, _, _, _ in edges:
        type_counts[edge_type] += 1

    counts = list(type_counts.values())
    return _shannon_entropy(counts)


def local_entropy_batch(
    graph_store: GraphStore,
    corpus: str,
    edge_types: set[str] | None = None,
) -> dict[str, float]:
    """Local entropy for all nodes in a corpus. Returns {doc_id: entropy}."""
    et = edge_types or {"derives_from", "contradicts", "related_to", "same_tag", "co_session"}

    # Build type counts per node from all edges
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for etype in et:
        for f, t, _ in graph_store.get_edges_by_type(corpus, etype):
            type_counts[f][etype] += 1
            type_counts[t][etype] += 1

    result: dict[str, float] = {}
    for nid, counts in type_counts.items():
        vals = list(counts.values())
        result[nid] = _shannon_entropy(vals)

    return result


# ── Full analysis report ──────────────────────────────────────────


def full_analysis(
    graph_store: GraphStore,
    embed_cache: EmbeddingCache,
    corpus: str,
    top_k: int = 10,
) -> dict:
    """Run all spectral analyses and return a unified report.

    This is what the agent calls via memory(action="analyze").
    """
    report: dict = {}

    # 1. RMT anomalies
    rmt = rmt_anomalies(embed_cache, corpus, top_k=top_k)
    report["rmt_novel"] = [{"id": doc_id, "novelty": round(s, 3)} for doc_id, s in rmt]

    # 2. PageRank
    pr = pagerank(graph_store, corpus)
    if pr:
        top_pr = sorted(pr.items(), key=lambda x: -x[1])[:top_k]
        report["pagerank_top"] = [{"id": doc_id, "rank": round(r, 6)} for doc_id, r in top_pr]
    else:
        report["pagerank_top"] = []

    # 3. Betweenness centrality
    bc = betweenness_centrality(graph_store, corpus, sample=200)
    if bc:
        top_bc = sorted(bc.items(), key=lambda x: -x[1])[:top_k]
        report["bridges"] = [{"id": doc_id, "centrality": round(c, 6)} for doc_id, c in top_bc]
    else:
        report["bridges"] = []

    # 4. Laplacian analysis
    lap = laplacian_analysis(graph_store, corpus)
    report["graph_structure"] = {
        "spectral_gap": round(lap["spectral_gap"], 4),
        "num_components": lap["num_components"],
        "boundary_nodes": [{"id": nid, "fiedler_abs": round(s, 4)} for nid, s in lap["boundary_nodes"][:5]],
    }

    # 5. Local entropy (batch)
    le = local_entropy_batch(graph_store, corpus)
    if le:
        # Top diverse hubs
        top_le = sorted(le.items(), key=lambda x: -x[1])[:top_k]
        report["diverse_hubs"] = [{"id": doc_id, "entropy": round(e, 3)} for doc_id, e in top_le]
    else:
        report["diverse_hubs"] = []

    return report
