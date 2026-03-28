"""Graph search — beam search + MMR over the observation graph."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ark.engine.embedding_cache import EmbeddingCache
    from ark.engine.graph_store import GraphStore


@dataclass(slots=True)
class GraphHit:
    doc_id: str
    l0: str
    score: float
    relation: str | None = None
    current: bool = True
    hop: int = 0


@dataclass(slots=True)
class GraphResult:
    seeds: list[GraphHit] = field(default_factory=list)
    neighbors: list[GraphHit] = field(default_factory=list)


_EDGE_WEIGHTS = {"derives_from": 0.9, "contradicts": 0.85, "related_to": 1.0, "same_tag": 0.6, "co_session": 0.5}
_HOP_DECAY = 0.8


def _score_neighbor(query_vec, neighbor_vec, parent_score, hop, edge_type, edge_weight, is_current):
    cos_sim = 0.5
    if neighbor_vec is not None:
        dot = float(np.dot(query_vec, neighbor_vec))
        cos_sim = max(0.0, dot)
    type_weight = _EDGE_WEIGHTS.get(edge_type, 0.5)
    if edge_type == "related_to":
        type_weight = edge_weight
    temporal = 1.0 if is_current else 0.6
    return parent_score * (_HOP_DECAY ** hop) * cos_sim * type_weight * temporal


def graph_search(
    seed_ids, query_vec, graph_store, embed_cache, l0_lookup,
    hops=2, beam_width=8, diverse=False, mmr_lambda=0.7, edge_types=None,
):
    q = np.array(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-12:
        q = q / q_norm

    seen: set[str] = set()
    seeds = []
    for doc_id, score in seed_ids:
        seeds.append(GraphHit(doc_id=doc_id, l0=l0_lookup.get(doc_id, ""), score=score))
        seen.add(doc_id)

    frontier = [(doc_id, score, 0) for doc_id, score in seed_ids]
    all_neighbors: list[GraphHit] = []

    for hop in range(1, hops + 1):
        candidates = []
        neighbor_ids: set[str] = set()
        edges_to_score = []

        for parent_id, parent_score, parent_hop in frontier:
            if parent_hop >= hop:
                continue
            for to_id, edge_type, weight in graph_store.get_neighbors(parent_id, edge_types=edge_types):
                if to_id not in seen:
                    neighbor_ids.add(to_id)
                    edges_to_score.append((to_id, parent_score, edge_type, f"{edge_type} {parent_id[:8]}", weight))
            for from_id, edge_type, weight in graph_store.get_predecessors(parent_id, edge_types=edge_types):
                if from_id not in seen:
                    neighbor_ids.add(from_id)
                    edges_to_score.append((from_id, parent_score, edge_type, f"{parent_id[:8]} {edge_type}", weight))

        if not neighbor_ids:
            break

        vecs = embed_cache.get_many(list(neighbor_ids))

        for neighbor_id, parent_score, edge_type, relation, edge_weight in edges_to_score:
            if neighbor_id in seen:
                continue
            nvec = vecs.get(neighbor_id)
            nvec_np = None
            if nvec is not None:
                nvec_np = np.array(nvec, dtype=np.float32)
                n_norm = np.linalg.norm(nvec_np)
                if n_norm > 1e-12:
                    nvec_np = nvec_np / n_norm
            score = _score_neighbor(q, nvec_np, parent_score, hop, edge_type, edge_weight, True)
            candidates.append((score, neighbor_id, relation))

        candidates.sort(reverse=True, key=lambda x: x[0])
        new_frontier = []

        for score, doc_id, relation in candidates[:beam_width]:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            all_neighbors.append(GraphHit(doc_id=doc_id, l0=l0_lookup.get(doc_id, ""), score=score, relation=relation, hop=hop))
            new_frontier.append((doc_id, score, hop))

        frontier = new_frontier

    if diverse and len(all_neighbors) > 1:
        all_neighbors = _mmr_rerank(all_neighbors, q, embed_cache, mmr_lambda)

    return GraphResult(seeds=seeds, neighbors=all_neighbors)


def annotate_edges(doc_ids, graph_store, l0_lookup, max_per_doc=3):
    result = {}
    for doc_id in doc_ids:
        edges = graph_store.get_all_edges(doc_id, current_only=True)
        annotations = []
        for other_id, edge_type, direction, weight, _valid_at in edges[:max_per_doc]:
            annotations.append({"id": other_id, "l0": l0_lookup.get(other_id, ""), "type": edge_type, "direction": direction, "weight": round(weight, 2)})
        if annotations:
            result[doc_id] = annotations
    return result


def _mmr_rerank(candidates, query_vec, embed_cache, lam=0.7):
    vecs = embed_cache.get_many([c.doc_id for c in candidates])
    mat = []
    valid_candidates = []
    for c in candidates:
        v = vecs.get(c.doc_id)
        if v is not None:
            arr = np.array(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 1e-12:
                mat.append(arr / norm)
                valid_candidates.append(c)
    if len(valid_candidates) <= 1:
        return candidates
    mat_np = np.array(mat, dtype=np.float32)
    q_sims = mat_np @ query_vec
    selected = []
    remaining = set(range(len(valid_candidates)))
    result = []
    for _ in range(len(valid_candidates)):
        best_score = -float("inf")
        best_idx = -1
        for idx in remaining:
            relevance = float(q_sims[idx])
            max_sim = 0.0
            if selected:
                sims_to_selected = mat_np[idx] @ mat_np[selected].T
                max_sim = float(np.max(sims_to_selected))
            score = lam * relevance - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        remaining.discard(best_idx)
        selected.append(best_idx)
        result.append(valid_candidates[best_idx])
    return result


def deduplicate_hits(hits: list[GraphHit]) -> list[GraphHit]:
    """Remove duplicate doc_ids, keeping the highest-scored hit for each."""
    best: dict[str, GraphHit] = {}
    for hit in hits:
        existing = best.get(hit.doc_id)
        if existing is None or hit.score > existing.score:
            best[hit.doc_id] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def parallel_graph_search(
    seed_ids,
    queries: list[tuple[str, list[float]]],
    graph_store: GraphStore,
    embed_cache: EmbeddingCache,
    l0_lookup,
    use_case: str | None = None,
    hops=2,
    beam_width=8,
    diverse=False,
    mmr_lambda=0.7,
    edge_types=None,
) -> GraphResult:
    """Run graph_search for multiple query angles in parallel, merge and re-rank."""

    def _run_single(idx_query):
        idx, (_text, qvec) = idx_query
        result = graph_search(
            seed_ids, qvec, graph_store, embed_cache, l0_lookup,
            hops=hops, beam_width=beam_width, diverse=diverse,
            mmr_lambda=mmr_lambda, edge_types=edge_types,
        )
        return idx, result

    # Run all queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        results = list(pool.map(_run_single, enumerate(queries)))

    # Merge all hits from parallel searches
    all_seeds: list[GraphHit] = []
    all_neighbors: list[GraphHit] = []

    for _idx, result in results:
        all_seeds.extend(result.seeds)
        all_neighbors.extend(result.neighbors)

    # Dedup seeds and neighbors independently
    merged_seeds = deduplicate_hits(all_seeds)
    merged_neighbors = deduplicate_hits(all_neighbors)

    # Apply use_case re-ranking boost
    if use_case is not None and merged_neighbors:
        uc_vecs = embed_cache.get_many([use_case])
        uc_vec_raw = uc_vecs.get(use_case)
        if uc_vec_raw is not None:
            uc_vec = np.array(uc_vec_raw, dtype=np.float32)
            uc_norm = np.linalg.norm(uc_vec)
            if uc_norm > 1e-12:
                uc_vec = uc_vec / uc_norm

            neighbor_ids = [h.doc_id for h in merged_neighbors]
            nvecs = embed_cache.get_many(neighbor_ids)

            for hit in merged_neighbors:
                nvec_raw = nvecs.get(hit.doc_id)
                if nvec_raw is not None:
                    nvec = np.array(nvec_raw, dtype=np.float32)
                    n_norm = np.linalg.norm(nvec)
                    if n_norm > 1e-12:
                        nvec = nvec / n_norm
                        sim = float(np.dot(uc_vec, nvec))
                        sim = max(0.0, sim)
                        hit.score *= 0.5 + 0.5 * sim

            # Re-sort after boosting
            merged_neighbors.sort(key=lambda h: h.score, reverse=True)

    return GraphResult(seeds=merged_seeds, neighbors=merged_neighbors)
