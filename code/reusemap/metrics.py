"""Retrieval quality metrics against qrels, and workload-weighted aggregation."""
import numpy as np


def recall_at_k(ranked, rels, k, thr=1):
    rel = {d for d, r in rels.items() if r >= thr}
    if not rel:
        return None
    return len(rel.intersection(int(d) for d in ranked[:k])) / len(rel)


def ndcg_at_k(ranked, rels, k):
    gains = [rels.get(int(d), 0) for d in ranked[:k]]
    dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


METRICS = {
    "recall": lambda ranked, rels, k, thr: recall_at_k(ranked, rels, k, thr),
    "ndcg": lambda ranked, rels, k, thr: ndcg_at_k(ranked, rels, k),
}


def weighted_quality(rank_fn, eval_q, qrels, weights, k, metric="recall", thr=1):
    """Workload-weighted mean of a metric over evaluation queries.

    rank_fn(qi) -> ranked doc ids. Queries whose metric is undefined (no
    relevant documents) are skipped and the weights renormalised.
    """
    f = METRICS[metric]
    num = den = 0.0
    for qi in eval_q:
        rels = qrels.get(int(qi))
        if not rels:
            continue
        v = f(rank_fn(qi), rels, k, thr)
        if v is None:
            continue
        w = float(weights[qi])
        num += w * v
        den += w
    return num / den if den > 0 else float("nan")
