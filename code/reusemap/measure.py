"""Pair-level measurements: reference qualities, residual miss rho(K'), window gap G(K')."""
import numpy as np

from .metrics import weighted_quality


def _union_rank(ids, scores, k):
    """Unique ids ranked by descending score (ties by id)."""
    u, first = np.unique(ids, return_index=True)
    s = scores[first]
    o = np.lexsort((u, -s))
    return u[o][:k]


def reference_qualities(p, metric, thr=1):
    K = p.cfg["K"]
    q = lambda f: weighted_quality(f, p.eval_q, p.qrels, p.weights, K, metric, thr)
    return {
        "M_S": q(lambda qi: p.src_top[qi, :K]),
        "M_T": q(lambda qi: p.tgt_top[qi, :K]),
        "M_A": q(lambda qi: p.map_top[qi, :K]),  # adapter-mapped queries over old index
    }


def _subset_weights(p, qs, weights, where):
    """Weights for `qs`, indexed BY QUERY ID -- the convention weighted_quality already uses.

    Both measurement functions take `weights` the same way on purpose: an array over all Q
    queries, read as `weights[qi]`. An array of len(qs) is rejected rather than used, because
    the two are indistinguishable by value and the positional reading would silently score the
    wrong queries. (This asymmetry existed for one round: residual_miss multiplied
    positionally while window_gap indexed by id, and window_gap's docstring claimed they
    matched.)
    """
    if weights is None:
        w = np.asarray(p.weights, float)
    else:
        w = np.asarray(weights, float)
        if w.ndim != 1 or w.shape[0] < int(np.max(qs)) + 1:
            raise ValueError(
                f"{where}: `weights` is indexed by query id, so it needs at least "
                f"{int(np.max(qs)) + 1} entries (got shape {w.shape}). It looks like a "
                f"per-position array of len(queries)={len(qs)}; build a full-length array "
                f"instead, e.g. w = np.zeros(p.Q); w[qs] = 1.0")
    tot = float(w[qs].sum())
    if tot <= 0:
        raise ValueError(f"{where}: the given queries carry zero total weight. An "
                         f"evaluation-only set (TREC-DL) has no workload weight, so pass a "
                         f"uniform array for it rather than the workload's")
    return w


def residual_miss(p, kprimes, top="src", queries=None, weights=None):
    """rho(K') = 1 - E_q |TopK_t(q) & TopK'_window(q)| / K, workload-weighted over eval queries.
    Windows are prefixes of one deep retrieval, so rho is non-increasing in K'.

    `queries` and `weights` default to the primary eval set under the workload, which is what
    every reported number uses. `weights` is indexed by query id, exactly as in window_gap and
    weighted_quality. Pass the secondary (TREC-DL) set with uniform weights to get the same
    quantity on deeply-judged queries -- those are evaluation-only, so their workload weight
    may be zero and normalising by it would divide by zero.
    """
    K = p.cfg["K"]
    win = p.src_top if top == "src" else p.map_top
    qs = p.eval_q if queries is None else np.asarray(queries, np.int64)
    wfull = _subset_weights(p, qs, weights, "residual_miss")
    w = wfull[qs]
    w = w / w.sum()
    out = {}
    for kp in kprimes:
        if kp > win.shape[1]:
            continue
        ov = np.array([len(np.intersect1d(p.tgt_top[qi, :K], win[qi, :kp], assume_unique=True))
                       for qi in qs]) / K
        out[int(kp)] = float(1 - (w * ov).sum())
    return out


def window_gap(p, kprimes, M_T, metric, thr=1, top="src", queries=None, weights=None):
    """G(K') = M_T - quality of target re-ranking the source window only.

    `queries` and `weights` are taken exactly as in residual_miss -- `weights` indexed by query
    id, over all Q queries -- and `M_T` must be the full-corpus target quality measured on the
    SAME query set: mixing a TREC-DL window score against a dev-set M_T would report the
    difference between two query populations as a window gap.
    """
    K = p.cfg["K"]
    win = p.src_top if top == "src" else p.map_top
    qs = p.eval_q if queries is None else np.asarray(queries, np.int64)
    w = _subset_weights(p, qs, weights, "window_gap")
    out = {}
    for kp in kprimes:
        if kp > win.shape[1]:
            continue

        def rank(qi, kp=kp):
            ids = win[qi, :kp]
            return _union_rank(ids, p.tscore(qi, ids), K)

        out[int(kp)] = float(M_T - weighted_quality(rank, qs, p.qrels, w, K, metric, thr))
    return out


def secondary_frontier(p, kprimes, metric, thr=1):
    """rho(K') and G(K') on the secondary (deeply-judged) evaluation set, uniformly weighted.

    This exists to test one specific alternative explanation for G < 0. The primary metric is
    Recall@10 against MS MARCO dev qrels, which carry about one judged passage per query and
    were pooled from older systems. A window strategy is restricted to the source index's
    top-K', so it returns documents closer to that pool, while exhaustive target retrieval
    surfaces unjudged documents that score as irrelevant by construction. That mechanism alone
    predicts G < 0, with no cascade effect -- and G < 0 is what the primary metric reports on 9
    of 10 text pairs. TREC-DL is judged deeply enough to separate the two: if G < 0 survives
    here it is a real cascade, and if it does not, the primary G is an artefact of judgment
    sparsity and every claim resting on it has to be restated.

    Returns {} when no secondary set is configured, so callers need no branch.
    """
    sec = getattr(p, "secondary_q", None)
    if sec is None or not len(sec):
        return {}
    qs = np.asarray(sec, np.int64)
    w = np.zeros(p.Q, float)
    w[qs] = 1.0                      # uniform: these queries are never served, so workload
    K = p.cfg["K"]                   # weight is zero for all of them
    m_t = weighted_quality(lambda qi: p.tgt_top[qi, :K], qs, p.qrels, w, K, metric, thr)
    return {"queries": int(len(qs)), "metric": metric, "thr": int(thr), "M_T": float(m_t),
            "rho": residual_miss(p, kprimes, queries=qs, weights=w),
            "G": window_gap(p, kprimes, m_t, metric, thr, queries=qs, weights=w)}
