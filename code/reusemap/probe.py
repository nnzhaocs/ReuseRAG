"""RQ3: bounded-cost estimation of window residual miss (intruder counts).

For query q with window W (source top-K'), let w_1 >= ... >= w_K be the top-K target
scores inside W and I(w) the number of documents outside W with target score > w.
The target top-K over the whole corpus contains j outside documents iff the j-th best
outside score exceeds w_{K-j+1}, so

    miss_q = |{ j in 1..K : I(w_{K-j+1}) >= j }|        (exact)
    min(I(w_K), K)                                        (upper bound only)

I(w) is estimated from a sample of documents encoded with the target model.
Estimators:
  uniform    : sample m docs uniformly from D;   I_hat(w) = N/m * #{sampled, outside W, score > w}
  stratified : sample m_A from A = source ranks (K', c*K'] and reuse the shared uniform
               sample S (size m) for the rest, counting hits only in B = D minus (W u A):
               I_hat = |A|/m_A * hits_A + N/m * hits_B
               N/m is correct: S is uniform over all of D, so E[hits_B] = (m/N)*|{d in B: s>w}|.
               The ratio form |B|/|S n B| * hits_B is also (approximately) unbiased with the
               same variance in practice; tests/test_invariants.py checks both are exact when
               the samples are the whole corpus / whole stratum.
Cost per query = |W| (window encodings) + sample size.  Sampled docs are migration work,
so in deployment they are not wasted.
"""
import numpy as np


def exact_miss(p, qi, kp):
    K = p.cfg["K"]
    return len(np.setdiff1d(p.tgt_top[qi, :K], p.src_top[qi, :kp], assume_unique=True))


def _miss_from_counts(I_at_w, K):
    # I_at_w[k-1] = estimated # outside docs scoring above w_k (w sorted descending)
    return sum(1 for j in range(1, K + 1) if I_at_w[K - j] >= j)


def estimate_miss(p, qi, kp, shared, kind="uniform", rng=None, mA=0, c=None):
    """shared: a document sample (indices) drawn ONCE and encoded once for all probe
    queries. For 'stratified', mA extra docs are drawn from this query's source ranks
    (K', c*K'] (query-specific encodings)."""
    K = p.cfg["K"]
    c = c or p.cfg.get("probe_c", 4)
    rng = rng or np.random.default_rng(0)
    W = p.src_top[qi, :kp]
    in_w = np.zeros(p.N, bool)
    in_w[W] = True
    w = np.sort(p.tscore(qi, W))[::-1][:K]
    m = len(shared)
    if kind == "uniform":
        sc = p.tscore(qi, shared[~in_w[shared]])
        I = np.array([(sc > t).sum() for t in w]) * (p.N / m)
        extra = np.empty(0, np.int64)
    elif kind == "stratified":
        A = p.src_top[qi, kp:min(c * kp, p.src_top.shape[1])]
        mA = min(len(A), mA)
        SA = rng.choice(A, size=mA, replace=False) if mA else np.empty(0, np.int64)
        excl = in_w.copy()
        excl[A] = True
        scA = p.tscore(qi, SA)
        scB = p.tscore(qi, shared[~excl[shared]])
        I = np.array([(scA > t).sum() * (len(A) / max(mA, 1)) + (scB > t).sum() * (p.N / m)
                      for t in w])
        extra = SA
    else:
        raise ValueError(kind)
    return {"miss_hat": _miss_from_counts(I, K), "bound_hat": float(min(I[-1], K)), "extra": extra}


def evaluate_probe(p, kp, budgets_frac, kinds=("uniform", "stratified"), n_queries=200,
                   strat_per_query=None, seed=0):
    """Compare estimated vs exact rho(K') on a random subset of evaluation queries.

    Cost accounting (target encodings, deduplicated): |probe windows U shared sample U
    stratum samples|. Every encoded doc is migration work. At scale the probe windows
    dominate (n_queries * K'), so budget n_queries accordingly.
    """
    rng = np.random.default_rng(seed)
    K = p.cfg["K"]
    # probe queries come from past traffic (what exists before migration), never eval queries
    pool = getattr(p, "traffic_pool", p.eval_q)
    if getattr(p, "leaky", False):
        pool = p.eval_q
    qs = rng.choice(pool, size=min(n_queries, len(pool)), replace=False)
    w = p.weights[qs]
    if w.sum() <= 0:
        raise ValueError("every probe query has weight 0 under this workload; rho would be NaN")
    w = w / w.sum()
    true = np.array([exact_miss(p, qi, kp) for qi in qs]) / K
    rho_true = float((w * true).sum())
    win_ids = np.unique(p.src_top[qs, :kp])
    win_union = len(win_ids)
    strat_per_query = strat_per_query if strat_per_query is not None else kp
    out = {"kprime": int(kp), "rho_true": rho_true, "n_queries": int(len(qs)),
           "n_queries_requested": int(n_queries),
           "window_encodings": int(win_union), "estimates": []}
    for frac in budgets_frac:
        m = max(1, int(frac * p.N))
        shared = rng.choice(p.N, size=min(m, p.N), replace=False)
        for kind in kinds:
            est = [estimate_miss(p, qi, kp, shared, kind, rng, strat_per_query) for qi in qs]
            mh = np.array([e["miss_hat"] for e in est]) / K
            bh = np.array([e["bound_hat"] for e in est]) / K
            total = len(np.unique(np.concatenate([win_ids, shared] + [e["extra"] for e in est])))
            out["estimates"].append({
                "kind": kind, "budget_frac": frac, "sample_m": int(len(shared)),
                "rho_hat": float((w * mh).sum()),
                "rho_bound": float((w * bh).sum()),
                "abs_err": float(abs((w * mh).sum() - rho_true)),
                "per_query_mae": float(np.mean(np.abs(mh - true))),
                "total_encodings": int(total),
                "total_encodings_frac": float(total / p.N),
            })
    return out
