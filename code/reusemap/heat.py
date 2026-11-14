"""RQ4: dense-only heat and the AUC gate.

kocc_s(d) / kocc_t(d): workload-weighted number of queries whose source / target top-K
contains d.  dense_only(d): weighted number of queries for which d is in the source top-K
but NOT in the BM25 top-Kb.

To avoid a magnitude confound (documents with large kocc_s mechanically show large
absolute change), the gate uses fractions:
  signal = dense_only / kocc_s          (share of d's exposure with no lexical support)
  label  = |kocc_t - kocc_s| / kocc_s > median   (relative exposure change)
over documents with kocc_s > 0.  The gate opens only if the bootstrap 95% CI of the
AUC excludes 0.5.  kocc_t needs target vectors, so in deployment it is computed on the
probe sample only; offline we can also compute it on all documents.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def exposure(top, K, weights, N):
    e = np.zeros(N)
    for qi in range(top.shape[0]):
        e[top[qi, :K]] += weights[qi]
    return e


def dense_only(p, K, Kb, weights=None):
    w = p.weights if weights is None else weights
    h = np.zeros(p.N)
    for qi in np.flatnonzero(w > 0):
        s = p.src_top[qi, :K]
        h[s[~np.isin(s, p.bm25[qi, :Kb])]] += w[qi]
    return h


def heat_gate(p, sample_docs=None, n_boot=1000, seed=0, weights=None):
    """weights: the traffic distribution heat is measured on (default p.weights).
    The runner passes the traffic weights, so heat never comes from evaluation queries
    unless they are re-asked (repeat_rate > 0)."""
    w = p.weights if weights is None else weights
    if p.bm25 is None:
        return {"available": False, "reason": "no bm25.npy"}
    K = p.cfg["K"]
    Kb = p.cfg.get("bm25_heat_depth", K)
    assert Kb <= p.bm25.shape[1], f"bm25_heat_depth={Kb} exceeds bm25.npy depth {p.bm25.shape[1]}"
    ks = exposure(p.src_top, K, w, p.N)
    kt = exposure(p.tgt_top, K, w, p.N)
    h = dense_only(p, K, Kb, w)
    docs = np.flatnonzero(ks > 0)
    if sample_docs is not None:
        docs = np.intersect1d(docs, sample_docs)
    if len(docs) < 2:
        return {"available": False, "reason": f"only {len(docs)} exposed documents", "gate_open": False}
    sig = h[docs] / ks[docs]
    rel = np.abs(kt[docs] - ks[docs]) / ks[docs]
    med = np.median(rel)
    lab, label_rule = (rel > med).astype(int), "rel > median"
    if lab.min() == lab.max():  # heavy ties at the median (e.g. many docs at rel = 1)
        lab, label_rule = (rel >= med).astype(int), "rel >= median (ties at the median)"
    out = {"available": True, "n_docs": int(len(docs)), "signal_scores": h / np.maximum(ks, 1e-12),
           "label_rule": label_rule, "label_median": float(med), "positive_frac": float(lab.mean())}
    if lab.min() == lab.max():
        out.update(auc=float("nan"), ci=[float("nan")] * 2, gate_open=False)
        return out
    auc = roc_auc_score(lab, sig)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(docs), len(docs))
        if lab[i].min() != lab[i].max():
            boots.append(roc_auc_score(lab[i], sig[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out.update(auc=float(auc), ci=[float(lo), float(hi)], gate_open=bool(lo > 0.5))
    return out
