"""Expected union-of-windows coverage under i.i.d. arrivals: the cost model for windowed reuse.

Setting. A workload is a distribution p over distinct queries; arrivals are i.i.d. draws from p
(replay v3 draws its stream this way). Query q's source window W_q has K' documents. The WINDOW
MASS of document d is P(d) = sum_q p_q [d in W_q]: the probability that one arrival touches d.

Theorem (exact curve). After m arrivals, the expected fraction of the corpus touched is
    E[cov(m)] = 1 - (1/N) sum_d (1 - P(d))^m.
Each document is untouched after m independent arrivals with probability (1 - P(d))^m.

Theorem (independence bound). sum_d P(d) = K' for EVERY workload and EVERY index, since each
window holds exactly K' documents and p sums to 1. Since x -> (1 - x)^m is convex, Jensen gives
    E[cov(m)] <= 1 - (1 - K'/N)^m,
with equality iff P is constant: the closed form that assumes independent windows is an upper
bound on expected cost-at-service for any workload, and its excess is a Jensen gap driven by the
dispersion of window mass -- i.e. by hubness of the source index.

Under a uniform workload over Q distinct queries P(d) = c(d)/Q, where c(d) is the number of
queries whose window contains d, so the curve needs only the MULTIPLICITY HISTOGRAM
h[c] = |{d : c(d) = c}|, a statistic computed in one retrieval pass.
"""
import numpy as np


def expected_coverage_from_mass(P, m, N=None):
    """E[cov(m)] for window masses P (array over documents, zeros allowed) and arrivals m."""
    P = np.asarray(P, float); m = np.atleast_1d(np.asarray(m, float))
    N = len(P) if N is None else N
    nz = P[P > 0]
    if len(nz) and nz.max() > 1 + 1e-9:
        raise ValueError("window mass above 1: P(d) is a probability per arrival")
    # untouched mass: docs with P = 0 contribute 1 each; others (1-P)^m. log1p for stability.
    L = np.log1p(-np.minimum(nz, 1 - 1e-15))
    untouched = (N - len(nz)) + np.exp(np.outer(m, L)).sum(1)
    return 1.0 - untouched / N


def expected_coverage_from_hist(hist, Q, m, N):
    """Uniform workload over Q distinct queries: hist maps multiplicity c -> number of docs."""
    m = np.atleast_1d(np.asarray(m, float))
    cs = np.array([int(c) for c in hist if int(c) > 0], float)
    ns = np.array([hist[c] for c in hist if int(c) > 0], float)
    if cs.size and cs.max() > Q:
        raise ValueError(f"multiplicity {cs.max()} exceeds the number of queries {Q}")
    untouched = (N - ns.sum()) + (ns[None, :] * np.exp(np.outer(m, np.log1p(-np.minimum(cs / Q, 1 - 1e-15))))).sum(1)
    return 1.0 - untouched / N


def independent_bound(kprime, N, m):
    m = np.atleast_1d(np.asarray(m, float))
    return 1.0 - (1.0 - kprime / N) ** m


def mass_dispersion(P, N=None):
    """Summary of how unevenly window mass is spread: the quantity the Jensen gap depends on."""
    P = np.asarray(P, float); N = len(P) if N is None else N
    mu = P.sum() / N
    var = (np.square(P).sum() / N) - mu ** 2
    nz = P[P > 0]
    top1 = np.sort(nz)[::-1][:max(1, int(0.01 * N))].sum() / max(P.sum(), 1e-300)
    return {"mean": mu, "cv": float(np.sqrt(max(var, 0)) / mu) if mu > 0 else None,
            "frac_docs_touched_by_support": len(nz) / N, "mass_in_top1pct_docs": float(top1)}
