"""A model pair: cached encodings, qrels, workload, and precomputed rankings.

Pair directory layout (all document/query indices are 0-based row indices):
  doc_src.npy   [N, ds] float32   source-model document vectors
  doc_tgt.npy   [N, dt] float32   target-model document vectors
  q_src.npy     [Q, ds] float32   source-model query vectors
  q_tgt.npy     [Q, dt] float32   target-model query vectors
  qrels.tsv     "qidx<TAB>docidx<TAB>rel" (graded or binary)
  bm25.npy      [Q, Kb] int64     optional, BM25 top-Kb doc indices per query
  doc_len.npy   [N]               optional, document lengths (for O-size)
  weights.npy   [Q]               optional, workload weights (e.g. ORCAS frequency)
  meta.json                       optional, free-form (model names, corpus, ...)
Vectors are L2-normalised on load; scores are inner products (= cosine).

Query roles (no evaluation leak):
  eval_q        judged queries held out for evaluation (fraction `eval_frac` of the
                primary-judged queries, seeded); the secondary set (e.g. TREC-DL,
                `secondary_eval_qids`) is always evaluation-only.
  traffic_pool  every other query: the "past traffic" that warms the cache.
A traffic draw comes from the eval set with probability `repeat_rate` (re-asked
queries) and from the traffic pool otherwise. repeat_rate = 0 means no evaluated query
is ever served before it is evaluated. eval_frac = 0 reproduces the old leaky design
(eval = traffic = every judged query) and is flagged as such.

n_eval_queries (development runs) subsamples the eval set AFTER the traffic pool is fixed.
Held-out queries that are not sampled join neither role: the traffic pool is identical to
the full run, so a development run differs from it only in how many queries are evaluated.
"""
import json
import os

import numpy as np

from .compat import FITTERS, compatibility, relative_residual
from .search import normalize_rows, topk_ip


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q, d, r = line.split()[:3]
            qrels.setdefault(int(q), {})[int(d)] = int(r)
    return qrels


def load_vectors(path, check_rows=2000, tol=1e-3):
    """Memory-map vectors; normalise (which copies into RAM) only if they are not already
    unit-norm. encode.py writes normalised vectors, so real pairs stay memory-mapped."""
    X = np.load(path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"{path} is {X.ndim}-D with shape {X.shape}; expected a 2-D matrix of "
                         f"row vectors. A 1-D array would otherwise reach np.linalg.norm(axis=1) "
                         f"and be reported as an AxisError against numpy, not against this file.")
    if X.shape[0] == 0:
        # np.all over an empty array is vacuously True, so an empty encoding would pass the
        # unit-norm check below and come back as an empty memmap; the first complaint would
        # then be compat.py asking for 2 documents, pointing at the wrong file entirely
        raise ValueError(f"{path} has 0 rows: an empty encoding is a broken encode, not a "
                         f"corpus of nothing. Re-run encode.py and check it with "
                         f"scripts/encode_check.py.")
    idx = np.linspace(0, X.shape[0] - 1, min(check_rows, X.shape[0])).astype(int)
    norms = np.linalg.norm(np.asarray(X[idx], dtype=np.float32), axis=1)
    if X.dtype == np.float32 and np.all(np.abs(norms - 1) < tol):
        return X
    return normalize_rows(np.asarray(X))


def make_weights(n_queries, workload, seed=0, path=None):
    """Workload weights over queries: 'uniform', 'zipf:<alpha>', or 'file'."""
    if workload == "file":
        w = np.load(path).astype(np.float64)
    elif workload == "uniform":
        w = np.ones(n_queries)
    elif workload.startswith("zipf:"):
        alpha = float(workload.split(":")[1])
        ranks = np.empty(n_queries)
        ranks[np.random.default_rng(seed).permutation(n_queries)] = np.arange(1, n_queries + 1)
        w = ranks ** (-alpha)
    else:
        raise ValueError(workload)
    return w / w.sum()


class Pair:
    def __init__(self, pair_dir, cfg):
        self.dir, self.cfg = pair_dir, cfg
        ld = lambda n: np.load(os.path.join(pair_dir, n))
        lv = lambda n: load_vectors(os.path.join(pair_dir, n))
        self.doc_src = lv("doc_src.npy")
        self.doc_tgt = lv("doc_tgt.npy")
        for n in ("q_src.npy", "q_tgt.npy"):   # queries are small and loaded into RAM; same checks as docs
            shp = np.load(os.path.join(pair_dir, n), mmap_mode="r").shape
            if len(shp) != 2 or shp[0] == 0 or shp[1] == 0:
                raise ValueError(f"{os.path.join(pair_dir, n)}: expected a non-empty 2-D array, got shape {shp}")
        self.q_src = normalize_rows(ld("q_src.npy"))
        self.q_tgt = normalize_rows(ld("q_tgt.npy"))
        self.N, self.Q = self.doc_src.shape[0], self.q_src.shape[0]
        # not an assert: `python -O` strips asserts, and misaligned rows give plausible wrong numbers
        if self.doc_tgt.shape[0] != self.N or self.q_tgt.shape[0] != self.Q:
            raise ValueError(f"{pair_dir}: source/target rows differ (docs {self.N} vs "
                             f"{self.doc_tgt.shape[0]}, queries {self.Q} vs {self.q_tgt.shape[0]})")
        self.qrels = load_qrels(os.path.join(pair_dir, "qrels.tsv"))
        opt = lambda n: ld(n) if os.path.exists(os.path.join(pair_dir, n)) else None
        self.bm25 = opt("bm25.npy")
        self.doc_len = opt("doc_len.npy")
        wpath = os.path.join(pair_dir, "weights.npy")
        workload = cfg.get("workload", "uniform")
        if os.path.exists(wpath) and workload != "file":
            raise ValueError(f"{wpath} exists but workload={workload!r} would ignore it; set "
                             f"workload='file' to use it (e.g. ORCAS clicks) or remove the file")
        self.weights = make_weights(self.Q, workload, cfg.get("seed", 0), wpath)
        mp = os.path.join(pair_dir, "meta.json")
        self.meta = json.load(open(mp)) if os.path.exists(mp) else {}
        rng = np.random.default_rng(cfg.get("seed", 0))
        judged = np.array(sorted(q for q in self.qrels if q < self.Q), dtype=np.int64)
        sec_path = cfg.get("secondary_eval_qids")
        if sec_path and not os.path.exists(sec_path):
            raise FileNotFoundError(f"secondary_eval_qids={sec_path} does not exist; refusing to fall "
                                    f"back silently (the secondary metric would be computed on the "
                                    f"primary queries instead)")
        self.secondary_q = (np.loadtxt(sec_path, dtype=np.int64, ndmin=1)
                            if sec_path else np.empty(0, np.int64))
        bad = self.secondary_q[(self.secondary_q < 0) | (self.secondary_q >= self.Q)]
        if len(bad):
            raise ValueError(f"secondary_eval_qids {sec_path} has {len(bad)} query indices outside "
                             f"[0, {self.Q}) (e.g. {bad[:3].tolist()}): wrong file for this pair?")
        primary_judged = np.setdiff1d(judged, self.secondary_q)  # dev-style queries only
        frac = cfg.get("eval_frac", 0.5)
        self.leaky = not frac
        if self.leaky:
            self.eval_q = primary_judged
            self.traffic_pool = np.arange(self.Q)
        else:
            perm = rng.permutation(primary_judged)
            self.eval_q = np.sort(perm[:max(1, int(round(frac * len(perm))))])
            held = np.union1d(self.eval_q, self.secondary_q)
            self.traffic_pool = np.setdiff1d(np.arange(self.Q), held)
        n_eval = cfg.get("n_eval_queries")  # after traffic_pool on purpose: see module docstring
        if n_eval and n_eval < len(self.eval_q):
            self.eval_q = np.sort(rng.choice(self.eval_q, n_eval, replace=False))
        if len(self.traffic_pool) == 0 and cfg.get("replay", True):
            raise ValueError("no traffic queries left after holding out the eval set; "
                             "lower eval_frac or add unjudged traffic queries")
        if self.weights[self.eval_q].sum() <= 0:
            raise ValueError("every evaluation query has weight 0 under this workload; rho, G, "
                             "quality and the probe would all be NaN")

    def traffic_weights(self, repeat_rate=0.0):
        """Per-query draw probabilities for the traffic stream (sums to 1)."""
        if not 0.0 <= repeat_rate <= 1.0:
            raise ValueError(f"repeat_rate must be in [0, 1], got {repeat_rate}")
        if self.leaky:
            return self.weights / self.weights.sum()
        w = np.zeros(self.Q)
        tp = self.weights[self.traffic_pool]
        if repeat_rate < 1:
            if tp.sum() <= 0:
                raise ValueError("every traffic-pool query has weight 0 in weights.npy; "
                                 "the traffic stream would be empty")
            w[self.traffic_pool] = (1 - repeat_rate) * tp / tp.sum()
        if repeat_rate > 0:
            ev = self.weights[self.eval_q]
            if ev.sum() <= 0:
                raise ValueError("every eval query has weight 0, so repeat_rate > 0 cannot re-ask any")
            w[self.eval_q] += repeat_rate * ev / ev.sum()
        return w / w.sum()

    def all_query_weights(self):
        """Draw probabilities over every SERVABLE query - the traffic pool and the primary
        evaluation queries together - under the workload (sums to 1).

        Used by replay v3 (`eval_arrival: "first_service"`). There an evaluation query may be
        in the stream, because it is measured at its FIRST service and never scored again, so
        it cannot be warmed by its own repair before being evaluated. `repeat_rate` is not
        used in v3: the mix of evaluation and other queries is the workload's own, not a knob.

        The secondary (TREC-DL) evaluation set is excluded here, since DESIGN.md makes it
        evaluation-only. A consequence: under v3 the secondary metric has no arrival values
        and is reported from the fully-migrated state only.
        """
        w = np.asarray(self.weights, float).copy()
        if len(self.secondary_q):
            w[self.secondary_q] = 0.0
        s = w.sum()
        if s <= 0:
            raise ValueError("every servable query has weight 0 under this workload; the v3 "
                             "arrival stream would be empty")
        return w / s

    # ------------------------------------------------------------------
    def kprime_list(self):
        """Window depths from fractions of N (cfg 'kprime_frac') or absolute ('kprime')."""
        if "kprime" in self.cfg:
            ks = self.cfg["kprime"]
        else:
            ks = [max(self.cfg["K"], int(round(f * self.N))) for f in self.cfg["kprime_frac"]]
        ks = [int(k) for k in ks]
        # A depth of 0 or below is a config error, not a degenerate window. It survives
        # min(k, N) unchanged and reaches topk_ip as k=0, where np.argpartition(-sc, k-1)
        # partitions on index -1 -- the last element -- so the call returns something rather
        # than failing, and the window measured is not the window requested.
        bad = sorted(k for k in ks if k <= 0)
        if bad:
            raise ValueError(f"window depth must be positive, got {bad} in "
                             f"{'kprime' if 'kprime' in self.cfg else 'kprime_frac'}="
                             f"{self.cfg.get('kprime', self.cfg.get('kprime_frac'))}")
        return sorted(set(min(k, self.N) for k in ks))

    def precompute(self):
        cfg, K = self.cfg, self.cfg["K"]
        ks = self.kprime_list()
        if cfg.get("tgt_depth", 1000) < K:
            raise ValueError(f"tgt_depth={cfg.get('tgt_depth')} < K={K}: the target top-K cannot be formed")
        self.Kw = int(min(cfg.get("window", ks[0]), self.N))  # depth used by strategies
        kmax = max(ks + [self.Kw])
        c = cfg.get("probe_c", 4)
        # one retrieval at the deepest needed depth; smaller windows are prefixes (nested).
        # The probe's stratum needs c * (strategy window), not c * (largest K' measured).
        self.src_top, _ = topk_ip(self.q_src, self.doc_src, min(self.N, max(kmax, c * self.Kw)))
        self.tgt_top, self.tgt_sc = topk_ip(self.q_tgt, self.doc_tgt,
                                            min(self.N, cfg.get("tgt_depth", 1000)))
        # kappa is a MAP COORDINATE: always semi-orthogonal Procrustes on documents,
        # whatever adapter the S1/S3 strategies use, so it is comparable across arms.
        proc, self.compat = compatibility(
            self.doc_src, self.doc_tgt, "procrustes",
            cfg.get("adapter_n_fit", 10000), cfg.get("adapter_n_eval", 5000),
            cfg.get("seed", 0))
        self.adapter, self.adapter_info = self._fit_strategy_adapter(proc)
        self.q_map = self.adapter(self.q_tgt)
        self.map_top, _ = topk_ip(self.q_map, self.doc_src, min(self.N, kmax))
        return self

    def _fit_strategy_adapter(self, proc):
        """Adapter used by S1/S3. `adapter` picks the family (any key of compat.FITTERS);
        `adapter_fit` picks the training pairs: 'docs' (document pairs), 'queries'
        (traffic-pool query pairs only, never eval queries), or 'both'."""
        cfg = self.cfg
        kind, fit_on = cfg.get("adapter", "procrustes"), cfg.get("adapter_fit", "docs")
        if kind == "procrustes" and fit_on == "docs":
            return proc, {"adapter": kind, "fit_on": fit_on, "same_as_kappa_fit": True}
        rng = np.random.default_rng(cfg.get("seed", 0) + 1)
        n_fit = cfg.get("adapter_n_fit", 10000)
        Xs, Xt = [], []
        if fit_on in ("docs", "both"):
            d = rng.choice(self.N, min(n_fit, self.N), replace=False)
            Xs.append(np.asarray(self.doc_src[d]))
            Xt.append(np.asarray(self.doc_tgt[d]))
        if fit_on in ("queries", "both"):
            q = self.traffic_pool if not self.leaky else np.arange(self.Q)
            q = rng.choice(q, min(n_fit, len(q)), replace=False)
            Xs.append(self.q_src[q])
            Xt.append(self.q_tgt[q])
        if not Xs:
            raise ValueError(f"adapter_fit={fit_on!r}")
        ad = FITTERS[kind](np.vstack(Xt), np.vstack(Xs))
        ev = rng.choice(self.N, min(cfg.get("adapter_n_eval", 5000), self.N), replace=False)
        return ad, {"adapter": kind, "fit_on": fit_on, "n_fit": int(sum(len(x) for x in Xs)),
                    "doc_residual": relative_residual(ad, np.asarray(self.doc_tgt[ev]),
                                                      np.asarray(self.doc_src[ev]))}

    def tscore(self, qi, ids):
        return (self.doc_tgt[ids] @ self.q_tgt[qi]).astype(np.float32)

    def target_topk_batch(self, qids, in_m, k):
        """target_topk_over for many queries at once: prefix path where the precomputed
        target top-`tgt_depth` already holds k migrated docs, and ONE matrix product over the
        migrated docs for the rest (instead of an O(N) scan per query)."""
        out, rest = {}, []
        for qi in qids:
            pref, psc = self.tgt_top[qi], self.tgt_sc[qi]
            mask = in_m[pref]
            if mask.sum() >= k or mask.all():
                out[int(qi)] = (pref[mask][:k], psc[mask][:k])
            else:
                rest.append(int(qi))
        if rest:
            ids = np.flatnonzero(in_m)
            if len(ids) == 0:
                for qi in rest:
                    out[qi] = (ids, np.empty(0, np.float32))
            else:
                D = np.asarray(self.doc_tgt[ids])
                for a in range(0, len(rest), 1024):
                    blk = rest[a:a + 1024]
                    S = (self.q_tgt[blk] @ D.T).astype(np.float32)
                    kk = min(k, len(ids))
                    for row, qi in zip(S, blk):
                        part = np.argpartition(-row, kk - 1)[:kk] if kk < len(ids) else np.arange(len(ids))
                        o = np.lexsort((ids[part], -row[part]))
                        out[qi] = (ids[part][o], row[part][o])
        return out

    def target_topk_over(self, qi, in_m, k):
        """Exact target top-k restricted to migrated docs (in_m mask)."""
        pref, psc = self.tgt_top[qi], self.tgt_sc[qi]
        mask = in_m[pref]
        if mask.sum() >= k or mask.all():
            return pref[mask][:k], psc[mask][:k]
        ids = np.flatnonzero(in_m)
        if len(ids) == 0:
            return ids, np.empty(0, np.float32)
        sc = self.tscore(qi, ids)
        o = np.lexsort((ids, -sc))[:k]
        return ids[o], sc[o]
