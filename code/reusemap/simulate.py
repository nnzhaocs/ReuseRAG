"""Replay of migration strategies along the encoding-cost axis.

State: a boolean mask in_m over documents that already have a target vector.
Cost C = |in_m| (every target encoding is counted exactly once, whether it came from
background migration or on-demand window repair).

Each round:
  1. a stream of workload queries is served (window strategies may encode missing
     window documents: 'sync' encodes them before ranking, 'async' queues them ahead
     of background work);
  2. background migration encodes up to b = ceil(N / rounds) documents
     (async queue first, then the background order; O-demand has no background order);
  3. a read-only checkpoint evaluates every metric on the evaluation queries.

Checkpoints are STATE-BASED by default: a query whose window is fully encoded ("warm")
is ranked by target scores over its window plus the target index over migrated docs; a
query with any unencoded window document ("cold") is served its nomination-order answer
(the old system's ranking for S2/S4/S5, the mapped ranking for S3). Mixing target-scored
and unscored documents in one ranking compares scores across spaces and is avoided. Scoring not-yet-encoded window docs at a checkpoint would give window
strategies target quality at C = 0 without counting those encodings. 'sync' and 'async'
therefore differ only in the stream: sync encodes a served query's whole window inline
(counted immediately; its per-query synchronous encodings are reported as a latency
proxy), async queues the missing docs ahead of background work. checkpoint='served'
reproduces the user-visible sync ranking for diagnostics only.

Strategies
  S0 blue-green rebuild  : source ranking until all docs migrated, then target.
  S1 space mapping       : adapter-mapped query over the old index until complete, then target.
  S2 window re-rank      : source window, ranked by target scores.
  S3 mapped window       : window nominated by the adapter-mapped query.
  S4 window U BM25       : source window plus BM25 candidates.
  S5 window U mapped U BM25.
All window strategies also search the target index over migrated docs (new writes and
migrated docs can only be found there).
"""
import math
from collections import deque

import numpy as np

from .measure import _union_rank
from .metrics import METRICS

WINDOW_STRATEGIES = ("S2", "S3", "S4", "S5")

# Orders whose background work is held back until every evaluation query has been served.
# 'demand_then_rand' is the cost-minimal admissible policy under the v3 objective: it lets
# demand warm the windows that are actually needed, then completes the migration (so unlike
# 'demand' it does complete, and so stays clear of Thaw's C2 'abandon completion' question).
ORDERS_DELAYED = ("demand_then_rand",)


def order_delays_background(order):
    return order in ORDERS_DELAYED


def _ordered_unique(*arrs):
    cat = np.concatenate(arrs)
    _, first = np.unique(cat, return_index=True)
    return cat[np.sort(first)]


def build_windows(p, strategy, queries):
    Kw, Kb = p.Kw, p.cfg.get("bm25_depth", p.Kw)
    out = {}
    for qi in queries:
        src = p.src_top[qi, :Kw]
        if strategy == "S2":
            w = src
        elif strategy == "S3":
            w = p.map_top[qi, :Kw]
        elif strategy == "S4":
            w = _ordered_unique(src, p.bm25[qi, :Kb])
        elif strategy == "S5":
            w = _ordered_unique(src, p.map_top[qi, :Kw], p.bm25[qi, :Kb])
        else:
            raise ValueError(strategy)
        out[int(qi)] = w
    return out


def background_order(p, order, rng, signal_scores=None):
    N = p.N
    if order == "rand":
        return rng.permutation(N)
    if order == "size":
        if p.doc_len is None:
            return None
        return np.lexsort((rng.permutation(N), p.doc_len))  # shortest first, random ties
    if order == "signal":
        if signal_scores is None:
            return None
        tie = rng.permutation(N)
        return np.lexsort((tie, -signal_scores))
    if order == "demand":
        return np.empty(0, np.int64)
    if order == "demand_then_rand":
        return rng.permutation(N)  # held back until all arrivals are done (see ORDERS_DELAYED)
    raise ValueError(order)


def expected_arrival_rounds(arrival_w, can_arrive, stream_len, target_frac=0.99,
                            max_rounds=10 ** 7, weighted=False):
    """Rounds until `target_frac` of `can_arrive` has been served at least once, in expectation.

    Draws are with replacement from `arrival_w`, so this is a (weighted) coupon-collector
    quantity, NOT len(can_arrive)/stream_len: after T rounds the probability that query q has
    been served is 1 - (1 - w_q) ** (T * stream_len).

    `weighted` picks what "a fraction of the population" means, and under a skewed workload the
    two answers differ by orders of magnitude. With a 35,000x spread between the heaviest and
    lightest evaluation query, 99% of the QUERY COUNT takes ~30,000 rounds (the tail is almost
    never drawn) while 99% of the WORKLOAD WEIGHT is served by round ~100. Every quality number
    here is workload-weighted, so `weighted=True` is the notion that matches the measurement and
    is what the runner gates on; the count is reported alongside it as a diagnostic.
    """
    ca = np.asarray(can_arrive)
    if ca.dtype == bool:
        # `can_arrive` is a list of INDICES, not a mask -- both callers pass one (runner.py
        # builds a sorted index array, predict.py an arange). A bool mask would survive the
        # int64 cast below as [1,1,1,...], i.e. element 1 repeated, and return a plausible
        # wrong number: on a 4-query example, 6 rounds instead of 13.
        raise ValueError("can_arrive must be query INDICES, not a boolean mask; pass "
                         "np.flatnonzero(mask) instead")
    w = np.asarray(arrival_w, float)[ca.astype(np.int64)]
    if len(w) == 0:
        return 0
    tot = w.sum()
    if weighted and tot > 0:
        frac = lambda T: float((w * (1.0 - np.power(1.0 - w, max(T, 0) * stream_len))).sum() / tot)
    else:
        frac = lambda T: float(np.mean(1.0 - np.power(1.0 - w, max(T, 0) * stream_len)))
    if frac(max_rounds) < target_frac:
        return math.inf
    lo, hi = 0, 1
    while frac(hi) < target_frac:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if frac(mid) < target_frac:
            lo = mid
        else:
            hi = mid
    return hi


class Replay:
    def __init__(self, p, metrics):
        self.p = p
        self.metrics = metrics  # list of (name, metric, thr, eval_query_array)
        self.K = p.cfg["K"]
        all_eval = np.unique(np.concatenate([m[3] for m in metrics]))
        self.eval_q = all_eval
        self._win = {}
        self._wsc = {}

    def windows(self, strategy):
        if strategy not in self._win:
            p = self.p
            self._win[strategy] = build_windows(p, strategy, range(p.Q))
            self._wsc[strategy] = {int(qi): p.tscore(qi, self._win[strategy][int(qi)])
                                   for qi in self.eval_q}
        return self._win[strategy], self._wsc[strategy]

    # ---------------------------------------------------------------
    def rank(self, strategy, qi, in_m, complete, mode):
        p, K = self.p, self.K
        if strategy == "S0":
            return p.tgt_top[qi, :K] if complete else p.src_top[qi, :K]
        if strategy == "S1":
            return p.tgt_top[qi, :K] if complete else p.map_top[qi, :K]
        win, wsc = self.windows(strategy)
        W, sc = win[qi], wsc[qi]
        cache = getattr(self, "_mtop", None)
        mids, msc = cache[qi] if cache is not None and qi in cache else p.target_topk_over(qi, in_m, K)
        if mode == "sync":
            return _union_rank(np.concatenate([W, mids]), np.concatenate([sc, msc]), K)
        if in_m[W].all():  # warm
            return _union_rank(np.concatenate([W, mids]), np.concatenate([sc, msc]), K)
        return W[:K]  # cold: nomination-order answer

    def checkpoint(self, strategy, in_m, mode, how="state"):
        p = self.p
        complete = bool(in_m.all())
        rmode = "sync" if (how == "served" and mode == "sync") else "async"
        self._mtop = None
        if strategy in WINDOW_STRATEGIES and hasattr(p, "target_topk_batch"):
            self._mtop = p.target_topk_batch([int(q) for q in self.eval_q], in_m, self.K)
        ranks = {int(qi): self.rank(strategy, int(qi), in_m, complete, rmode) for qi in self.eval_q}
        self._mtop = None
        out = {}
        for name, metric, thr, evq in self.metrics:
            f = METRICS[metric]
            num = den = 0.0
            for qi in evq:
                rels = p.qrels.get(int(qi))
                if not rels:
                    continue
                v = f(ranks[int(qi)], rels, self.K, thr)
                if v is None:
                    continue
                w = float(p.weights[qi])
                num += w * v
                den += w
            out[name] = num / den if den else float("nan")
        return out

    # ---------------------------------------------------------------
    def run(self, strategy, order, mode="sync", rounds=40, stream_len=None, seed=0,
            signal_scores=None, max_rounds=None, how="state", traffic_w=None):
        """traffic_w: per-query draw probabilities for the served stream (Pair.traffic_weights).
        Defaults to the held-out-safe distribution with repeat_rate = 0."""
        p = self.p
        if order_delays_background(order):
            return None  # v3-only order; it needs arrival semantics (see run_arrivals)
        rng = np.random.default_rng(seed)
        # A separate generator for the served stream, so the traffic sequence is identical for
        # every strategy and order. Sharing one generator made the stream depend on how many
        # draws background_order had consumed first ('demand' consumes none, 'rand' consumes a
        # permutation), i.e. different orders were compared on different traffic.
        rng_stream = np.random.default_rng((seed, 1))
        N = p.N
        b = int(math.ceil(N / rounds))
        stream_len = stream_len if stream_len is not None else max(1, p.Q // 4)
        if traffic_w is None:
            traffic_w = p.traffic_weights(0.0) if hasattr(p, "traffic_weights") else p.weights
        self.served = []  # query ids served, for audit (tests check no eval leak)
        max_rounds = max_rounds or 4 * rounds
        bg = background_order(p, "rand" if strategy in ("S0", "S1") else order, rng, signal_scores)
        if bg is None:
            return None
        in_m = np.zeros(N, bool)
        queued = np.zeros(N, bool)
        queue = deque()
        ptr = 0
        traj = [(0, self.checkpoint(strategy, in_m, mode, how))]
        self.sync_encodings = []  # per served query, encodings done before ranking
        windowed = strategy in WINDOW_STRATEGIES
        win = self.windows(strategy)[0] if windowed else None
        for r in range(1, max_rounds + 1):
            if windowed:
                drawn = rng_stream.choice(p.Q, stream_len, p=traffic_w)
                self.served.append(drawn)
                for qi in drawn:
                    W = win[int(qi)]
                    miss = W[~in_m[W]]
                    if mode == "sync":
                        in_m[miss] = True
                        self.sync_encodings.append(len(miss))
                    else:
                        new = miss[~queued[miss]]
                        queued[new] = True
                        queue.extend(new.tolist())
            budget = b
            while budget > 0 and queue:  # async repairs first
                d = queue.popleft()
                if not in_m[d]:
                    in_m[d] = True
                    budget -= 1
            while budget > 0 and ptr < len(bg):
                d = bg[ptr]
                ptr += 1
                if not in_m[d]:
                    in_m[d] = True
                    budget -= 1
            traj.append((int(in_m.sum()), self.checkpoint(strategy, in_m, mode, how)))
            if in_m.all():
                break
            if not windowed and ptr >= len(bg):
                break
            if order == "demand" and not queue and traj[-1][0] == traj[-2][0] and r > rounds:
                break  # pure lazy has stopped making progress
        return traj

    # ---------------------------------------------------------------
    def _accumulate(self, qi, ranked, metric_sets, acc, tgt_acc, src_acc):
        """Add one arrival's delivered value, plus the value the fully-migrated target system
        and the system being replaced would have given THE SAME query, to the running totals.

        The two references are accumulated over exactly the queries that arrived, so the
        comparison is never a restricted subset against a whole-eval-set constant: M_T and M_S
        are means over all evaluation queries, and the arrived subset's means differ from them.
        """
        p = self.p
        rels = p.qrels.get(int(qi))
        if not rels:
            return
        w = float(p.weights[qi])
        for name, metric, thr, qset in metric_sets:
            if int(qi) not in qset:
                continue
            f = METRICS[metric]
            v = f(ranked, rels, self.K, thr)
            t = f(p.tgt_top[qi, :self.K], rels, self.K, thr)
            s = f(p.src_top[qi, :self.K], rels, self.K, thr)
            if v is None or t is None or s is None:
                continue
            acc[name][0] += w * v
            acc[name][1] += w
            tgt_acc[name][0] += w * t
            tgt_acc[name][1] += w
            src_acc[name][0] += w * s
            src_acc[name][1] += w

    def run_arrivals(self, strategy, order, rounds=100, stream_len=None, seed=0,
                     signal_scores=None, max_rounds=None, arrival_w=None,
                     per_query_sample=200, per_query_every=None):
        """Replay v3: every evaluation query is measured ONCE, when it is first served.

        Why this shape (2026-10-29):
          * The served stream is drawn from the whole workload (traffic pool AND evaluation
            queries) with its own generator, so the arrival sequence is IDENTICAL for every
            strategy and background order. All combinations are therefore compared on the same
            queries, in the same order, at the same points of the migration.
          * A window strategy repairs a served query's missing window documents inline. Those
            encodings persist in in_m and are counted in C exactly like background work: no
            encoding is ever free (DECISIONS.md row 13).
          * An evaluation query is measured at its FIRST service only, so no query is ever
            scored after its own repair. That is what made v1's O-demand win by construction
            (DECISIONS.md row 17); here demand gets no such advantage.
          * Checkpoints are no longer round boundaries: the measurement points are arrivals.
            v2's state-based checkpoint re-scored all evaluation queries every round, which
            also meant a window was only usable once ALL of its documents happened to be
            migrated - probability f**K', i.e. 50/50 only at f = 0.5**(1/K') (0.9986 at
            K'=500). That is why every v2 trajectory was flat until ~95% of N.

        Returns a dict of raw measurements (see summarize_arrivals), or None if the order is
        not available for this pair (e.g. 'size' without doc lengths).
        """
        p = self.p
        N = p.N
        self._mtop = None
        if arrival_w is None:
            arrival_w = (p.all_query_weights() if hasattr(p, "all_query_weights") else p.weights)
        arrival_w = np.asarray(arrival_w, float)
        rng = np.random.default_rng(seed)             # background order only
        rng_stream = np.random.default_rng((seed, 1))  # served stream: order-independent
        b = int(math.ceil(N / rounds))
        stream_len = int(stream_len if stream_len is not None else max(1, p.Q // 4))
        max_rounds = int(max_rounds or 4 * rounds)
        bg = background_order(p, "rand" if strategy in ("S0", "S1") else order, rng, signal_scores)
        if bg is None:
            return None
        delay_bg = order_delays_background(order) and strategy not in ("S0", "S1")
        windowed = strategy in WINDOW_STRATEGIES
        win = self.windows(strategy)[0] if windowed else None
        in_m = np.zeros(N, bool)
        can_arrive = np.array(sorted(int(q) for q in self.eval_q if arrival_w[int(q)] > 0), np.int64)
        pending = set(int(q) for q in can_arrive)
        metric_sets = [(name, metric, thr, set(int(q) for q in evq))
                       for name, metric, thr, evq in self.metrics]
        acc = {name: [0.0, 0.0] for name, _m, _t, _s in metric_sets}
        tgt_acc = {name: [0.0, 0.0] for name, _m, _t, _s in metric_sets}
        src_acc = {name: [0.0, 0.0] for name, _m, _t, _s in metric_sets}
        arrivals, traj, pq_curve = [], [], []
        unpaid = 0  # rankings that scored a document with a target vector it had not been given
        repairs_all = 0  # inline encodings over EVERY served query, not only measured arrivals
        n_pre_complete = 0  # arrivals served while the migration was still incomplete
        # Workload weight served, not just the count: every quality number is weight-weighted,
        # and under a skewed workload the two differ by orders of magnitude (see
        # expected_arrival_rounds). The runner gates on the weight.
        arrived_weight = 0.0
        expected_weight = float(np.asarray(p.weights, float)[can_arrive].sum()) if len(can_arrive) else 0.0
        # its own generator, so the sampled queries do not depend on how many draws
        # background_order consumed first - the curve stays comparable across orders
        pq_sample = (np.random.default_rng((seed, 2))
                     .choice(can_arrive, min(per_query_sample, len(can_arrive)), replace=False)
                     if windowed and len(can_arrive) else np.empty(0, np.int64))
        ptr = 0
        last_arrival_C = None
        r = 0
        # ~200 points on the read-only per-query cost curve, however long the run is
        pq_every = int(per_query_every or max(1, max_rounds // 200))
        for r in range(1, max_rounds + 1):
            if len(pq_sample) and (r - 1) % pq_every == 0:
                # read-only: what it would cost RIGHT NOW to serve a fixed sample of queries at
                # target quality. Never changes state; this is the per-query cost axis and it is
                # what makes the (1 - f) * K' prediction checkable.
                need = [int((~in_m[win[int(q)]]).sum()) for q in pq_sample]
                pq_curve.append((int(in_m.sum()), float(np.mean(need))))
            drawn = rng_stream.choice(p.Q, stream_len, p=arrival_w)
            for q in drawn:
                qi = int(q)
                nrep = 0
                if windowed:
                    W = win[qi]
                    miss = W[~in_m[W]]
                    nrep = int(len(miss))
                    in_m[miss] = True  # charged: C = |in_m|
                    repairs_all += nrep
                if qi in pending:
                    pending.discard(qi)
                    arrived_weight += float(p.weights[qi])
                    last_arrival_C = int(in_m.sum())
                    was_complete = bool(in_m.all())
                    n_pre_complete += 0 if was_complete else 1
                    ranked = self.rank(strategy, qi, in_m, was_complete, "async")
                    if windowed and len(ranked) and not in_m[ranked].all():
                        unpaid += 1  # would violate DECISIONS.md row 13; checked by smoke.py
                    self._accumulate(qi, ranked, metric_sets, acc, tgt_acc, src_acc)
                    arrivals.append((qi, last_arrival_C, nrep))
            if arrivals:
                # recorded after this round's arrivals but BEFORE its background work, so the
                # cost on the x axis is the cost at which that quality was actually served
                traj.append((int(in_m.sum()),
                             {name: (acc[name][0] / acc[name][1] if acc[name][1] else float("nan"))
                              for name, _m, _t, _s in metric_sets}))
            budget = 0 if (delay_bg and pending) else b
            while budget > 0 and ptr < len(bg):
                d = bg[ptr]
                ptr += 1
                if not in_m[d]:
                    in_m[d] = True
                    budget -= 1
            if not pending and (in_m.all() or ptr >= len(bg)):
                break
        return {"arrivals": arrivals, "traj": traj, "per_query_cost": pq_curve,
                "acc": {k: list(v) for k, v in acc.items()},
                "tgt_acc": {k: list(v) for k, v in tgt_acc.items()},
                "src_acc": {k: list(v) for k, v in src_acc.items()},
                "repairs_all_total": int(repairs_all),
                "hit_round_cap": bool(pending),
                "n_pre_complete": int(n_pre_complete),
                "completed": bool(in_m.all()), "C_final": int(in_m.sum()), "unpaid_scored": int(unpaid),
                "n_expected": int(len(can_arrive)), "n_arrived": int(len(arrivals)),
                "arrived_weight": arrived_weight, "expected_weight": expected_weight,
                "last_arrival_C": last_arrival_C, "rounds_used": int(r),
                "max_rounds": max_rounds}


def summarize_arrivals(raw, key, M_S, M_T, eps, N):
    """Summary of one v3 run for the metric `key`.

    debt  (PRIMARY): weighted mean of (what the fully-migrated target system would have
          delivered for that query) - (what was actually delivered when it arrived). It is the
          quality lost per query during the migration, and it is the only axis that
          discriminates once completion is required: at completion every strategy has encoded
          exactly N documents (in_m flips each document once), so total cost is N for all of
          them. Cost only separates strategies if completion is NOT required, and that
          question - whether never completing is acceptable - is Thaw's C2 and out of scope
          here (DESIGN.md scope fence).
    C_serve (secondary): encodings at the last arrival if the population was served within eps
          of the target, else N. Reported for continuity with the v2 C_eps framing; under a
          completion requirement it is a diagnostic, not the objective.

          It is NOT a saving, and it is not one quantity: three different situations produce a
          C_serve, and `C_serve_basis` says which one. Read the column with the basis beside it,
          or two rows that mean different things get compared as if they were the same number.
            "at_last_arrival"     - the strategy repaired windows and reached target quality;
                                    C_serve is the work it had done by the last arrival. This is
                                    the only basis on which two rows are directly comparable.
            "no_repair_needed"    - it reached target quality having repaired nothing, because
                                    the source was already within eps of the target. C_serve is
                                    then just how far the background happened to get, so it
                                    shrinks as traffic_ratio rises (0.63 of N at ratio 4 down to
                                    0.00 at 256 on the 1M run) while nothing about the strategy
                                    changed. A small number here is a quality concession, not a
                                    cheaper migration.
            "never_reached_target" - charged N by convention, so the column reads 1.0000 whatever
                                    work was actually done.
    C_total (= C_final / N): the documents the strategy ever encodes. Every strategy that
          completes encodes exactly N, so this column is 1.0000 for every admissible row; it is
          here so that a reader comparing C_serve across rows can see that the remainder is paid
          later rather than saved. What incomplete migration would cost instead is Thaw's C2 and
          out of scope (DESIGN.md scope fence).
    """
    num, den = raw["acc"].get(key, [0.0, 0.0])
    tnum, tden = raw["tgt_acc"].get(key, [0.0, 0.0])
    snum, sden = raw.get("src_acc", {}).get(key, [0.0, 0.0])
    served_mean = (num / den) if den else float("nan")
    target_mean = (tnum / tden) if tden else float("nan")
    source_mean = (snum / sden) if sden else float("nan")
    debt = (target_mean - served_mean) if den and tden else float("nan")
    reaches = bool(den and tden and served_mean >= target_mean - eps)
    if not reaches or raw["last_arrival_C"] is None:
        c_serve, c_basis = float(N), "never_reached_target"
    elif raw.get("repairs_all_total", 0) > 0:
        c_serve, c_basis = float(raw["last_arrival_C"]), "at_last_arrival"
    else:
        # reached the target without repairing anything: the source already served within eps,
        # so this number is the background's progress, not the strategy's cost (see the docstring)
        c_serve, c_basis = float(raw["last_arrival_C"]), "no_repair_needed"
    reps = np.array([a[2] for a in raw["arrivals"]], float) if raw["arrivals"] else np.zeros(0)
    pts = [(c, m[key]) for c, m in raw["traj"] if m.get(key) == m.get(key)]  # drop NaN
    out = {
        # M_S/M_T are means over the WHOLE evaluation set; target_mean/source_mean are over the
        # queries that actually arrived. Recorded side by side so a table can show both, and so
        # nothing silently compares a subset mean against a whole-set constant.
        "M_S_all_eval": float(M_S), "M_T_all_eval": float(M_T),
        "served_mean": served_mean, "target_mean": target_mean, "source_mean": source_mean,
        "debt": debt, "debt_vs_source": (served_mean - source_mean) if den and sden else float("nan"),
        "reaches_target": reaches, "C_serve": c_serve, "C_serve_frac": c_serve / N,
        "C_serve_basis": c_basis,
        "completed": raw["completed"], "C_final": raw["C_final"],
        "C_total_frac": raw["C_final"] / N,
        "unpaid_scored": raw.get("unpaid_scored", 0),
        "repairs_all_total": raw.get("repairs_all_total", 0),
        "hit_round_cap": bool(raw.get("hit_round_cap", False)),
        "n_pre_complete": raw.get("n_pre_complete", 0),
        "n_expected": raw["n_expected"], "n_arrived": raw["n_arrived"],
        "eval_served_frac": (raw["n_arrived"] / raw["n_expected"]) if raw["n_expected"] else float("nan"),
        # the weighted share is what the quality numbers are averaged over, so it is the one the
        # runner gates on; the count above is a diagnostic
        "eval_served_weight_frac": ((raw.get("arrived_weight", 0.0) / raw["expected_weight"])
                                    if raw.get("expected_weight") else float("nan")),
        "rounds_used": raw["rounds_used"],
        "running_min": float(min(v for _c, v in pts)) if pts else float("nan"),
        "repairs_total": float(reps.sum()), "repairs_per_query_mean": float(reps.mean()) if len(reps) else 0.0,
        "repairs_per_query_p95": float(np.percentile(reps, 95)) if len(reps) else 0.0,
        "auc_points": pts,
        "auc_range_frac": [pts[0][0] / N, pts[-1][0] / N] if pts else [float("nan")] * 2,
    }
    out["auc"] = auc_over(pts, N, pts[-1][0] if pts else 0.0, pts[0][0] if pts else 0.0)
    return out


def auc_over(pts, N, c_hi, c_lo=0.0):
    """Trapezoid area of quality against C/N over the closed range [c_lo, c_hi], linearly
    interpolating the curve at both ends and holding it at its last value beyond its own range.

    Every caller passes the SAME [c_lo, c_hi] for all combinations of a setting. Integrating
    each curve over its own realised range is what made 'demand' - which stops as soon as the
    served queries are satisfied - incomparable with the completing orders, while the area was
    still used to break ties.
    """
    if not pts or not (c_hi > c_lo):
        return float("nan")
    xs = np.array([c for c, _v in pts], float)
    vs = np.array([v for _c, v in pts], float)
    # np.interp needs a strictly increasing xp, and duplicate costs do occur: a round whose
    # arrivals all needed no repair (and whose background was held back, as under
    # demand_then_rand) leaves C unchanged while the running mean moves. Keep the LAST value at
    # each distinct cost - the most informed running mean at that cost.
    order = np.argsort(xs, kind="stable")
    xs, vs = xs[order], vs[order]
    last = np.concatenate([xs[1:] != xs[:-1], [True]])
    xs, vs = xs[last], vs[last]
    # The curve starts at the first arrival, not at C = 0 (nothing has been served before that,
    # so there is no quality to plot). np.interp holds the end values outside [xs[0], xs[-1]], so
    # a range extending below xs[0] is filled with the first value; callers pass one common
    # [c_lo, c_hi] for every combination, and runner._replay_v3 sets c_lo to the LATEST first
    # arrival across combinations so no curve is extrapolated backwards further than another.
    grid = np.unique(np.concatenate([[c_lo, c_hi], xs[(xs > c_lo) & (xs < c_hi)]]))
    v = np.interp(grid, xs, vs)  # np.interp holds the end values outside [xs[0], xs[-1]]
    x = grid / N
    return float(np.sum((x[1:] - x[:-1]) * (v[1:] + v[:-1]) / 2))


def summarize(traj, key, M_S, M_T, eps, N):
    C = np.array([c for c, _ in traj], float)
    v = np.array([m[key] for _, m in traj], float)
    hit = np.flatnonzero(v >= M_T - eps)
    c_eps = float(C[hit[0]]) if len(hit) else math.inf
    x = C / N
    auc = float(np.sum((x[1:] - x[:-1]) * (v[1:] + v[:-1]) / 2))
    return {
        "C_eps": c_eps,
        "C_eps_frac": c_eps / N if math.isfinite(c_eps) else math.inf,
        "floor": float(v.min()),
        "final": float(v[-1]),
        "final_C": int(C[-1]),
        "complete": bool(C[-1] == N),
        "auc": auc,
    }
