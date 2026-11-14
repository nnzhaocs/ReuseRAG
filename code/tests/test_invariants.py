"""Run: python3 tests/test_invariants.py   (no pytest needed)"""
import json
import math
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_synthetic import make_pair  # noqa: E402
from reusemap import probe  # noqa: E402
from reusemap.measure import reference_qualities, residual_miss, window_gap  # noqa: E402
from reusemap.pair import Pair  # noqa: E402

CFG = {"K": 10, "kprime": [10, 30, 100, 600], "window": 30, "probe_c": 4, "tgt_depth": 50,
       "adapter_n_fit": 200, "adapter_n_eval": 200, "seed": 0}


def build(tmp, **kw):
    make_pair(tmp, N=600, Q=60, bm25_depth=30, **kw)
    return Pair(tmp, dict(CFG)).precompute()


def test_identity_rho_zero(tmp):
    p = build(tmp, identity=True)
    rho = residual_miss(p, p.kprime_list())
    assert all(abs(v) < 1e-9 for v in rho.values()), rho


def test_full_window_gap_zero(tmp):
    p = build(tmp, src_dims=12, src_noise=0.4)
    ref = reference_qualities(p, "recall")
    G = window_gap(p, [p.N], ref["M_T"], "recall")
    assert abs(G[p.N]) < 1e-12, G


def test_rho_monotone(tmp):
    p = build(tmp, src_dims=12, src_noise=0.4)
    rho = residual_miss(p, p.kprime_list())
    ks = sorted(rho)
    assert all(rho[a] >= rho[b] - 1e-12 for a, b in zip(ks, ks[1:])), rho


def test_exact_miss_formula(tmp):
    """With the 'sample' equal to the whole corpus, the plug-in count is exact."""
    p = build(tmp, src_dims=12, src_noise=0.4)
    allD = np.arange(p.N)
    for qi in p.eval_q[:40]:
        for kp in (10, 30, 100):
            est = probe.estimate_miss(p, qi, kp, allD, "uniform")
            assert est["miss_hat"] == probe.exact_miss(p, qi, kp), (qi, kp)
            assert est["bound_hat"] >= est["miss_hat"]


def test_restricted_topk(tmp):
    p = build(tmp, src_dims=12, src_noise=0.4)
    rng = np.random.default_rng(1)
    for frac in (0.005, 0.05, 0.5, 1.0):
        in_m = rng.random(p.N) < frac
        for qi in p.eval_q[:10]:
            ids, _ = p.target_topk_over(qi, in_m, 10)
            cand = np.flatnonzero(in_m)
            sc = p.doc_tgt[cand] @ p.q_tgt[qi]
            ref = cand[np.lexsort((cand, -sc))][:10]
            assert np.array_equal(ids, ref), (frac, qi)
        batch = p.target_topk_batch(list(p.eval_q[:10]), in_m, 10)
        for qi in p.eval_q[:10]:
            assert np.array_equal(batch[int(qi)][0], p.target_topk_over(qi, in_m, 10)[0]), ("batch", frac, qi)



def test_stratified_exact_at_full_sample(tmp):
    """Stratified estimator is exact when it samples the whole stratum and whole corpus."""
    p = build(tmp, src_dims=12, src_noise=0.4)
    allD = np.arange(p.N)
    for qi in p.eval_q[:30]:
        for kp in (10, 30):
            est = probe.estimate_miss(p, qi, kp, allD, "stratified", mA=10 ** 9)
            assert est["miss_hat"] == probe.exact_miss(p, qi, kp), (qi, kp)


def test_metrics_hand_examples(tmp):
    from reusemap.metrics import ndcg_at_k, recall_at_k
    assert recall_at_k([5, 3, 9], {3: 1, 7: 1}, 3) == 0.5
    assert recall_at_k([5, 3, 9], {3: 2, 7: 1}, 3, thr=2) == 1.0
    assert recall_at_k([5], {3: 0}, 3) is None
    # ideal ranking -> nDCG 1; relevant doc at rank 2 of one -> 1/log2(3)
    assert abs(ndcg_at_k([3, 7], {3: 2, 7: 1}, 10) - 1.0) < 1e-12
    assert abs(ndcg_at_k([9, 3], {3: 1}, 10) - 1 / np.log2(3)) < 1e-12


def test_procrustes_recovers_rotation(tmp):
    from reusemap.compat import compatibility
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((3000, 16)).astype(np.float32)
    R, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    Xt = (Xs @ R.T).astype(np.float32)          # target = rotated source
    _, rep = compatibility(Xs, Xt, "procrustes", 1000, 1000)
    assert rep["residual"] < 1e-4 and rep["kappa"] > 0.9999, rep


def test_winner_rule_and_ties(tmp):
    from reusemap.winner import pick_winner
    inf = float("inf")
    S = {"S0/rand": {"C_eps": 100.0, "floor": 0.50, "auc": 0.6},
         "S2/rand": {"C_eps": 40.0, "floor": 0.40, "auc": 0.7},   # inadmissible (floor)
         "S3/rand": {"C_eps": 60.0, "floor": 0.495, "auc": 0.6},
         "S4/size": {"C_eps": 60.0, "floor": 0.495, "auc": 0.6},  # exact tie with S3/rand
         "S5/demand": {"C_eps": inf, "floor": 0.5, "auc": 0.9}}   # never reaches target
    w, ties = pick_winner(S, M_S=0.5, delta=0.01, N=100, return_ties=True)
    assert w == "S3/rand" and ties == ["S3/rand", "S4/size"], (w, ties)


def test_lopo_majority_and_regret(tmp):
    from reusemap.winner import lopo
    def pair(name, k, r, win):
        summ = {c: {"C_eps": (10.0 if c == win else 50.0), "floor": 1.0, "auc": 0.5}
                for c in ("S2/rand", "S3/rand")}
        return {"name": name, "features": {"kappa": k, "rho": r}, "winner": win,
                "summaries": summ, "M_S": 1.0, "delta": 0.0, "N": 100}
    pairs = [pair("a", 0.9, 0.1, "S3/rand"), pair("b", 0.85, 0.15, "S3/rand"),
             pair("c", 0.1, 0.9, "S2/rand"), pair("d", 0.15, 0.85, "S2/rand")]
    out = lopo(pairs)
    # majority on 3 training pairs always picks the class the held-out pair is NOT in
    assert out["summary"]["majority"]["acc"] == 0.0
    assert abs(out["summary"]["majority"]["mean_regret"] - 0.4) < 1e-12
    assert out["summary"]["1nn[both]"]["acc"] == 1.0
    assert out["summary"]["1nn[both]"]["mean_regret"] == 0.0


def test_heat_gate_detects_signal(tmp):
    """Docs hot only under the dense model lose exposure under the target -> AUC > 0.5;
    with no relation the CI should cover 0.5 in most seeds."""
    from types import SimpleNamespace
    from reusemap.heat import heat_gate
    rng = np.random.default_rng(0)
    N, Q, K = 400, 300, 10
    lexical = rng.random(N) < 0.5
    pool = np.arange(300)            # docs the source retrieves; 300..N only the target surfaces
    src_top = np.stack([rng.choice(pool, K, replace=False) for _ in range(Q)])
    bm25 = np.stack([np.concatenate([s[lexical[s]], rng.choice(np.flatnonzero(~np.isin(np.arange(N), s)),
                                                                  K, replace=False)])[:K] for s in src_top])
    # target keeps lexical docs, swaps each dense-only one for a doc not already in the row
    def swap(s):
        keep = s[lexical[s]]
        fresh = rng.choice(np.arange(300, N), K - len(keep), replace=False)
        return np.concatenate([keep, fresh])
    tgt_top = np.stack([swap(s) for s in src_top])
    assert all(len(np.unique(r)) == K for r in tgt_top)
    p = SimpleNamespace(cfg={"K": K}, src_top=src_top, tgt_top=tgt_top, bm25=bm25, N=N, Q=Q,
                        weights=np.ones(Q) / Q)
    g = heat_gate(p, n_boot=300)
    assert g["gate_open"] and g["ci"][0] > 0.5, {k: g[k] for k in ("auc", "ci", "gate_open")}
    p.tgt_top = np.stack([rng.choice(N, K, replace=False) for _ in range(Q)])
    g2 = heat_gate(p, n_boot=300)
    assert abs(g2["auc"] - 0.5) < 0.1, {k: g2[k] for k in ("auc", "ci", "gate_open")}


def test_replay_semantics(tmp):
    from reusemap.simulate import Replay
    p = build(tmp, src_dims=12, src_noise=0.4)
    rp = Replay(p, [("primary", "recall", 1, p.eval_q)])
    full = np.ones(p.N, bool)
    for st in ("S2", "S3", "S4", "S5"):
        for qi in p.eval_q[:20]:
            assert np.array_equal(rp.rank(st, int(qi), full, True, "async"), p.tgt_top[qi, :10]), st
    cold = np.zeros(p.N, bool)
    for qi in p.eval_q[:20]:   # cold query is served its nomination-order answer
        assert np.array_equal(rp.rank("S2", int(qi), cold, False, "async"), p.src_top[qi, :10])
        assert np.array_equal(rp.rank("S3", int(qi), cold, False, "async"), p.map_top[qi, :10])
    tr = rp.run("S2", "demand", "sync", rounds=10, stream_len=20, seed=0)
    Cs = [c for c, _ in tr]
    assert all(a <= b for a, b in zip(Cs, Cs[1:])), "cost axis must be non-decreasing"
    assert len(rp.sync_encodings) > 0 and min(rp.sync_encodings) >= 0



def test_no_eval_leak_and_repeat_rate(tmp):
    from reusemap.simulate import Replay
    p = build(tmp, src_dims=12, src_noise=0.4)
    assert len(np.intersect1d(p.eval_q, p.traffic_pool)) == 0 and not p.leaky
    rp = Replay(p, [("primary", "recall", 1, p.eval_q)])
    rp.run("S2", "demand", "sync", rounds=10, stream_len=50, seed=0, traffic_w=p.traffic_weights(0.0))
    served = np.concatenate(rp.served)
    assert len(np.intersect1d(served, p.eval_q)) == 0, "eval query served at repeat_rate=0"
    rp.run("S2", "demand", "sync", rounds=10, stream_len=50, seed=0, traffic_w=p.traffic_weights(1.0))
    assert np.isin(np.concatenate(rp.served), p.eval_q).all(), "repeat_rate=1 must serve eval only"
    w = p.traffic_weights(0.3)
    assert abs(w[p.eval_q].sum() - 0.3) < 1e-9 and abs(w.sum() - 1) < 1e-9


def test_leaky_mode_is_explicit(tmp):
    make_pair(tmp, N=600, Q=60, bm25_depth=30, src_dims=12, src_noise=0.4)
    p = Pair(tmp, dict(CFG, eval_frac=0)).precompute()
    assert p.leaky and np.array_equal(np.sort(p.traffic_pool), np.arange(p.Q))


def test_kappa_independent_of_strategy_adapter(tmp):
    make_pair(tmp, N=600, Q=60, bm25_depth=30, identity=True)
    for kind, fit in (("procrustes", "docs"), ("affine", "both"), ("affine", "queries")):
        p = Pair(tmp, dict(CFG, adapter=kind, adapter_fit=fit)).precompute()
        assert abs(p.compat["kappa"] - 1) < 1e-4, (kind, fit, p.compat)
        assert p.adapter_info["fit_on"] == fit


def test_secondary_queries_never_traffic(tmp):
    make_pair(tmp, N=600, Q=60, bm25_depth=30, src_dims=12, src_noise=0.4)
    sec = os.path.join(tmp, "sec.txt")
    np.savetxt(sec, np.arange(5), fmt="%d")
    p = Pair(tmp, dict(CFG, secondary_eval_qids=sec)).precompute()
    assert not np.isin(np.arange(5), p.traffic_pool).any()
    assert not np.isin(np.arange(5), p.eval_q).any(), "secondary set must not enter primary eval"


def test_runner_grid_and_stream_len(tmp):
    from reusemap import runner, smoke
    make_pair(tmp, N=600, Q=60, bm25_depth=30, src_dims=12, src_noise=0.4)
    cfg = dict(CFG, rounds=10, traffic_ratio=[0.5, 2.0], repeat_rate=[0.0, 0.5],
               primary_setting={"traffic_ratio": 2.0, "repeat_rate": 0.0},
               probe_budgets_frac=[0.05], probe_queries=10, orders=["rand", "demand"])
    cfg.pop("kprime")
    cfg["kprime_frac"] = [0.02, 0.05]
    res = runner.run_pair(tmp, cfg, name="grid", log=lambda *_: None)
    assert len(res["settings"]) == 4
    b = int(np.ceil(res["N"] / 10))
    for st in res["settings"]:
        assert st["stream_len"] == max(1, int(round(st["traffic_ratio"] * b / res["window"])))
    prim = [s for s in res["settings"] if s["traffic_ratio"] == 2.0 and s["repeat_rate"] == 0.0][0]
    assert res["winner"] == prim["winner"]
    fails, _ = smoke.check(res)
    assert not fails, fails


def test_mapped_query_detector(tmp):
    """Clean documents, corrupted source queries: target queries mapped into the source
    space must beat the source's own queries, and the runner must say so."""
    from reusemap import runner
    make_pair(tmp, N=600, Q=60, bm25_depth=30, identity=True)
    q = np.load(os.path.join(tmp, "q_src.npy"))
    q = q + 3.0 * np.random.default_rng(0).standard_normal(q.shape).astype(np.float32)
    np.save(os.path.join(tmp, "q_src.npy"), (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32))
    cfg = dict(CFG, rounds=5, probe_budgets_frac=[0.05], probe_queries=5,
               strategies=["S0"], orders=["rand"])
    res = runner.run_pair(tmp, cfg, name="det", log=lambda *_: None)
    assert res["reference"]["M_A"] > res["reference"]["M_S"]
    assert any("SOURCE QUERY ENCODING SUSPECT" in w for w in res["warnings"]), res["warnings"]


# ---------------------------------------------------------------- encoding contracts
class _Tok:
    def __init__(self, side="right", eos_id=2, settable=True):
        self.padding_side, self.eos_token_id, self._settable, self._add = side, eos_id, settable, False

    @property
    def add_eos_token(self):
        return self._add

    @add_eos_token.setter
    def add_eos_token(self, v):
        if self._settable:
            self._add = v

    def __call__(self, text):
        ids = [5, 6]
        return {"input_ids": ids + [self.eos_token_id] if self._add else ids}


class _Pool:
    def __init__(self, mode):
        for k, attr in __import__("reusemap.models", fromlist=["x"]).POOLING_FLAGS.items():
            setattr(self, attr, k == mode)


class _ST(list):
    def __init__(self, mode="cls", side="right", prompts=None, tok=None):
        super().__init__([object(), _Pool(mode)])
        self.tokenizer = tok or _Tok(side)
        self.prompts = prompts or {}
        self.max_seq_length = None


def test_contract_file_valid_and_bge_m3_regression(tmp):
    from reusemap import models
    allc = models.load_contracts()
    assert allc["BAAI/bge-m3"]["query_prefix"] == "", "bge-m3 must not get the BGE English instruction"
    assert allc["BAAI/bge-base-en-v1.5"]["query_prefix"].startswith("Represent this sentence")
    assert allc["intfloat/e5-base-v2"]["doc_prefix"] == "passage: "


def test_contract_refusals_and_overrides(tmp):
    from reusemap import models
    for bad in ("bge-m3", "BAAI/bge-m3-unknown"):   # no fuzzy matching
        try:
            models.get_contract(bad)
            raise AssertionError(f"{bad} should be refused")
        except models.ContractError:
            pass
    # a model whose contract is still blocked in configs/models.json (codesage was unblocked in
    # Phase 2.1 of the 2026-11-07 round, so it can no longer stand in for "blocked")
    blocked = "jinaai/jina-embeddings-v2-base-code"
    try:
        models.get_contract(blocked)
        raise AssertionError("blocked model should be refused")
    except models.ContractError:
        pass
    c = models.get_contract(blocked, allow_blocked=True)
    assert c["status"] == "blocked"
    c = models.get_contract("BAAI/bge-m3", overrides={"query_prefix": "x"})
    assert c["query_prefix"] == "x" and c["overridden"] == ["query_prefix"]


def test_apply_contract(tmp):
    from reusemap import models
    c = models.get_contract("Qwen/Qwen3-Embedding-8B", allow_blocked=True)
    st = _ST(mode="lasttoken", side="right", prompts={"query": "Instruct: ...\nQuery:"})
    f = models.apply_contract(st, c, 512)
    assert st.tokenizer.padding_side == "left" and any("padding_side enforced" in x for x in f)
    assert st.max_seq_length == 512
    for bad_st, msg in ((_ST(mode="mean", prompts={"query": "q"}), "pooling"),
                        (_ST(mode="lasttoken", prompts={}), "prompt")):
        try:
            models.apply_contract(bad_st, c, 512)
            raise AssertionError(msg + " mismatch should raise")
        except models.ContractError:
            pass
    try:
        models.apply_contract(_ST(mode="lasttoken", prompts={"query": "q"}), c, 100000)
        raise AssertionError("cap should raise")
    except models.ContractError:
        pass


def test_eos_enforcement(tmp):
    from reusemap import models
    c = dict(models.get_contract("BAAI/bge-m3"), add_eos=True, pooling=None, padding_side=None)
    st = _ST()
    assert "add_eos enforced" in " ".join(models.apply_contract(st, c, 256))
    st = _ST(tok=_Tok(settable=False))
    try:
        models.apply_contract(st, c, 256)
        raise AssertionError("unenforceable EOS should raise")
    except models.ContractError:
        pass


def test_padding_check_catches_leaky_padding(tmp):
    from reusemap import models
    rng = np.random.default_rng(0)
    base = {}

    def vec(t):
        if t not in base:
            base[t] = rng.standard_normal(16)
        return base[t]

    def clean(texts, bs):
        V = np.stack([vec(t) for t in texts])
        return V / np.linalg.norm(V, axis=1, keepdims=True)

    def leaky(texts, bs):  # pad tokens pulled into the pooled vector, as with a missing mask
        L = max(len(t) for t in texts)
        V = np.stack([vec(t) + 0.15 * (L - len(t)) * np.ones(16) for t in texts])
        return V / np.linalg.norm(V, axis=1, keepdims=True)

    texts = ["q" * n for n in range(1, 40)]
    assert models.padding_check(clean, texts)["status"] == "ok"
    assert models.padding_check(leaky, texts)["status"] == "fail"


def test_guards(tmp):
    from reusemap.winner import pick_winner
    make_pair(tmp, N=600, Q=60, bm25_depth=30, src_dims=12, src_noise=0.4)
    p = Pair(tmp, dict(CFG)).precompute()
    for bad in (-0.1, 1.5):
        try:
            p.traffic_weights(bad)
            raise AssertionError("repeat_rate out of range must raise")
        except ValueError:
            pass
    assert abs(p.traffic_weights(1.0)[p.eval_q].sum() - 1) < 1e-12
    sec = os.path.join(tmp, "sec_bad.txt")
    np.savetxt(sec, [0, 10_000], fmt="%d")
    try:
        Pair(tmp, dict(CFG, secondary_eval_qids=sec))
        raise AssertionError("out-of-range secondary ids must raise")
    except ValueError:
        pass
    S = {"S2/rand": {"C_eps": 10.0, "floor": float("nan"), "auc": 0.9},
         "S0/rand": {"C_eps": 50.0, "floor": 0.6, "auc": 0.5}}
    assert pick_winner(S, M_S=0.6, delta=0.01, N=100) == "S0/rand", "nan floor must be inadmissible"


def test_loud_failures(tmp):
    make_pair(tmp, N=600, Q=60, bm25_depth=30, src_dims=12, src_noise=0.4)
    try:
        Pair(tmp, dict(CFG, secondary_eval_qids=os.path.join(tmp, "missing.txt")))
        raise AssertionError("missing secondary file must raise")
    except FileNotFoundError:
        pass
    np.save(os.path.join(tmp, "weights.npy"), np.ones(60))
    try:
        Pair(tmp, dict(CFG, workload="uniform"))
        raise AssertionError("weights.npy ignored by workload=uniform must raise")
    except ValueError:
        pass
    p = Pair(tmp, dict(CFG, workload="file"))
    assert abs(p.weights.sum() - 1) < 1e-12


def test_more_guards(tmp):
    make_pair(tmp, N=600, Q=60, bm25_depth=30, src_dims=12, src_noise=0.4)
    p = Pair(tmp, dict(CFG))
    w = np.ones(60)
    w[p.eval_q] = 0
    np.save(os.path.join(tmp, "weights.npy"), w)
    try:
        Pair(tmp, dict(CFG, workload="file"))
        raise AssertionError("all-zero eval weights must raise")
    except ValueError:
        pass
    os.remove(os.path.join(tmp, "weights.npy"))
    try:
        Pair(tmp, dict(CFG, tgt_depth=5)).precompute()
        raise AssertionError("tgt_depth < K must raise")
    except ValueError:
        pass
    p = Pair(tmp, dict(CFG)).precompute()
    out = probe.evaluate_probe(p, 30, [0.05], n_queries=10_000)
    assert out["n_queries"] < out["n_queries_requested"] == 10_000
    p.weights = np.zeros_like(p.weights)
    p.weights[p.eval_q] = 1.0                    # traffic (probe) queries all weight 0
    try:
        probe.evaluate_probe(p, 30, [0.05], n_queries=10)
        raise AssertionError("zero-weight probe queries must raise")
    except ValueError:
        pass
    full, dev = Pair(tmp, dict(CFG)), Pair(tmp, dict(CFG, n_eval_queries=5))
    assert np.array_equal(full.traffic_pool, dev.traffic_pool), "dev runs must keep the same traffic"
    assert len(dev.eval_q) == 5 and np.isin(dev.eval_q, full.eval_q).all()


def test_measure_only_calibration_mode(tmp):
    from reusemap import runner, smoke
    make_pair(tmp, N=600, Q=60, bm25_depth=30, src_dims=12, src_noise=0.4)
    cfg = dict(CFG, replay=False, eval_frac=1.0)
    res = runner.run_pair(tmp, cfg, name="cal", log=lambda *_: None)
    assert res["measure_only"] and res["winner"] is None and res["settings"] == []
    judged = len({q for q in Pair(tmp, cfg).qrels})
    assert res["n_eval"] == judged, "calibration must evaluate every judged query"
    assert set(res["G"]) and not smoke.check(res)[0]

# ----------------------------------------------------------------------------------
# Replay v3 (first-arrival measurement). See reusemap/simulate.py:run_arrivals.
# ----------------------------------------------------------------------------------
V3 = dict(CFG, rounds=10, max_rounds_factor=40, eval_arrival="first_service",
          objective="debt", tau={"debt": 0.0, "c_serve_frac": 0.0},
          traffic_ratio=[4.0], primary_setting={"traffic_ratio": 4.0},
          orders=["rand", "size", "demand", "demand_then_rand"],
          probe_budgets_frac=[0.05], probe_queries=10, min_eval_served=0.5)


def _v3_run(tmp, **kw):
    from reusemap import runner
    make_pair(tmp, N=600, Q=120, bm25_depth=30, src_dims=12, src_noise=0.4)
    return runner.run_pair(tmp, dict(V3, **kw), name="v3", log=lambda *_: None)


def test_v3_invariants_hold(tmp):
    from reusemap import smoke
    res = _v3_run(tmp)
    assert res["replay_version"] == 3 and res["objective"] == "debt"
    fails, _notes = smoke.check(res)
    assert not fails, fails
    s = res["settings"][0]
    assert s["arrivals_identical"] and s["n_arrival_prefix_checked"] > 0
    for combo, su in s["summaries"].items():
        assert su["unpaid_scored"] == 0, combo
        assert su["repairs_all_total"] <= su["C_final"], combo
        assert abs(su["debt"] - (su["target_mean"] - su["served_mean"])) < 1e-12, combo
        if combo.endswith("/demand"):
            assert su["repairs_all_total"] == su["C_final"], f"{combo} did background work"
        else:
            assert su["completed"] and su["C_final"] == res["N"], combo
    # S0 serves the replaced system's answer to every arrival that precedes completion
    s0 = s["summaries"]["S0/rand"]
    assert s0["n_pre_complete"] <= s0["n_arrived"] and s0["repairs_all_total"] == 0


def test_v3_arrival_sequence_is_strategy_independent(tmp):
    """Every combination must be compared on the same queries in the same order. Sharing one
    generator between the background order and the stream broke this: 'demand' drew no
    permutation first, so it got a different traffic stream than 'rand'."""
    from reusemap.pair import Pair
    from reusemap.simulate import Replay
    make_pair(tmp, N=600, Q=120, bm25_depth=30, src_dims=12, src_noise=0.4)
    p = Pair(tmp, dict(CFG)).precompute()
    rp = Replay(p, [("primary", "recall", 1, p.eval_q)])
    w = p.all_query_weights()
    seqs = {}
    for st, od in (("S2", "rand"), ("S2", "demand"), ("S4", "size"), ("S0", "rand")):
        raw = rp.run_arrivals(st, od, rounds=10, stream_len=8, seed=0, max_rounds=200, arrival_w=w)
        seqs[(st, od)] = [a[0] for a in raw["arrivals"]]
    n = min(len(v) for v in seqs.values())
    first = next(iter(seqs.values()))[:n]
    assert all(v[:n] == first for v in seqs.values()), seqs
    assert n > 0 and len(set(first)) == len(first), "a query must arrive at most once"
    assert not np.isin(p.secondary_q, first).any() if len(p.secondary_q) else True


def test_v3_c_serve_is_N_when_the_target_is_not_reached(tmp):
    from reusemap.simulate import summarize_arrivals
    raw = {"acc": {"primary": [0.5, 1.0]}, "tgt_acc": {"primary": [0.9, 1.0]},
           "src_acc": {"primary": [0.4, 1.0]}, "arrivals": [(0, 10, 3)], "traj": [(10, {"primary": 0.5})],
           "per_query_cost": [], "completed": True, "C_final": 100, "n_expected": 1, "n_arrived": 1,
           "last_arrival_C": 10, "rounds_used": 1, "max_rounds": 10}
    s = summarize_arrivals(raw, "primary", 0.4, 0.9, 0.01, 100)
    assert not s["reaches_target"] and s["C_serve"] == 100 and s["C_serve_frac"] == 1.0
    assert abs(s["debt"] - 0.4) < 1e-12 and abs(s["source_mean"] - 0.4) < 1e-12
    s2 = summarize_arrivals(raw, "primary", 0.4, 0.9, 0.5, 100)  # eps wide enough to reach
    assert s2["reaches_target"] and s2["C_serve"] == 10


def test_v3_tau_band_picks_the_simplest(tmp):
    from reusemap.winner import pick_winner_v3
    base = {"completed": True, "served_mean": 0.8, "source_mean": 0.7, "C_serve_frac": 1.0}
    sums = {"S5/signal": dict(base, debt=0.0100), "S2/rand": dict(base, debt=0.0120),
            "S0/rand": dict(base, debt=0.0400)}
    assert pick_winner_v3(sums, 0.7, 0.01, "debt", band=0.0) == "S5/signal"
    w, band = pick_winner_v3(sums, 0.7, 0.01, "debt", band=0.005, return_band=True)
    assert w == "S2/rand" and set(band) == {"S2/rand", "S5/signal"}, (w, band)
    # not completed, or served worse than the replaced system, is inadmissible
    sums["S2/rand"]["completed"] = False
    sums["S5/signal"]["served_mean"] = 0.5
    assert pick_winner_v3(sums, 0.7, 0.01, "debt", band=0.005) == "S0/rand"


def test_v3_auc_uses_one_common_range(tmp):
    from reusemap.simulate import auc_over
    short = [(0, 0.0), (50, 1.0)]            # stops at C = 50
    long = [(0, 0.0), (50, 1.0), (100, 1.0)]  # same curve, carries on
    assert abs(auc_over(short, 100, 50) - auc_over(long, 100, 50)) < 1e-12
    # held at its last value beyond its own range, so the areas match over [0, 100] too
    assert abs(auc_over(short, 100, 100) - auc_over(long, 100, 100)) < 1e-12
    # a flat curve of height h over [lo, hi] has area h * (hi - lo) / N
    assert abs(auc_over([(0, 0.5), (100, 0.5)], 100, 100, 50) - 0.5 * 0.5) < 1e-12
    assert math.isnan(auc_over([], 100, 100))


def test_expected_arrival_rounds_with_eval_weights_that_sum_below_one(tmp):
    """The runner passes per-draw probabilities of the EVALUATION queries only (1/6,980 each for
    3,490 of them), which sum to 0.5, not 1. The weighted fraction is normalised by their own sum,
    so the answer must equal the closed form 1 - (1 - w)^(T*s) >= target for equal weights."""
    import math
    from reusemap.simulate import expected_arrival_rounds
    w = np.full(8, 1.0 / 16)            # 8 evaluation queries out of 16 servable, uniform
    s, target = 4, 0.99
    T = expected_arrival_rounds(w, np.arange(8), s, target, weighted=True)
    want = math.ceil(math.log(1 - target) / (s * math.log(1 - 1.0 / 16)))
    if T != want:
        raise AssertionError(f"expected {want} rounds, got {T}")


def test_expected_arrival_rounds_is_coupon_collector(tmp):
    from reusemap.simulate import expected_arrival_rounds
    n, L = 100, 5
    w = np.full(n, 1.0 / n)
    q = np.arange(n)
    T = expected_arrival_rounds(w, q, L, 0.99)
    frac = lambda t: float(np.mean(1 - (1 - 1.0 / n) ** (t * L)))
    assert frac(T) >= 0.99 > frac(T - 1), (T, frac(T), frac(T - 1))
    # NOT n / stream_len: with replacement it takes far longer than 20 rounds
    assert T > n / L * 3
    assert expected_arrival_rounds(w, q, 2 * L, 0.99) < T
    assert expected_arrival_rounds(np.array([1.0, 0.0]), [1], 1, 0.99) == math.inf


def test_provenance_dirty_is_part_of_the_run_key(tmp):
    from reusemap import provenance as pv
    clean, dirty = "abc123", "abc123-dirty:deadbeef"
    k1 = pv.run_key(clean, "cfg", "data")
    k2 = pv.run_key(dirty, "cfg", "data")
    k3 = pv.run_key("abc123-dirty:0bad0bad", "cfg", "data")
    assert k1 != k2 != k3 and k1 != k3, "a dirty scratch run must not share a key with the clean run"
    assert pv.is_dirty(dirty) and not pv.is_dirty(clean)
    assert pv.run_status([], clean) == "ok" and pv.run_status([], dirty) == "ok_dirty"
    assert pv.run_status(["I4 failed"], clean) == "invariant_fail"
    reg = os.path.join(tmp, "RUNS.tsv")
    pv.register(reg, "P2", k2, dirty, "cfg", "data", pv.run_status([], dirty), "out.json")
    assert pv.lookup(reg, k2) is None, "an ok_dirty row must not suppress the clean rerun"


def test_self_retrieval_check_catches_a_collapsed_space(tmp):
    from reusemap.models import self_retrieval_check
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((200, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    texts = [f"doc {i}" for i in range(200)]
    idx, pool = np.arange(0, 60), np.arange(200)
    good = self_retrieval_check(emb, lambda t, b: emb[[int(s.split()[1]) for s in t]], texts, idx, pool)
    assert good["status"] == "ok" and good["top1_rate"] == 1.0
    one = np.repeat(emb[:1], 60, axis=0)  # every query lands on the same point
    bad = self_retrieval_check(emb, lambda t, b: one[:len(t)], texts, idx, pool)
    assert bad["status"] == "fail" and bad["top1_rate"] < 0.1, bad


def test_scripts_refuse_misaligned_encodings(tmp):
    """The e5-base failure: 6,980 of 7,077 judged queries had no vector, and nothing said so."""
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    data, enc = os.path.join(tmp, "data"), os.path.join(tmp, "enc")
    os.makedirs(data), os.makedirs(enc)
    rng = np.random.default_rng(0)
    with open(os.path.join(data, "corpus.tsv"), "w") as f:
        for i in range(50):
            f.write(f"{i}\ttext {i}\n")
    with open(os.path.join(data, "queries.tsv"), "w") as f:
        for i in range(20):
            f.write(f"{i}\tquery {i}\n")
    with open(os.path.join(data, "qrels.tsv"), "w") as f:
        for i in range(20):
            f.write(f"{i}\t{i}\t1\n")
    np.save(os.path.join(enc, "doc.npy"), rng.standard_normal((50, 8)).astype(np.float32))
    np.save(os.path.join(enc, "q.npy"), rng.standard_normal((17, 8)).astype(np.float32))  # 3 short
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "encode_check.py"),
                        "--enc", enc, "--data", data], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout
    assert "row count matches" in r.stdout and "has a vector" in r.stdout
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "model_quality.py"),
                        "--qrels", os.path.join(data, "qrels.tsv"), "--model", f"m={enc}"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "no vector" in (r.stdout + r.stderr), (r.stdout, r.stderr)
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "assemble_pair.py"),
                        "--src", enc, "--tgt", enc, "--data", data,
                        "--out", os.path.join(tmp, "pair")], capture_output=True, text=True)
    assert r.returncode != 0 and "mismatched inputs" in (r.stdout + r.stderr), (r.stdout, r.stderr)


def test_arrival_weights_are_recorded_not_reconstructed(tmp):
    """predict.py cannot guess which queries are the held-out evaluation set: it is a seeded
    permutation of the judged queries, not the first n_eval rows. Under a uniform workload any
    choice gives the same answer, which is why guessing looked fine; under a weighted workload
    every arrival prediction would come from the wrong queries' weights."""
    from reusemap import runner
    from reusemap.pair import Pair
    make_pair(tmp, N=600, Q=120, bm25_depth=30, src_dims=12, src_noise=0.4)
    # a weighted workload where the first n_eval rows are NOT the evaluation set
    rng = np.random.default_rng(3)
    np.save(os.path.join(tmp, "weights.npy"), rng.gamma(0.5, 1.0, 120) + 1e-3)
    cfg = dict(CFG, workload="file", replay=False, eval_frac=0.5)
    res = runner.run_pair(tmp, cfg, name="w", log=lambda *_: None)
    arr = res["arrival"]
    p = Pair(tmp, cfg)
    assert arr["eval_q"] == sorted(int(q) for q in p.eval_q), "must record the real eval split"
    assert arr["eval_q"] != list(range(len(arr["eval_q"]))), "fixture must not be the first rows"
    w = p.all_query_weights()
    assert np.allclose(arr["eval_weights"], [w[q] for q in arr["eval_q"]])
    assert arr["n_servable"] == int((w > 0).sum()) and arr["workload"] == "file"
    # the recorded weights change the arrival estimate, so guessing was not harmless
    from reusemap.simulate import expected_arrival_rounds
    right = expected_arrival_rounds(np.asarray(arr["eval_weights"]),
                                    np.arange(len(arr["eval_weights"])), 4, 0.9)
    wrong = expected_arrival_rounds(w[:len(arr["eval_q"])] / w.sum(),
                                    np.arange(len(arr["eval_q"])), 4, 0.9)
    assert right != wrong, (right, wrong)


def test_predict_refuses_to_guess_a_weighted_arrival_distribution(tmp):
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    res = {"name": "P", "N": 1000, "window": 50, "n_eval": 10, "n_traffic_pool": 10,
           "reference": {"M_S": 0.5, "M_T": 0.7, "M_A": 0.4},
           "G": {"50": 0.0}, "rho": {"50": 0.1}, "config": {"workload": "file"}}
    cov = {"name": "src", "N": 1000, "rows": [{"kprime": 50, "coverage": 0.8}]}
    cfg = {"K": 10, "window": 50, "eps": 0.01, "rounds": 10, "traffic_ratio": [4.0],
           "workload": "file"}
    paths = {}
    for n, o in (("res.json", res), ("cov.json", cov), ("cfg.json", cfg)):
        paths[n] = os.path.join(tmp, n)
        json.dump(o, open(paths[n], "w"))
    cmd = [sys.executable, os.path.join(root, "scripts", "predict.py"),
           "--result", paths["res.json"], "--coverage", paths["cov.json"], "--config", paths["cfg.json"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0 and "refusing to guess" in (r.stdout + r.stderr), (r.stdout, r.stderr)
    # with the recorded block it runs, and uses those weights
    res["arrival"] = {"eval_q": [7, 9], "eval_weights": [0.3, 0.1], "n_servable": 5,
                      "workload": "file", "n_eval_zero_weight": 0}
    json.dump(res, open(paths["res.json"], "w"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and "exact per-evaluation-query" in r.stdout, (r.stdout, r.stderr)
    # a uniform workload still runs without the block (the old results stay usable)
    res.pop("arrival"), res["config"].update(workload="uniform"), cfg.update(workload="uniform")
    json.dump(res, open(paths["res.json"], "w")), json.dump(cfg, open(paths["cfg.json"], "w"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and "uniform weights assumed" in r.stdout, (r.stdout, r.stderr)


def test_served_fraction_is_weighted_not_counted(tmp):
    """Every quality number is workload-weighted, so the served-population gate must be too.
    Under skew the two differ by orders of magnitude and a count-based gate rejects every
    skewed workload (ORCAS, the Zipf code sweep) even when the measurement is representative."""
    from reusemap import runner, smoke
    from reusemap.simulate import expected_arrival_rounds
    make_pair(tmp, N=600, Q=120, bm25_depth=30, src_dims=12, src_noise=0.4)
    # A deterministic workload instead of a draw from gamma(0.3): every even-indexed query
    # carries weight 1 and every odd-indexed one 1e-9, so whichever half the seeded split holds
    # out, about half of the held-out queries are too light to be drawn inside max_rounds. No
    # part of the assertion then depends on the RNG. (Relaxing it to >= would be the wrong fix:
    # it would also pass if the weighted fraction were computed as the count.)
    w = np.where(np.arange(120) % 2 == 0, 1.0, 1e-9)
    np.save(os.path.join(tmp, "weights.npy"), w)
    cfg = dict(V3, workload="file", rounds=10, max_rounds_factor=10, traffic_ratio=[16.0],
               primary_setting={"traffic_ratio": 16.0}, min_eval_served=0.9,
               orders=["rand", "demand"], strategies=["S0", "S2"])
    res = runner.run_pair(tmp, cfg, name="wt", log=lambda *_: None)
    assert not smoke.check(res)[0], smoke.check(res)[0]
    st = res["settings"][0]
    s = st["summaries"]["S2/rand"]
    # Exact, not a margin: the heavy queries are the ones the stream can reach, so the count
    # share must equal the heavy share of THIS pair's held-out set and the weighted share must
    # be ~1. Deriving it from the actual split removes the last dependence on which half the
    # seeded permutation held out.
    from reusemap.pair import Pair
    ev = Pair(tmp, cfg).eval_q
    heavy = sum(1 for q in ev if w[int(q)] > 0.5)
    assert 0 < heavy < len(ev), "the fixture must hold out a mix of heavy and light queries"
    # The count share is the heavy share of this pair's held-out set, give or take a light query
    # that happened to be drawn. A light query has probability 1e-9/60 per draw and there are at
    # most max_rounds * stream_len = 3200 draws, so P(any of the ~30 light ones arrives) ~ 2e-6 -
    # but the slack is here rather than in a hand-picked threshold, so raising the light weight
    # or the round count cannot start failing this.
    lo = heavy / len(ev)
    assert lo <= s["eval_served_frac"] <= lo + 2 / len(ev), \
        (s["eval_served_frac"], lo, heavy, len(ev))
    assert s["eval_served_weight_frac"] > 0.999, s["eval_served_weight_frac"]
    assert st["eval_served_weight_frac"] >= cfg["min_eval_served"] and st["winner"] is not None, \
        "a representative weighted sample must not be rejected for its query count"
    # and the estimator agrees on the direction
    a = res["arrival"]
    w, q = np.asarray(a["eval_weights"]), np.arange(len(a["eval_weights"]))
    assert (expected_arrival_rounds(w, q, 8, 0.99, weighted=True)
            < expected_arrival_rounds(w, q, 8, 0.99)), "weighted must be reached sooner under skew"


def test_measure_only_matches_the_replay_it_feeds(tmp):
    """Phase 6.5 joins G(K') from a measure-only pass with coverage(K') and predicts the replay.
    The two passes must therefore use the SAME config: borrowing one that merely sets
    `replay: false` (nq_100k_v2.json) would measure a different window ladder, a different
    primary metric, and - through eval_frac - a different held-out split, so the recorded
    arrival block would describe the wrong query population."""
    from reusemap import runner
    make_pair(tmp, N=600, Q=120, bm25_depth=30, src_dims=12, src_noise=0.4)
    full = runner.run_pair(tmp, dict(V3), name="full", log=lambda *_: None)
    meas = runner.run_pair(tmp, dict(V3, replay=False), name="meas", log=lambda *_: None)
    assert meas["measure_only"] and not full.get("measure_only")
    assert meas["window"] == full["window"], "same window depth"
    assert set(meas["G"]) == set(full["G"]), "same K' ladder, or predict.py drops depths"
    assert meas["n_eval"] == full["n_eval"]
    assert meas["arrival"]["eval_q"] == full["arrival"]["eval_q"], \
        "the recorded arrival population must be the one the replay evaluates"
    assert meas["reference"] == full["reference"] and meas["features"] == full["features"]
    # the borrowed-config failure mode this guards against
    other = runner.run_pair(tmp, dict(V3, replay=False, eval_frac=1.0), name="other",
                            log=lambda *_: None)
    assert other["arrival"]["eval_q"] != full["arrival"]["eval_q"]


def test_band_tiebreak_cost_vs_simplicity(tmp):
    """Inside the equivalence band, 'cost' prefers the cheaper combination and 'simplicity' the
    simpler one. On the 1M run the band held every window strategy, so simplicity picked S2/rand
    (C_serve = 1.000) over S2/demand_then_rand (0.879) and discarded the whole measured saving.
    'simplicity' stays the default so an old result re-runs to its own winner from its config."""
    from reusemap.winner import BAND_TIEBREAKS, pick_winner_v3
    base = {"completed": True, "served_mean": 0.87, "source_mean": 0.83}
    sums = {"S2/rand": dict(base, debt=-0.0003, C_serve_frac=1.0),
            "S2/demand_then_rand": dict(base, debt=0.0000, C_serve_frac=0.879),
            "S5/signal": dict(base, debt=0.0003, C_serve_frac=1.0),
            "S0/rand": dict(base, debt=0.0410, C_serve_frac=1.0)}
    w_s, band = pick_winner_v3(sums, 0.83, 0.01, "debt", band=0.005, return_band=True)
    assert w_s == "S2/rand" and set(band) == {"S2/rand", "S2/demand_then_rand", "S5/signal"}
    w_c = pick_winner_v3(sums, 0.83, 0.01, "debt", band=0.005, band_tiebreak="cost")
    assert w_c == "S2/demand_then_rand", w_c
    # equal cost inside the band falls back to simplicity
    sums["S2/demand_then_rand"]["C_serve_frac"] = 1.0
    assert pick_winner_v3(sums, 0.83, 0.01, "debt", band=0.005, band_tiebreak="cost") == "S2/rand"
    # an inadmissible cheaper option is still excluded
    sums["S2/demand_then_rand"].update(C_serve_frac=0.5, completed=False)
    assert pick_winner_v3(sums, 0.83, 0.01, "debt", band=0.005, band_tiebreak="cost") == "S2/rand"
    try:
        pick_winner_v3(sums, 0.83, 0.01, "debt", band=0.005, band_tiebreak="cheapest")
        raise AssertionError("an unknown band_tiebreak must raise")
    except ValueError:
        pass
    assert BAND_TIEBREAKS == ("simplicity", "cost")


def test_sweep_matches_measure_and_coverage(tmp):
    """The sweep is the gate for every later decision, so its three quantities must equal the
    direct computations: residual_miss, window_gap, and the union of the servable queries'
    windows (the set the replay actually pays for)."""
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    make_pair(tmp, N=600, Q=120, bm25_depth=30, src_dims=12, src_noise=0.4)
    cfg = dict(CFG, window=30, eps=0.01, eps_sensitivity=[0.01], eval_frac=0.5,
               workload="uniform", primary={"metric": "recall", "thr": 1}, secondary=[])
    cfg.pop("kprime", None)
    cpath = os.path.join(tmp, "cfg.json")
    json.dump(cfg, open(cpath, "w"))
    out = os.path.join(tmp, "sw")
    ladder = [5, 15, 30, 60]
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "sweep.py"), tmp,
                        "--config", cpath, "--kprime", ",".join(map(str, ladder)),
                        "--out-dir", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.load(open(os.path.join(out, "frontier.json")))
    assert got["pilot"] is False and len(got["per_pair"]) == 1
    depths = {d["kprime"]: d for d in got["per_pair"][0]["depths"]}

    from reusemap.measure import reference_qualities, residual_miss, window_gap
    from reusemap.pair import Pair
    p = Pair(tmp, dict(cfg, kprime=ladder)).precompute()
    ref = reference_qualities(p, "recall", 1)
    rho = residual_miss(p, ladder, "src")
    G = window_gap(p, ladder, ref["M_T"], "recall", 1, "src")
    w = p.all_query_weights()
    rows = np.flatnonzero(np.asarray(w, float) > 0)
    for k in ladder:
        assert abs(depths[k]["rho"] - rho[k]) < 1e-12, k
        assert abs(depths[k]["G"] - G[k]) < 1e-12, k
        direct = len(np.unique(p.src_top[np.ix_(rows, np.arange(min(k, p.src_top.shape[1])))]))
        assert depths[k]["covered"] == direct, (k, depths[k]["covered"], direct)
        assert abs(depths[k]["ceiling_frac"] - (1 - direct / p.N)) < 1e-12, k
        assert depths[k]["admissible"]["0.01"] == (G[k] <= 0.01), k
    # coverage is non-decreasing in K', so the ceiling is non-increasing
    cov = [depths[k]["coverage"] for k in ladder]
    assert all(a <= b + 1e-12 for a, b in zip(cov, cov[1:])), cov
    # and the frontier is the smallest admissible depth
    adm = [k for k in ladder if G[k] <= 0.01]
    want = min(adm) if adm else None
    assert got["frontier"][0]["kstar"] == want, (got["frontier"][0], want)
    # without --mapped, rho_mapped is absent
    assert all(d["rho_mapped"] is None for d in depths.values())

    # --mapped adds rho over the adapter-mapped window (S3's nominator). It needs no extra
    # retrieval: Pair.precompute builds map_top unconditionally, so the flag only adds the rho.
    out2 = os.path.join(tmp, "sw_mapped")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "sweep.py"), tmp,
                        "--config", cpath, "--kprime", ",".join(map(str, ladder)),
                        "--mapped", "--out-dir", out2], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    dm = {d["kprime"]: d for d in json.load(open(os.path.join(out2, "frontier.json")))["per_pair"][0]["depths"]}
    rho_m = residual_miss(p, ladder, "map")
    for k in ladder:
        assert abs(dm[k]["rho_mapped"] - rho_m[k]) < 1e-12, k
        assert abs(dm[k]["rho"] - rho[k]) < 1e-12, k          # unchanged by the flag
        assert dm[k]["covered"] == depths[k]["covered"], k    # ditto

    # --pilot leaves coverage EXACT (the servable set is {q: weight > 0}, independent of
    # n_eval_queries); only rho and G are subsampled. The pilot warning says so, so it is tested.
    out3 = os.path.join(tmp, "sw_pilot")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "sweep.py"), tmp,
                        "--config", cpath, "--kprime", ",".join(map(str, ladder)),
                        "--pilot", "5", "--out-dir", out3], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    pil = json.load(open(os.path.join(out3, "frontier.json")))["per_pair"][0]
    dp = {d["kprime"]: d for d in pil["depths"]}
    assert pil["pilot"] is True and pil["n_eval"] == 5
    for k in ladder:
        assert dp[k]["covered"] == depths[k]["covered"], ("coverage must be exact under pilot", k)
    e = pil["pilot_extrapolation"]
    assert e["full_n_eval"] == got["per_pair"][0]["n_eval"] > 5, e   # not a hard-coded 3490
    assert e["q_scale"] > 1.0 and e["cov_scale"] == 1.0, e
    assert e["per_pair_seconds"] >= e["precompute_seconds"], e


def test_auc_ignores_duplicate_costs(tmp):
    """A round whose arrivals needed no repair leaves C unchanged while the running mean moves,
    so the trajectory can hold two points at one cost. np.interp needs a strictly increasing xp."""
    from reusemap.simulate import auc_over
    dup = [(0, 0.0), (50, 0.4), (50, 0.6), (100, 0.6)]
    clean = [(0, 0.0), (50, 0.6), (100, 0.6)]  # the later value is the informed one
    assert abs(auc_over(dup, 100, 100) - auc_over(clean, 100, 100)) < 1e-12
    assert math.isfinite(auc_over([(10, 0.5), (10, 0.5)], 100, 100, 0.0))


def test_coverage_script_matches_a_direct_union(tmp):
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    make_pair(tmp, N=400, Q=40, bm25_depth=20, src_dims=12, src_noise=0.4)
    out = os.path.join(tmp, "cov.json")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "coverage.py"),
                        "--docs", os.path.join(tmp, "doc_src.npy"),
                        "--queries", os.path.join(tmp, "q_src.npy"),
                        "--kprime", "5,20,100", "--out", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = {x["kprime"]: x for x in json.load(open(out))["rows"]}
    from reusemap.search import normalize_rows, topk_ip
    X = normalize_rows(np.load(os.path.join(tmp, "doc_src.npy")))
    Q = normalize_rows(np.load(os.path.join(tmp, "q_src.npy")))
    top, _ = topk_ip(Q, X, 100)
    for k in (5, 20, 100):
        assert rows[k]["covered"] == len(np.unique(top[:, :k])), k
        assert abs(rows[k]["coverage"] - rows[k]["covered"] / 400) < 1e-12
    assert rows[5]["coverage"] <= rows[20]["coverage"] <= rows[100]["coverage"]
    assert rows[5]["ceiling_frac"] >= rows[100]["ceiling_frac"]

    # --weights applies the same zero-weight exclusion as sweep.py's servable set. This is the
    # only point where the two scripts' query-set definitions are made to agree, so it is tested:
    # without it the RUNBOOK's "a disagreement is a bug" is a false alarm on a weighted workload.
    wts = np.ones(40)
    wts[1::2] = 0.0                                   # drop every odd query
    wpath = os.path.join(tmp, "w.npy")
    np.save(wpath, wts)
    out2 = os.path.join(tmp, "cov_w.json")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "coverage.py"),
                        "--docs", os.path.join(tmp, "doc_src.npy"),
                        "--queries", os.path.join(tmp, "q_src.npy"),
                        "--kprime", "5,20", "--weights", wpath, "--out", out2],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "excluded 20 of 40 queries with weight 0" in r.stdout, r.stdout
    gotw = json.load(open(out2))
    assert gotw["n_queries"] == 20 and gotw["weights_file"] == os.path.abspath(wpath)
    kept = np.arange(0, 40, 2)
    for k in (5, 20):
        direct = len(np.unique(top[np.ix_(kept, np.arange(k))]))
        row = next(x for x in gotw["rows"] if x["kprime"] == k)
        assert row["covered"] == direct, (k, row["covered"], direct)
        # and dropping half the queries can only shrink the union
        assert row["covered"] <= rows[k]["covered"]


def test_c_serve_basis_names_which_of_the_three_situations_produced_it(tmp):
    """C_serve is three different quantities and the column is unreadable without knowing which.
    A strategy that repaired windows reports work it did; one that reached the target having
    repaired nothing reports how far its background got (so the number falls as traffic_ratio
    rises while the strategy is unchanged); one that never reached the target is charged N."""
    from reusemap.simulate import summarize_arrivals
    N = 1000

    def mk(served, target, reps_all, last_C):
        raw = {"acc": {"primary": [served, 1.0]}, "tgt_acc": {"primary": [target, 1.0]},
               "src_acc": {"primary": [0.80, 1.0]}, "arrivals": [], "traj": [],
               "last_arrival_C": last_C, "completed": True, "C_final": N,
               "repairs_all_total": reps_all, "n_expected": 1, "n_arrived": 1,
               "expected_weight": 1.0, "arrived_weight": 1.0, "rounds_used": 3,
               "per_query_cost": []}
        return summarize_arrivals(raw, "primary", 0.80, 0.87, 0.01, N)

    repaired = mk(0.87, 0.87, 264, 300)
    assert repaired["C_serve_basis"] == "at_last_arrival"
    assert repaired["C_serve_frac"] == 0.3

    # S0 on a pair whose gap is already inside eps: reaches the target with zero repairs
    no_gap = mk(0.869, 0.87, 0, 30)
    assert no_gap["C_serve_basis"] == "no_repair_needed", no_gap["C_serve_basis"]
    assert no_gap["C_serve_frac"] == 0.03
    # ... and the same strategy at a higher traffic ratio reports a smaller number for the same
    # behaviour, which is exactly why the basis has to travel with the column
    assert mk(0.869, 0.87, 0, 0)["C_serve_frac"] == 0.0

    missed = mk(0.80, 0.87, 0, 30)
    assert missed["C_serve_basis"] == "never_reached_target"
    assert missed["C_serve_frac"] == 1.0, "a miss is charged N by convention"

    # C_total is what it ever encodes; every completing strategy encodes exactly N, which is why
    # cost cannot separate them and debt is the objective (DESIGN.md scope fence, Thaw's C2)
    for s in (repaired, no_gap, missed):
        assert s["C_total_frac"] == 1.0, s["C_total_frac"]


def test_cost_tiebreak_reports_when_it_ranks_across_bases(tmp):
    """band_tiebreak='cost' orders the band by C_serve_frac, which is only a cost when every
    member got there the same way. The clash is reported, not silently ranked: this is the P14
    case, where S0's 0.0300 beat the window's 0.2642 though both encode N."""
    from reusemap.winner import cost_basis_conflict, pick_winner_v3, tied_with
    base = {"completed": True, "served_mean": 0.871, "source_mean": 0.8709}
    sums = {
        "S0/rand": dict(base, debt=0.0043, C_serve_frac=0.0300, C_serve_basis="no_repair_needed"),
        "S2/demand_then_rand": dict(base, debt=-0.0004, C_serve_frac=0.2642,
                                    C_serve_basis="at_last_arrival"),
        "S2/rand": dict(base, debt=-0.0004, C_serve_frac=0.2943, C_serve_basis="at_last_arrival"),
    }
    w, band = pick_winner_v3(sums, 0.8709, 0.01, "debt", band=0.005,
                             return_band=True, band_tiebreak="cost")
    assert w == "S0/rand", "the pre-registered ordering is unchanged by this check"
    assert cost_basis_conflict(sums, band) == ["at_last_arrival", "no_repair_needed"]
    # a band whose members share a basis is comparable, so nothing is reported
    assert cost_basis_conflict(sums, ["S2/demand_then_rand", "S2/rand"]) == []
    # exact ties are a different set from the band: two combos share this debt, three are in band
    assert tied_with(sums, "S2/demand_then_rand") == ["S2/rand", "S2/demand_then_rand"]
    assert len(band) == 3
    assert tied_with(sums, "S0/rand") == ["S0/rand"], "the winner alone, when nothing matches it"
    assert tied_with(sums, None) == [] and tied_with(sums, "S9/none") == []


def test_tau_curve_matches_the_winner_rule_at_the_config_tau(tmp):
    """The tau curve must be the same rule the run used, swept - not a reimplementation of it.
    At the config's own tau every row has to reproduce the winner the result already recorded."""
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    make_pair(tmp, N=600, Q=120, bm25_depth=30, src_dims=12, src_noise=0.4)
    cfg = dict(V3, traffic_ratio=[8.0, 32.0], primary_setting={"traffic_ratio": 32.0},
               tau={"debt": 0.005, "c_serve_frac": 0.02}, band_tiebreak="cost")
    out = os.path.join(tmp, "res.json")
    from reusemap import runner
    res = runner.run_pair(tmp, cfg, name="PT", log=lambda *_: None)
    runner.save(res, out)
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "tau_curve.py"), out,
                        "--taus", "0,0.005,0.02", "--csv", os.path.join(tmp, "tb")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import csv as _csv
    rows = list(_csv.DictReader(open(os.path.join(tmp, "tb", "tau_curve.csv"))))
    assert rows, r.stdout
    by = {(x["pair"], float(x["traffic_ratio"]), float(x["tau"])): x["winner"] for x in rows}
    for st in res["settings"]:
        got = by[("PT", st["traffic_ratio"], 0.005)]
        want = st["winner"] or ""
        assert got == want, (st["traffic_ratio"], got, want)
        # tau=0 can only narrow the band, never widen it
        assert int(next(x for x in rows if float(x["tau"]) == 0.0
                        and float(x["traffic_ratio"]) == st["traffic_ratio"])["n_in_band"]) \
            <= len(st["winner_ties"]) or not st["winner"]
    # development runs are refused unless asked for, like every other paper table
    res["config"]["_dev_overrides"] = {"traffic_ratio": [8.0]}
    runner.save(res, out)
    r2 = subprocess.run([sys.executable, os.path.join(root, "scripts", "tau_curve.py"), out],
                        capture_output=True, text=True)
    assert r2.returncode == 2 and "development overrides" in r2.stdout, (r2.returncode, r2.stdout)


def test_compat_refuses_degenerate_input_rather_than_scoring_it(tmp):
    """A collapsed encoding must not come back as a number. The obvious guard for CKA's zero
    denominator -- return 1.0 -- is the one answer that must not be given: when a block is
    constant the numerator is 0 too, so 1.0 would report a collapsed space as the most
    compatible pair in the study and send it straight into the map table."""
    import numpy as _np
    from reusemap import compat
    rng = _np.random.default_rng(0)
    good = rng.normal(size=(64, 8)).astype(_np.float32)
    flat = _np.tile(good[:1], (64, 1))                      # every row identical

    assert 0.0 <= compat.linear_cka(good, rng.normal(size=(64, 5)).astype(_np.float32)) <= 1.0001
    for X, Y, which in ((flat, good, "first"), (good, flat, "second"), (flat, flat, "both")):
        try:
            compat.linear_cka(X, Y)
        except ValueError as e:
            assert "no variance left" in str(e) and which in str(e), str(e)
        else:
            raise AssertionError(f"linear_cka scored a degenerate {which} block")

    ad = compat.fit_procrustes(good, good)
    try:
        compat.relative_residual(ad, good, _np.zeros_like(good))
    except ValueError as e:
        assert "zero norm" in str(e)
    else:
        raise AssertionError("relative_residual divided by a zero-norm source block")

    # the adapter maps TARGET rows into SOURCE space, so a swapped call must say so
    try:
        ad(rng.normal(size=(4, 3)).astype(_np.float32))
    except ValueError as e:
        assert "swapped" in str(e)
    else:
        raise AssertionError("the adapter multiplied through a wrong-width block")

    # mismatched corpora, unknown family, and too few documents to split
    for kw, msg in (({"doc_tgt": good[:32]}, "same corpus in the same order"),
                    ({"kind": "nope"}, "unknown adapter kind"),
                    ({"doc_src": good[:1], "doc_tgt": good[:1]}, "at least 2 documents")):
        args = {"doc_src": good, "doc_tgt": good.copy(), "kind": "procrustes"}
        args.update(kw)
        try:
            compat.compatibility(**args)
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(f"compatibility accepted {kw}")

    # and the normal path still works, on both families
    reps = {}
    for kind in ("procrustes", "affine"):
        _, reps[kind] = compat.compatibility(good, good.copy(), kind=kind, n_fit=20, n_eval=20)
        assert reps[kind]["adapter"] == kind and reps[kind]["n_fit"] == 20
    # an identity pair is recovered exactly by Procrustes, which is unregularised
    assert abs(reps["procrustes"]["kappa"] - 1.0) < 1e-3, reps["procrustes"]["kappa"]
    # affine with its penalty chosen on held-out pairs recovers it too
    assert reps["affine"]["kappa"] > 0.99, reps["affine"]["kappa"]


def test_secondary_frontier_measures_G_on_the_deeply_judged_set(tmp):
    """G(K') on the secondary set, which is the experiment that decides what G < 0 means.

    The primary metric is Recall@10 against sparse qrels (~1 judged passage per query, pooled
    from older systems). A window strategy is confined to the source index's top-K', so it
    returns documents closer to that pool, while exhaustive target retrieval surfaces unjudged
    documents that count as irrelevant by construction. That alone predicts G < 0 with no
    cascade effect -- and the 1M run reports G < 0 on 9 of 10 text pairs. This test checks the
    plumbing that lets the two be told apart: the secondary frontier must be computed on the
    secondary queries, uniformly weighted (their workload weight is zero), against an M_T
    measured on those same queries.
    """
    import numpy as _np
    from reusemap import measure, pair, runner
    make_pair(tmp, N=600, Q=120, bm25_depth=30, src_dims=12, src_noise=0.4)
    # hold out the last 20 queries as the deeply-judged secondary set
    sec = list(range(100, 120))
    sec_path = os.path.join(tmp, "trecdl.txt")
    with open(sec_path, "w") as fh:
        fh.write("\n".join(str(i) for i in sec))
    cfg = dict(V3, secondary_eval_qids=sec_path,
               secondary=[{"metric": "recall", "thr": 1}])
    p = pair.Pair(tmp, cfg)
    p.precompute()                 # tgt_top/src_top, as runner.run_pair does first
    assert sorted(int(x) for x in p.secondary_q) == sec, p.secondary_q
    # the secondary queries really are evaluation-only: excluded from the SERVING weights.
    # p.weights is the raw workload over all Q queries and is NOT zero there, which is why the
    # frontier below weights them uniformly itself rather than trusting the default.
    assert float(p.all_query_weights()[p.secondary_q].sum()) == 0.0, \
        "secondary set must never be served"
    assert float(p.traffic_weights(0.0)[p.secondary_q].sum()) == 0.0

    sf = measure.secondary_frontier(p, p.kprime_list(), "recall", 1)
    assert sf and sf["queries"] == len(sec), sf
    assert set(sf["rho"]) == set(sf["G"]), (sorted(sf["rho"]), sorted(sf["G"]))
    assert 0.0 <= sf["M_T"] <= 1.0, sf["M_T"]
    # rho is non-increasing in K' here too, since windows are prefixes of one retrieval
    ks = sorted(sf["rho"])
    assert all(sf["rho"][a] >= sf["rho"][b] - 1e-12 for a, b in zip(ks, ks[1:])), sf["rho"]
    # at a window as deep as the corpus the re-rank sees everything, so G must vanish
    deep = measure.window_gap(p, [p.N], sf["M_T"], "recall", 1,
                              queries=p.secondary_q,
                              weights=_np.where(_np.isin(_np.arange(p.Q), p.secondary_q), 1.0, 0.0))
    if deep:
        assert abs(next(iter(deep.values()))) < 1e-9, deep

    # defaults must be untouched: the primary numbers cannot move
    assert measure.residual_miss(p, p.kprime_list()) == \
        measure.residual_miss(p, p.kprime_list(), "src", None, None)
    # a query set carrying no weight is refused rather than normalised by zero. This is the
    # real case under an ORCAS workload, where TREC-DL queries have no clicks and so no weight
    try:
        measure.residual_miss(p, p.kprime_list(), "src", queries=p.secondary_q,
                              weights=_np.zeros(p.Q))
    except ValueError as e:
        assert "zero total weight" in str(e), str(e)
    else:
        raise AssertionError("residual_miss normalised by a zero total weight")

    # BOTH functions take `weights` indexed by query id. A per-position array of len(queries)
    # is rejected, not used: the two are indistinguishable by value, and the positional
    # reading would silently score the wrong queries. residual_miss multiplied positionally
    # for one round while window_gap indexed by id, and its docstring claimed they matched.
    for fn, args in ((measure.residual_miss, (p, p.kprime_list(), "src")),
                     (measure.window_gap, (p, p.kprime_list(), sf["M_T"], "recall", 1, "src"))):
        try:
            fn(*args, queries=p.secondary_q, weights=_np.ones(len(p.secondary_q)))
        except ValueError as e:
            assert "indexed by query id" in str(e), (fn.__name__, str(e))
        else:
            raise AssertionError(f"{fn.__name__} accepted a per-position weights array")
    # and the id-indexed form gives the same answer through either function's path
    wfull = _np.zeros(p.Q); wfull[p.secondary_q] = 1.0
    assert measure.residual_miss(p, p.kprime_list(), "src",
                                 queries=p.secondary_q, weights=wfull) == sf["rho"]

    # and the runner attaches it to a measure-only result
    res = runner.run_pair(tmp, dict(cfg, replay=False), name="SEC", log=lambda *_: None)
    assert res["secondary_frontier"]["queries"] == len(sec), res["secondary_frontier"]
    prim = cfg.get("primary") or runner.DEFAULT_CFG["primary"]
    direct = measure.window_gap(p, p.kprime_list(), res["reference"]["M_T"],
                                prim["metric"], prim["thr"], "src")
    # res["G"] comes back through JSON, so its depths are strings
    assert {int(k): v for k, v in res["G"].items()} == direct, \
        f"the primary G must be unchanged by adding the secondary frontier: {res['G']} vs {direct}"


def test_background_order_moves_debt_when_the_background_gets_ahead(tmp):
    """Background order is NOT irrelevant to served quality -- that was a claim made from the
    primary setting alone and generalised wrongly.

    Window strategies also search the target-side index over already-migrated documents, so
    what the background has reached does change the served answer. At high traffic intensity
    the evaluation population is served before the background reaches anything, so every order
    ties and the winner's order component is arbitrary. At low intensity the background covers
    much of the corpus first and the orders separate. On the 1M run 42 of 48
    (pair x setting x strategy) cells varied somewhere, including the primary setting on one
    pair. This test pins both halves so the convenient version cannot come back.

    Note the non-monotonicity, which is the shape of the effect: orders tie at BOTH extremes,
    for opposite reasons. At very low intensity the background finishes before the population
    is served, so every order ends in the same fully-migrated state; at very high intensity it
    has not started. Separation lives in between -- on this fixture at ratios 2 to 16, peaking
    at 8, and on the 1M run at 4 and 16.
    """
    from reusemap import runner
    make_pair(tmp, N=400, Q=80, bm25_depth=20, src_dims=10, src_noise=0.4)

    def distinct_debts(ratio):
        cfg = dict(V3, traffic_ratio=[ratio], repeat_rate=[0.0],
                   primary_setting={"traffic_ratio": ratio, "repeat_rate": 0.0})
        res = runner.run_pair(tmp, cfg, name=f"ORD{ratio:g}", log=lambda *_: None)
        sums = res["settings"][0]["summaries"]
        return sorted({round(float(v["debt"]), 12) for c, v in sums.items()
                       if c.startswith("S2/") and v.get("completed")
                       and isinstance(v.get("debt"), (int, float))})

    sparse, mid, dense = distinct_debts(0.5), distinct_debts(8.0), distinct_debts(256.0)
    assert len(dense) == 1, f"at ratio 256 every completing order must tie, got {dense}"
    # the other end: the background finishes before the population is served, so every order
    # ends in the same state. A broken background budget or delay would separate them here.
    assert len(sparse) == 1, f"at ratio 0.5 every completing order must tie, got {sparse}"
    assert len(mid) >= 2, (
        f"at ratio 8 the background is partway through when the population is served, so the "
        f"orders must separate; got one value {mid}. If this ever legitimately ties, the claim "
        f"to make is about that operating point and not about the method -- see the 2026-11-05 "
        f"DECISIONS.md row.")


def test_build_subsample_refuses_a_duplicate_pid(tmp):
    """A duplicate pid used to collapse the document index space silently.

    `pid2idx[pid] = len(pid2idx)` re-assigns an existing key without growing the dict, so one
    duplicate writes two rows sharing an index, shifts every later pid down by one, and leaves
    "judged coverage: 100.00%" printing a pass. Before the guard, a collection of A,B,B,C,D,E
    produced index column 0,1,2,2,3,4 -- C and the duplicate B both at 2 -- so a qrel for C
    resolved to B's text, and M_T and G were computed against the wrong passages.
    """
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    d = os.path.join(tmp, "dup")
    os.makedirs(d, exist_ok=True)

    def write(name, text):
        with open(os.path.join(d, name), "w") as fh:
            fh.write(text)

    write("qrels.tsv", "q1\t0\tA\t1\nq2\t0\tC\t1\nq3\t0\tD\t1\n")
    write("queries.tsv", "q1\tquery one\nq2\tquery two\nq3\tquery three\n")

    def run(collection, out):
        write("collection.tsv", collection)
        return subprocess.run(
            [sys.executable, os.path.join(root, "scripts", "build_subsample.py"),
             "--collection", os.path.join(d, "collection.tsv"),
             "--qrels", os.path.join(d, "qrels.tsv") + ":dev",
             "--queries", os.path.join(d, "queries.tsv") + ":dev",
             "--N", "6", "--out", os.path.join(d, out)],
            capture_output=True, text=True)

    # a line with no tab is refused by name, not unpacked into a ValueError
    r = run("A\ta\nB\tb\nnotab\nD\td\nE\te\nF\tf\n", "notab")
    assert r.returncode != 0 and "no tab" in (r.stdout + r.stderr), (r.stdout, r.stderr)

    dup = "A\ta\nB\tb\nB\tb duplicate\nC\tc\nD\td\nE\te\n"
    r = run(dup, "bad")
    assert r.returncode != 0, f"a duplicate pid was accepted:\n{r.stdout}"
    assert "duplicate pid" in (r.stdout + r.stderr), (r.stdout, r.stderr)
    assert "'B'" in (r.stdout + r.stderr), "the message must name the offending pid"

    # the clean collection still works, and its index column is exactly 0..n-1 with one row each
    r = run("A\ta\nB\tb\nC\tc\nD\td\nE\te\nF\tf\n", "good")
    assert r.returncode == 0, (r.stdout, r.stderr)
    rows = [l.rstrip("\n").split("\t") for l in
            open(os.path.join(d, "good", "corpus.tsv"))]
    idx = [int(x[0]) for x in rows]
    assert idx == list(range(len(rows))), idx
    assert len({x[1] for x in rows}) == len(rows), "pids must stay unique"
    # and every qrel resolves to the pid it was written for
    pid_of = {int(x[0]): x[1] for x in rows}
    want = {"q1": "A", "q2": "C", "q3": "D"}
    qrows = [l.split() for l in open(os.path.join(d, "good", "qrels.tsv"))]
    qname = {int(l.split("\t")[0]): l.split("\t")[1]
             for l in open(os.path.join(d, "good", "queries.tsv"))}
    assert len(qrows) == 3, qrows
    for q, p, _rel in qrows:
        assert pid_of[int(p)] == want[qname[int(q)]], (q, p, pid_of[int(p)])


def test_build_subsample_flattens_tabs_so_encode_reads_the_whole_text(tmp):
    """A tab inside a passage used to truncate what got encoded, silently.

    corpus.tsv is `idx TAB pid TAB text` and encode.py reads the text as split("\\t")[-1], so
    one embedded tab meant only the fragment after it was ever encoded -- wrong vectors, with
    every row count still matching and nothing checking the text. build_code_corpus.py and
    build_nq_prefix.py already flattened; this is the MS MARCO path.
    """
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    d = os.path.join(tmp, "tabs")
    os.makedirs(d, exist_ok=True)

    def write(name, text):
        with open(os.path.join(d, name), "w") as fh:
            fh.write(text)

    # A's passage and q1's text both carry an embedded tab. (A lone CR cannot be tested through
    # the file: Python reads text with universal newlines, so it becomes a line break before
    # build_subsample ever sees the field, and the malformed line raises there instead.)
    write("collection.tsv", "A\tfirst half\tsecond half\nB\tb text\nC\tc text\nD\td\nE\te\n")
    write("qrels.tsv", "q1\t0\tA\t1\nq2\t0\tC\t1\n")
    write("queries.tsv", "q1\tquery\tone\nq2\tquery two\n")
    r = subprocess.run(
        [sys.executable, os.path.join(root, "scripts", "build_subsample.py"),
         "--collection", os.path.join(d, "collection.tsv"),
         "--qrels", os.path.join(d, "qrels.tsv") + ":dev",
         "--queries", os.path.join(d, "queries.tsv") + ":dev",
         "--N", "5", "--out", os.path.join(d, "out")],
        capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)

    for name, fields in (("corpus.tsv", 3), ("queries.tsv", 4)):
        for line in open(os.path.join(d, "out", name)):
            parts = line.rstrip("\n").split("\t")
            assert len(parts) == fields, f"{name}: {len(parts)} fields in {line!r}"
    # the whole passage survives, which is what encode.py's split("\t")[-1] now receives
    texts = {l.split("\t")[1]: l.rstrip("\n").split("\t")[-1]
             for l in open(os.path.join(d, "out", "corpus.tsv"))}
    assert texts["A"] == "first half second half", texts["A"]
    qtexts = [l.rstrip("\n").split("\t")[-1]
              for l in open(os.path.join(d, "out", "queries.tsv"))]
    assert "query one" in qtexts, qtexts


def test_compatibility_reports_cka_as_nan_rather_than_failing_at_two_documents(tmp):
    """The residual needs one eval row; CKA needs two. At N=2 the collapse guard added for a
    genuinely collapsed encoding fired on a single centred row instead, failing the whole
    measurement with a message pointing at the encoder."""
    import numpy as _np
    import math
    from reusemap import compat
    g = _np.random.default_rng(0).normal(size=(12, 5)).astype(_np.float32)
    _, rep = compat.compatibility(g[:2], g[:2].copy())
    assert rep["n_fit"] == 1 and rep["n_eval"] == 1, rep
    assert math.isnan(rep["cka"]), rep["cka"]
    assert math.isfinite(rep["kappa"]), "the residual is still meaningful at one eval row"
    for N in (3, 4, 12):
        _, rep = compat.compatibility(g[:N], g[:N].copy())
        assert math.isfinite(rep["cka"]), (N, rep["cka"])
    # called directly, linear_cka still refuses what it cannot define, and says which way
    for X, Y, msg in ((g[:1], g[:1], "at least 2 samples"), (g[:4], g[:3], "same sample count")):
        try:
            compat.linear_cka(X, Y)
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(f"linear_cka accepted {X.shape} vs {Y.shape}")
    # a mis-shaped bias is caught at construction, not at the first call
    try:
        compat.Adapter(_np.zeros((4, 3), _np.float32), _np.zeros(4, _np.float32))
    except ValueError as e:
        assert "b of shape" in str(e), str(e)
    else:
        raise AssertionError("Adapter accepted a bias of the wrong width")


def _result(name, run_key):
    return {"name": name, "provenance": {"run_key": run_key}}


def test_table_sidecar_catches_a_table_left_behind_by_a_rerun(tmp):
    """The 2026-11-07 archive's tau_curve.csv described the previous generation while map.csv
    and grid.csv beside it described the current one -- 177 of 416 rows disagreed, and nothing
    in the directory said so. The check has to fire on exactly that: same pair, new run_key."""
    from reusemap import provenance as prov
    before = [_result("P2", "aaaa1111"), _result("P11", "bbbb2222")]
    csv_path = os.path.join(tmp, "tau_curve.csv")
    open(csv_path, "w").close()
    prov.write_table_inputs(csv_path, before, tmp)
    if prov.check_table_inputs(csv_path, before):
        raise AssertionError("a freshly written table was reported stale against its own inputs")
    after = [_result("P2", "aaaa1111"), _result("P11", "cccc3333")]  # P11 re-run
    msgs = prov.check_table_inputs(csv_path, after)
    if not any("P11" in m and "older run" in m for m in msgs):
        raise AssertionError(f"re-running P11 did not make the table stale: {msgs}")
    if any("P2" in m for m in msgs):
        raise AssertionError(f"P2 did not change and must not be reported: {msgs}")


def test_table_sidecar_allows_excluded_pairs_but_not_a_missing_one(tmp):
    """`summarize.py --exclude P0 P6` is the documented way to build the paper tables, and the
    measure-only pairs appear in calib and nowhere else, so a result absent from a table is
    normal. A result the table WAS built from going missing is not."""
    from reusemap import provenance as prov
    csv_path = os.path.join(tmp, "map.csv")
    open(csv_path, "w").close()
    prov.write_table_inputs(csv_path, [_result("P2", "aaaa1111")], tmp)
    extra = [_result("P2", "aaaa1111"), _result("P0", "dddd4444")]
    if prov.check_table_inputs(csv_path, extra):
        raise AssertionError("a result the table excludes must not count as staleness")
    if not prov.check_table_inputs(csv_path, [_result("P0", "dddd4444")]):
        raise AssertionError("a recorded input that is gone must be reported")


def test_table_with_no_sidecar_is_reported_rather_than_passing(tmp):
    """Every table shipped so far has no sidecar. Silence there would make the checker pass on
    exactly the archives it exists to judge, so an unstamped table is a problem, not an ok."""
    from reusemap import provenance as prov
    csv_path = os.path.join(tmp, "grid.csv")
    open(csv_path, "w").close()
    msgs = prov.check_table_inputs(csv_path, [_result("P2", "aaaa1111")])
    if not msgs or "inputs.json" not in msgs[0]:
        raise AssertionError(f"an unstamped table must be reported: {msgs}")


def _load_plan():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "plan_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "scripts", "plan.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_planner_decision_cannot_see_the_replayed_winner(tmp):
    """The planner's table is only readable as a test if the decision is made from static
    inputs alone. `decide()` takes no result dict and no settings, so there is nowhere for a
    winner to enter; this pins that signature rather than trusting it."""
    import inspect
    plan = _load_plan()
    params = set(inspect.signature(plan.decide).parameters)
    leaky = {"settings", "result", "r", "summaries", "winner", "replay"}
    if params & leaky:
        raise AssertionError(f"decide() takes replay-derived arguments: {params & leaky}")
    for want in ("M_S", "M_T", "g_ladder", "coverage_rows", "tau", "eps"):
        if want not in params:
            raise AssertionError(f"decide() lost its static input {want!r}: {sorted(params)}")


def test_planner_d1_sends_a_gap_inside_the_band_to_s0(tmp):
    """D1 is the whole of the migrate-at-all decision: a target no better than the source by
    more than tau is not worth migrating to, and the measured pairs separate cleanly
    (max 'do not migrate' gap +0.00082, min 'migrate' gap +0.00657, tau 0.005 between them)."""
    plan = _load_plan()
    rows = [{"kprime": 50, "coverage": 0.2649, "ceiling_frac": 0.7351}]
    inside = plan.decide("Px", 0.80, 0.80082, {50: 0.0}, rows, tau=0.005, eps=0.01)
    if inside["family"] != "S0" or inside["kprime"] is not None:
        raise AssertionError(f"a gap inside the band must pick S0 with no depth: {inside}")
    outside = plan.decide("Py", 0.80, 0.80657, {50: 0.0}, rows, tau=0.005, eps=0.01)
    if outside["family"] != "S2" or outside["kprime"] != 50:
        raise AssertionError(f"a gap outside the band must pick the window: {outside}")


def test_planner_d3_refuses_a_depth_that_gives_back_the_upgrade(tmp):
    """D3 is not "smallest K' with G <= eps". That minimises cost subject to admissibility and
    never looks at debt, but the window serves target quality minus the window gap, so a
    shallower window raises debt. P5 is the real case: quality gap +0.00657 and G(10) = +0.00657,
    so at K' = 10 the window hands back exactly what migrating was worth, and the one-condition
    rule chose 10 there. The depth must also leave the window more than tau better than not
    migrating -- both quantities static, so the second condition is free.
    """
    plan = _load_plan()
    rows = [{"kprime": k, "coverage": c, "ceiling_frac": 1 - c}
            for k, c in ((10, 0.06), (25, 0.14), (50, 0.26))]
    ladder = {10: 0.00657, 25: -0.00883, 50: -0.00692}   # P5's measured frontier
    d = plan.decide("P5", 0.50, 0.50657, ladder, rows, tau=0.005, eps=0.01)
    if d["kprime"] == 10:
        raise AssertionError("D3 chose a depth where the window gives back the whole upgrade")
    if d["family"] != "S2" or d["kprime"] != 25:
        raise AssertionError(f"D3 should fall through to the next usable depth, got {d}")
    if "wins nothing" not in (d["why"] or ""):
        raise AssertionError(f"the reason must say why 10 was rejected: {d['why']!r}")
    # and a pair with a wide gap keeps the shallow depth: the rule must not just always skip one
    wide = plan.decide("Pw", 0.50, 0.56, ladder, rows, tau=0.005, eps=0.01)
    if wide["kprime"] != 10:
        raise AssertionError(f"a wide gap should still allow K'=10, got {wide['kprime']}")


def test_planner_refuses_a_window_whose_coverage_leaves_no_saving(tmp):
    """The code corpus has coverage 1.0 at every depth, so the window would touch the whole
    index and there is nothing to save -- D2 must say rebuild, not 'window at cost 1.0'."""
    plan = _load_plan()
    rows = [{"kprime": 50, "coverage": 1.0, "ceiling_frac": 0.0}]
    d = plan.decide("Pz", 0.70, 0.75, {50: 0.0}, rows, tau=0.005, eps=0.01)
    if d["family"] != "S0":
        raise AssertionError(f"a zero ceiling must send D2 to a rebuild: {d}")
    if "no saving" not in (d["why"] or ""):
        raise AssertionError(f"the reason must name the missing saving: {d['why']!r}")


def test_freeze_check_prints_an_unmeasured_cell_instead_of_dropping_it(tmp):
    """The 2026-11-07 FREEZE_CHECK.md was truncated mid-JSON at `"P11": [`, losing the four
    pairs that decide the window depth -- and nothing in the file said anything was missing.
    The generator must make absence visible: an unmeasured (pair, K') cell appears as
    `not measured`, the pair's depth reads `undecided` rather than a number inferred from the
    rungs that happen to exist, and K* is withheld entirely.
    """
    import subprocess
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    res = os.path.join(tmp, "Pz.json")
    with open(res, "w", encoding="utf-8") as fh:
        json.dump({"name": "Pz", "N": 1000, "window": 50,
                   "G": {"50": -0.001},                       # 10 and 25 absent
                   "config": {"eps": 0.01},
                   "provenance": {"run_key": "abc123", "git": "deadbeef"}}, fh)
    out = os.path.join(tmp, "FC.md")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "freeze_check.py"),
                        res, "--out", out, "--kprimes", "10,25,50"],
                       capture_output=True, text=True)
    if r.returncode != 1:
        raise AssertionError(f"an incomplete frontier must exit 1, got {r.returncode}: "
                             f"{r.stdout}{r.stderr}")
    text = open(out, encoding="utf-8").read()
    if text.count("not measured") != 2:
        raise AssertionError(f"both missing rungs must be printed:\n{text}")
    if "undecided" not in text:
        raise AssertionError("a pair with a missing rung must read 'undecided'")
    blocks = re.findall(r"```json\n(.*?)```", text, flags=re.S)
    if not blocks:
        raise AssertionError("no machine-readable block written")
    rec = json.loads(blocks[0])
    if rec["k_star"] is not None:
        raise AssertionError(f"K* must be withheld while cells are missing: {rec['k_star']}")
    if rec["G"]["Pz"]["25"] is not None:
        raise AssertionError("an unmeasured cell must serialise as null, not be omitted")


def test_freeze_check_withholds_k_star_when_a_pair_is_admissible_at_no_depth(tmp):
    """All cells measured is not the same as K* existing. The first version took max() over the
    pairs that HAD an admissible depth and skipped those that did not, so a pair whose G exceeds
    eps at every rung was dropped from the maximum -- and the record printed `K* = 25` with the
    sentence "every pair is admissible at this depth", which was false, and exited 0. Certifying
    a frozen depth no measurement supports is the one thing this file must never do.
    """
    import subprocess
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    for name, g in (("Pa", {"10": 0.02, "25": 0.004, "50": 0.001}),
                    ("Pb", {"10": 0.9, "25": 0.8, "50": 0.7})):   # Pb: never admissible
        with open(os.path.join(tmp, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"name": name, "N": 1000, "window": 50, "G": g,
                       "config": {"eps": 0.01},
                       "provenance": {"run_key": "k" + name, "git": "x"}}, fh)
    out = os.path.join(tmp, "FC.md")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "freeze_check.py"),
                        os.path.join(tmp, "Pa.json"), os.path.join(tmp, "Pb.json"),
                        "--out", out, "--kprimes", "10,25,50"], capture_output=True, text=True)
    if r.returncode == 0:
        raise AssertionError(f"exited 0 with an impossible K*:\n{r.stdout}")
    text = open(out, encoding="utf-8").read()
    if re.search(r"\*\*K\\\* = \d", text):
        raise AssertionError(f"printed a K* that does not exist:\n{text}")
    rec = json.loads(re.findall(r"```json\n(.*?)```", text, flags=re.S)[0])
    if rec.get("inadmissible") != ["Pb"]:
        raise AssertionError(f"the record must name the pair that blocks K*: {rec.get('inadmissible')}")
    if rec["k_star"] is not None:
        raise AssertionError(f"k_star must be null, got {rec['k_star']}")


def test_freeze_check_reads_the_11_1_sweep_files(tmp):
    """The shallow rungs (10/25/50) exist only in sweep.py's outputs. freeze_check read result
    JSONs alone, so on the real data it reported 39 cells 'not measured' and K* undecided."""
    import subprocess
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    json.dump({"name": "Pa", "N": 1000, "window": 50, "G": {"500": 0.0},
               "config": {"eps": 0.01}, "provenance": {"run_key": "ka", "git": "x"}},
              open(os.path.join(tmp, "Pa.json"), "w"))
    json.dump({"pair": "Pa", "depths": [{"kprime": 10, "G": 0.05}, {"kprime": 25, "G": 0.02},
                                        {"kprime": 50, "G": 0.004}]},
              open(os.path.join(tmp, "sweep_Pa.json"), "w"))
    out = os.path.join(tmp, "FC.md")
    subprocess.run([sys.executable, os.path.join(root, "scripts", "freeze_check.py"),
                    os.path.join(tmp, "Pa.json"), os.path.join(tmp, "sweep_Pa.json"),
                    "--out", out, "--kprimes", "10,25,50"], capture_output=True, text=True)
    rec = json.loads(re.findall(r"```json\n(.*?)```", open(out, encoding="utf-8").read(), flags=re.S)[0])
    if rec["k_star"] != 50 or rec.get("unmeasured"):
        raise AssertionError(f"sweep rungs not read: k_star={rec['k_star']}, unmeasured={rec.get('unmeasured')}")


def test_sample_sweep_writes_its_output(tmp):
    """A guard inserted at the wrong indentation once moved the whole main block under
    `if not isdir(results): sys.exit(...)`: with a valid --results the script exited 0 and wrote
    nothing, and the driver's sample stage 'succeeded'. Run it end to end and require the file."""
    import subprocess
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    pd = os.path.join(tmp, "PX"); make_pair(pd, N=600, Q=60, bm25_depth=30)
    os.makedirs(os.path.join(tmp, "res"))
    json.dump({"reference": {"M_S": 0.50, "M_T": 0.55}}, open(os.path.join(tmp, "res", "PX.json"), "w"))
    json.dump({}, open(os.path.join(tmp, "cfg.json"), "w"))
    out = os.path.join(tmp, "out")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "sample_sweep.py"), pd,
                        "--config", os.path.join(tmp, "cfg.json"), "--results", os.path.join(tmp, "res"),
                        "--fracs", "0.25", "--seeds", "1", "--out-dir", out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(out, "PX.json")):
        raise AssertionError(f"no output written (exit {r.returncode}):\n{r.stdout[-500:]}{r.stderr[-500:]}")
    if "smallest_frac_all_seeds_agree" not in json.load(open(os.path.join(out, "PX.json"))):
        raise AssertionError("output lacks the sweep's verdict")


def test_pair_refuses_malformed_query_vectors_with_the_file_name(tmp):
    from reusemap.pair import Pair
    make_pair(tmp, N=200, Q=20, bm25_depth=20)
    np.save(os.path.join(tmp, "q_tgt.npy"), np.zeros(5, np.float32))      # 1-D
    try:
        Pair(tmp, dict(CFG))
    except ValueError as e:
        if "q_tgt.npy" not in str(e):
            raise AssertionError(f"error does not name the file: {e}")
    else:
        raise AssertionError("a 1-D q_tgt.npy was accepted")


def test_check_planner_scores_a_plan_committed_before_its_replays(tmp):
    """A plan computed from static inputs, before any replay, writes verdict 'no replay' on every
    row. check_planner skipped on that column, so a genuinely pre-committed plan scored nothing."""
    import subprocess
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    plan_csv = os.path.join(tmp, "plan.csv")
    open(plan_csv, "w").write("pair,family,kprime,cost,verdict\nNa,S0,,,no replay\n")
    os.makedirs(os.path.join(tmp, "res"))
    json.dump({"window": 50, "G": {}, "primary_setting": {"traffic_ratio": 64.0},
               "settings": [{"traffic_ratio": 64.0, "winner": "S0/rand", "summaries": {}}]},
              open(os.path.join(tmp, "res", "Na.json"), "w"))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "check_planner.py"), "--plan", plan_csv,
                        "--results", os.path.join(tmp, "res")], capture_output=True, text=True)
    if "1 pairs scorable" not in r.stdout:
        raise AssertionError(f"a replayed pair was not scored:\n{r.stdout}{r.stderr}")
    # a replay with NO WINNER at the primary setting (runner warns, winner None) is reported, not a crash
    open(plan_csv, "a").write("Nb,S2,50,0.25,no replay\n")
    json.dump({"window": 50, "G": {}, "primary_setting": {"traffic_ratio": 64.0},
               "settings": [{"traffic_ratio": 64.0, "winner": None, "summaries": {}}]},
              open(os.path.join(tmp, "res", "Nb.json"), "w"))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "check_planner.py"), "--plan", plan_csv,
                        "--results", os.path.join(tmp, "res")], capture_output=True, text=True)
    if r.returncode != 0 or "no winner" not in r.stdout:
        raise AssertionError(f"a replay with no winner was not reported:\n{r.stdout}{r.stderr}")
    # and seed_summary counts it instead of crashing
    def res(seed, winner):
        return {"config": {"seed": seed, "tau": 0.005}, "primary_setting": {"traffic_ratio": 64.0},
                "settings": [{"traffic_ratio": 64.0, "winner": winner, "summaries": {
                    "S4/demand_then_rand": {"debt": 0.0, "C_serve_frac": 0.8},
                    "S2/demand_then_rand": {"debt": 0.006, "C_serve_frac": 0.26}}}]}
    fs = []
    for seed, w in ((0, "S4/demand_then_rand"), (1, None)):
        fs.append(os.path.join(tmp, f"s{seed}.json")); json.dump(res(seed, w), open(fs[-1], "w"))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "seed_summary.py")] + fs,
                       capture_output=True, text=True)
    if r.returncode != 0 or "wins on 1 of 2 (1 with no winner)" not in r.stdout:
        raise AssertionError(f"seed_summary on a seed with no winner:\n{r.stdout}{r.stderr}")


def test_primary_setting_is_chosen_by_ratio_not_by_winner_equality(tmp):
    """`map.csv`'s tied_exact came from `next(s for s in settings if s.winner == r.winner)`.
    Window strategies win at every traffic ratio, so that matched ratio 4 while primary_setting
    is ratio 64 -- on 10 of 13 pairs in the 2026-11-07 run, with tied_exact wrong on 8 (P11 read
    4 ties instead of 6). tied_exact is printed in the paper, so it was a wrong Table 1 number.
    """
    from reusemap.winner import primary_setting
    def st(ratio, ties):
        return {"traffic_ratio": ratio, "winner": "S2/rand", "tied_exact": ["S2/rand"] * ties}
    r = {"winner": "S2/rand", "primary_setting": {"traffic_ratio": 64.0},
         "settings": [st(4.0, 1), st(16.0, 1), st(64.0, 6)]}
    got = primary_setting(r)
    if got.get("traffic_ratio") != 64.0:
        raise AssertionError(f"picked ratio {got.get('traffic_ratio')}, want 64.0")
    if len(got.get("tied_exact") or []) != 6:
        raise AssertionError(f"tied_exact came from the wrong setting: "
                             f"{len(got.get('tied_exact') or [])}, want 6")
    # the old expression is what must NOT be reproduced
    old = next((x for x in r["settings"] if x.get("winner") == r.get("winner")), {})
    if old.get("traffic_ratio") == 64.0:
        raise AssertionError("fixture does not exercise the bug: make ratios share a winner")
    # and a result with no primary_setting yields {} rather than a wrong guess
    if primary_setting({"settings": [st(4.0, 1)]}) != {}:
        raise AssertionError("a result without primary_setting must yield {}")
    if primary_setting({"primary_setting": {"traffic_ratio": 999.0},
                        "settings": [st(4.0, 1)]}) != {}:
        raise AssertionError("an absent ratio must yield {} rather than a fallback setting")


def test_kprime_list_rejects_a_non_positive_depth(tmp):
    """A K' of 0 or below reaches topk_ip as k=0 and then np.argpartition(-sc, -1), which does
    not mean what the caller intended. Refuse it where the list is built."""
    from reusemap.pair import Pair
    make_pair(tmp, N=600, Q=60, bm25_depth=30)
    for bad in ([0, 50], [-10, 50]):
        try:
            Pair(tmp, {"K": 10, "kprime": bad, "window": 50}).kprime_list()
        except ValueError:
            continue
        except Exception as e:
            raise AssertionError(f"kprime={bad} raised {type(e).__name__}, want ValueError: {e}")
        raise AssertionError(f"kprime={bad} was accepted")


def test_every_source_file_parses_on_the_oldest_python_available(tmp):
    """plan.py shipped a nested f-string that reused the outer quote -- `f'{x['cost']:.4f}'`
    inside an f"..." -- which is PEP 701, valid on 3.12+ and a SyntaxError on 3.11 and earlier
    ("f-string: unmatched '['"). It parsed fine here, so nothing caught it; a reviewer on a
    stock 3.11 would have found the script unrunnable.

    `ast.parse(feature_version=(3, 11))` does NOT catch this, because feature_version does not
    revert the tokenizer. The only real check is an older parser, so this shells out to one if
    the machine has it and says so plainly when it does not.
    """
    import glob
    import shutil
    import subprocess
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    older = next((p for p in ("python3.9", "python3.10", "python3.11")
                  if shutil.which(p)), None)
    if older is None:
        print("    (no python3.9-3.11 on this machine; PEP 701 regressions cannot be checked"
              " here -- run this test on a box with one before releasing an artifact)", end="")
        return
    files = sorted(glob.glob(os.path.join(root, "scripts", "*.py"))
                   + glob.glob(os.path.join(root, "reusemap", "*.py"))
                   + glob.glob(os.path.join(root, "tests", "*.py")))
    prog = ("import ast,sys\n"
            "bad=[]\n"
            "for f in sys.argv[1:]:\n"
            "    try: ast.parse(open(f,encoding='utf-8').read())\n"
            "    except SyntaxError as e: bad.append('%s:%s %s' % (f, e.lineno, e.msg))\n"
            "print('\\n'.join(bad))\n")
    r = subprocess.run([older, "-c", prog] + files, capture_output=True, text=True)
    out = (r.stdout or "").strip()
    if out:
        raise AssertionError(f"{older} cannot parse:\n{out}")



def test_lowrank_at_full_rank_equals_affine(tmp):
    from reusemap.compat import fit_affine, fit_lowrank
    rng = np.random.default_rng(3)
    Xt = rng.standard_normal((800, 12)).astype(np.float32)
    Xs = (Xt @ rng.standard_normal((12, 8)) + 0.1 * rng.standard_normal((800, 8))).astype(np.float32)
    a, l = fit_affine(Xt, Xs), fit_lowrank(Xt, Xs, rank=8)
    assert np.allclose(a(Xt[:50], normalize=False), l(Xt[:50], normalize=False), atol=1e-4)
    l2 = fit_lowrank(Xt, Xs, rank=2)
    assert np.linalg.matrix_rank(l2.W, tol=1e-4) == 2


def test_mlp_adapter_learns_a_nonlinear_map_and_never_ends_below_its_start(tmp):
    from reusemap.compat import fit_affine, fit_mlp
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((4000, 12)).astype(np.float32)
    Xt = np.tanh(Z @ rng.standard_normal((12, 24))).astype(np.float32)
    Xs = (Z @ rng.standard_normal((12, 16))).astype(np.float32)
    def cos(ad, X, Y):
        P = ad(X, normalize=False)
        return float(((P / np.linalg.norm(P, axis=1, keepdims=True)) *
                      (Y / np.linalg.norm(Y, axis=1, keepdims=True))).sum(1).mean())
    base = cos(fit_affine(Xt[:3000], Xs[:3000]), Xt[3000:], Xs[3000:])
    m = fit_mlp(Xt[:3000], Xs[:3000], hidden=256, epochs=60)
    assert cos(m, Xt[3000:], Xs[3000:]) > base + 0.005, (base, cos(m, Xt[3000:], Xs[3000:]))
    # zero epochs: the residual branch is exactly zero, so the MLP IS its linear start
    m0 = fit_mlp(Xt[:3000], Xs[:3000], hidden=16, epochs=0, lr=1e-3)
    assert not m0.W2.any()


def test_affine_ridge_is_chosen_on_held_out_pairs_and_does_not_collapse(tmp):
    """Embedding vectors share a large common direction, so after centring the Gram matrix has
    a few large eigenvalues and many tiny ones. A fixed penalty of 1e-2 * n (lambda = 100 at
    n = 10,000) then exceeds nearly every eigenvalue and shrinks the affine map onto the mean
    vector: every query lands on the same documents. That is what the first run of
    adapter_families.py measured (relative recall 0-19% on the 1M text pairs, Procrustes
    37-89%). The default now picks the penalty on held-out pairs; it must beat the old one."""
    from reusemap.compat import fit_affine, fit_lowrank
    rng = np.random.default_rng(0)
    def emb(Z, d):
        X = Z @ (rng.standard_normal((Z.shape[1], d)) * np.linspace(1, 0.02, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        mu = rng.standard_normal(d); mu /= np.linalg.norm(mu)
        X = 0.66 * X + 0.75 * mu
        return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    Z = rng.standard_normal((6500, 32))
    S, T = emb(Z, 96), emb(Z + 0.3 * Z @ rng.standard_normal((32, 32)), 128)
    fit, q = np.arange(5000), np.arange(6000, 6500)
    truth = np.argsort(-(S[q] @ S[:6000].T), 1)[:, :10]
    def rec(ad):
        top = np.argsort(-(ad(T[q]) @ S[:6000].T), 1)[:, :10]
        return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, truth)])
    old, new = fit_affine(T[fit], S[fit], ridge=1e-2), fit_affine(T[fit], S[fit])
    assert rec(new) > rec(old) + 0.2, (rec(old), rec(new))
    assert new.ridge < old.ridge / 10, (old.ridge, new.ridge)
    # low-rank at full rank still equals affine, both with the chosen penalty
    lr = fit_lowrank(T[fit], S[fit], rank=96)
    assert lr.ridge == new.ridge and np.allclose(lr(T[q], normalize=False), new(T[q], normalize=False), atol=1e-3)


def test_adapter_families_script_never_fits_on_eval_queries(tmp):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from adapter_families import run
    from reusemap.runner import DEFAULT_CFG
    make_pair(tmp, N=1500, Q=200, bm25_depth=30)
    cfg = dict(DEFAULT_CFG); cfg.update({"K": 10, "adapter_n_fit": 500, "adapter_n_eval": 300})
    r = run(tmp, cfg, n_fit=500, log=lambda *_: None)
    fams = {(x["family"], x.get("rank"), x["fit_on"]) for x in r["adapters"]}
    assert len(fams) == 15 and {"procrustes", "affine", "lowrank", "mlp"} <= {f for f, _, _ in fams}
    assert all(0.0 <= x["M_A"] <= 1.0 for x in r["adapters"]) and r["M_S"] > 0 and r["M_T"] > 0
    q_fit = [x for x in r["adapters"] if x["fit_on"] == "queries"]
    assert all(x["n_fit"] <= 200 - r["n_eval"] for x in q_fit), "query fit must come from the traffic pool only"


def test_no_contract_sends_an_instruction_twice(tmp):
    """The model sees prompts[prompt_name] + prefix + text. The Qwen3 contracts set the Instruct
    string as a literal prefix AND prompt_name='query', whose prompt is that same string, so every
    Qwen3 query carried the instruction twice. No contract may set both; apply_contract refuses a
    model whose named prompt is non-empty when a prefix is also set; and encode_check fails
    vectors whose sidecar disagrees with the current contract."""
    from reusemap import models
    for mid, c in json.load(open(models.DEFAULT_PATH))["models"].items():
        for kind in ("query", "doc"):
            assert not (models.prefix_for(c, kind) and models.prompt_name_for(c, kind)), (mid, kind)
    c = {"query_prefix": "Instruct: x\nQuery:", "doc_prefix": "", "query_prompt_name": "query", "doc_prompt_name": None}
    assert models.double_instruction(c, "query", {"query": "Instruct: x\nQuery:"})
    assert models.double_instruction(c, "query", {"query": ""}) is None
    assert models.double_instruction(dict(c, query_prefix=""), "query", {"query": "Instruct: x\nQuery:"}) is None
    # a model whose DEFAULT prompt is non-empty doubles a literal prefix even without a prompt_name
    c2 = dict(c, query_prompt_name=None)
    assert models.double_instruction(c2, "query", {"query": "Instruct: x"}, default_prompt_name="query")
    assert models.double_instruction(c2, "query", {"query": "Instruct: x"}) is None
    rec = {"text_prefix": "Instruct: x\nQuery:", "prompt_name": "query"}
    assert models.stale_encoding(rec, dict(c, query_prefix=""), "query"), "a changed prefix must make the vectors stale"
    assert not models.stale_encoding(rec, c, "query")


def test_query_sweep_decides_exactly_as_the_planner(tmp):
    """query_sweep.py recomputes D1/D3 from fewer judged queries; it must apply the planner's
    rule, not a copy of it. On ladders that include K' = K (where G is the whole gap), every
    random case must agree with plan.decide."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    import plan, query_sweep
    rng = np.random.default_rng(5)
    for _ in range(400):
        ms = float(rng.uniform(0.6, 0.9)); gap = float(rng.choice([-0.002, 0.003, 0.006, 0.02, 0.05]))
        G = {10: gap}
        for k in (25, 50, 100, 200, 500):
            G[k] = float(rng.uniform(-0.01, 0.03))
        d = plan.decide("x", ms, ms + gap, G, None, tau=0.005, eps=0.01)
        assert query_sweep.decide(ms, ms + gap, G, 10, 0.005, 0.01) == (d["family"], d["kprime"]), (gap, G, d)


def test_check_planner_scores_only_pre_committed_replays_at_the_planned_depth(tmp):
    import subprocess
    os.makedirs(os.path.join(tmp, "res"))
    def res(name, window, winner, c, debt, G):
        s = {"traffic_ratio": 64.0, "winner": winner, "summaries": {
            "S2/demand_then_rand": {"C_serve_frac": c, "C_serve_basis": "at_last_arrival", "debt": debt}}}
        json.dump({"name": name, "window": window, "primary_setting": {"traffic_ratio": 64.0},
                   "settings": [s], "G": {str(window): G}}, open(os.path.join(tmp, "res", name + ".json"), "w"))
    res("PA", 25, "S2/demand_then_rand", 0.1450, 0.001, 0.002)   # attains its bound: pass
    res("PB", 25, "S2/demand_then_rand", 0.1600, 0.001, 0.002)   # exceeds the bound: fail
    res("PC", 50, "S2/demand_then_rand", 0.2600, 0.001, 0.002)   # wrong depth: not scorable
    with open(os.path.join(tmp, "plan.csv"), "w") as f:
        f.write("pair,family,kprime,cost,verdict\nPA,S2,25,0.1451,match\nPB,S2,25,0.1451,match\nPC,S2,25,0.1451,match\n")
    with open(os.path.join(tmp, "RUNS.tsv"), "w") as f:
        f.write("date\tname\tstatus\n2026-11-08T10:00:00\tPA\tok\n2026-11-08T10:00:00\tPB\tok\n2026-11-08T10:00:00\tPC\tok\n")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "check_planner.py"),
           "--plan", os.path.join(tmp, "plan.csv"), "--results", os.path.join(tmp, "res"),
           "--runs", os.path.join(tmp, "RUNS.tsv"), "--runs-tz", "+08:00"]
    pre = subprocess.run(cmd + ["--committed-at", "2026-11-08T09:00:00+08:00"], capture_output=True, text=True)
    assert pre.returncode == 1 and "PB | S2 | 25 | FAIL | pre-committed" in pre.stdout, pre.stdout
    assert "PA | S2 | 25 | PASS | pre-committed" in pre.stdout and "depth not scorable" in pre.stdout
    post = subprocess.run(cmd + ["--committed-at", "2026-11-08T11:00:00+08:00"], capture_output=True, text=True)
    assert post.returncode == 0 and "post-hoc" in post.stdout, post.stdout


def test_curve_check_lists_every_start_round_that_fits_a_thinned_trajectory(tmp):
    """The runner does not store r0. Above 400 rounds several r0 can give the same thinned length
    (640 rounds stored as 321 points fits r0 = 1 and r0 = 320). Silently taking the smallest one
    put m on the wrong scale; the check must see every candidate and score the worst case."""
    import math
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts",
                            "coverage_curve_check.py"), encoding="utf-8").read()
    ns = {"np": np, "math": math}
    exec(src[src.index("def rounds_candidates"):src.index("sys.path.insert(0")], ns)
    rc = ns["rounds_candidates"]
    if [r0 for r0, _ in rc(321, 640)] != [1, 320]:
        raise AssertionError(f"640 rounds / 321 points: {[r0 for r0, _ in rc(321, 640)]}")
    if [r0 for r0, _ in rc(354, 707)] != [1, 2, 354]:
        raise AssertionError(f"707 rounds / 354 points: {[r0 for r0, _ in rc(354, 707)]}")
    one = rc(120, 300)
    if len(one) != 1 or one[0][0] != 181 or one[0][1][-1] != 300:
        raise AssertionError("a series under 400 rounds has exactly one placement, ending at the last round")


def test_merge_frontier_refuses_a_depth_measured_twice_in_one_file(tmp):
    import subprocess
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    f = os.path.join(tmp, "sweep_P2.json")
    json.dump({"pair": "P2", "depths": [{"kprime": 25, "G": 0.004}, {"kprime": 25, "G": 0.02},
                                        {"kprime": 50, "G": 0.001}]}, open(f, "w"))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "merge_frontier.py"), f,
                        "--out", os.path.join(tmp, "fr.json")], capture_output=True, text=True)
    if r.returncode == 0 or "K'=[25]" not in (r.stderr + r.stdout):
        raise AssertionError(f"kept one of two K'=25 entries silently:\n{r.stdout}{r.stderr}")


def test_coverage_curve_matches_simulated_arrivals_and_never_exceeds_the_independent_bound(tmp):
    from reusemap.coverage_model import (expected_coverage_from_hist, expected_coverage_from_mass,
                                         independent_bound)
    from reusemap.search import topk_ip
    rng = np.random.default_rng(1)
    N, Q, K = 3000, 400, 20
    X = rng.standard_normal((N, 16)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = rng.standard_normal((6, 16))
    Qv = (C[rng.integers(0, 6, Q)] + 0.7 * rng.standard_normal((Q, 16))).astype(np.float32)
    top, _ = topk_ip(Qv, X, K)
    c = np.zeros(N, int); np.add.at(c, top.ravel(), 1)
    v, f = np.unique(c, return_counts=True)
    ms = np.array([30, 300, 3000])
    pred = expected_coverage_from_hist(dict(zip(v.tolist(), f.tolist())), Q, ms, N)
    for m, pr in zip(ms, pred):
        sim = []
        for _ in range(200):
            seen = np.zeros(N, bool); seen[top[rng.integers(0, Q, m)].ravel()] = True; sim.append(seen.mean())
        assert abs(np.mean(sim) - pr) < 4 * np.std(sim) / np.sqrt(200) + 1e-4, (m, np.mean(sim), pr)
    # the bound holds for ANY workload: random Dirichlet popularity over the same windows
    for s in range(5):
        pq = np.random.default_rng(s).dirichlet(np.full(Q, 0.3))
        P = np.zeros(N); np.add.at(P, top.ravel(), np.repeat(pq, K))
        assert abs(P.sum() - K) < 1e-9
        assert np.all(expected_coverage_from_mass(P, ms, N) <= independent_bound(K, N, ms) + 1e-12)


def test_zipf_scan_reproduces_the_weighted_measurement(tmp):
    """zipf_scan.py computes each query's recall once and re-weights; its weighted gap and G must
    equal what the measurement code gives for a Pair built with that workload, at alpha 0.9 and
    popularity order 0 (the replays' order). And depths <= K are refused, not turned into None."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    import zipf_scan
    pdir = os.path.join(tmp, "PZ")
    make_pair(pdir, N=600, Q=120, bm25_depth=20, seed=3)
    cfgf = os.path.join(tmp, "cfg.json")
    c = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "msmarco_1m_v3.json")))
    c.pop("secondary_eval_qids", None); c["window"] = 25
    json.dump(c, open(cfgf, "w"))
    a = zipf_scan.build_parser().parse_args([pdir, "--config", cfgf, "--kprime", "25,50,100", "--alpha", "0,0.9",
                                             "--pop-seeds", "2", "--out-dir", os.path.join(tmp, "zs")])
    out = zipf_scan.scan(a)["PZ"]
    row = next(r for r in out["rows"] if r["alpha"] == 0.9 and r["pop_seed"] == 0)
    from reusemap import runner
    cfg = dict(runner.DEFAULT_CFG); cfg.update(c); cfg.update(kprime=[25, 50, 100], workload="zipf:0.9", replay=False)
    cfg.pop("kprime_frac", None)
    p = Pair(pdir, cfg).precompute()
    ref = reference_qualities(p, cfg["primary"]["metric"], cfg["primary"]["thr"])
    assert abs(row["gap"] - (ref["M_T"] - ref["M_S"])) < 1e-10, (row["gap"], ref)
    g = window_gap(p, [25, 50, 100], ref["M_T"], cfg["primary"]["metric"], cfg["primary"]["thr"])
    for k in (25, 50, 100):
        assert abs(row["G"][k] - g[k]) < 1e-10, (k, row["G"][k], g[k])
    assert len(out["rows"]) == 1 + 2 and out["uniform"]["family"] in ("S0", "S2")
    try:
        zipf_scan.scan(zipf_scan.build_parser().parse_args([pdir, "--config", cfgf, "--kprime", "5,25",
                                                            "--out-dir", os.path.join(tmp, "zs2")]))
        raise AssertionError("--kprime 5 (<= K) was accepted")
    except SystemExit as e:
        assert "> K" in str(e)

if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as tmp:
                fn(tmp)
            print("ok ", name)
            n += 1
    print(f"{n} tests passed")
