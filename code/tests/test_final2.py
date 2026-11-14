"""Tests for the 2026-11-11 scripts (union_price, zipf2, replay_prereg, scale_subset, scale_diag).

    python tests/test_final2.py        # prints "N final2 tests passed"

Each test builds a synthetic pair (scripts/make_synthetic.py), so nothing here reads real data.
The central checks are invariants the final2 analysis relies on:
  * static candidate-union coverage contains the inline repairs seen by the delayed
    demand-then-rand reference replay; it is not assumed equal to first-arrival C_serve;
  * the static gap equals the rebuild's replay debt, under uniform and Zipf weights;
  * the nested corpora keep every judged passage, nest, and carry the exact full-corpus vectors.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_synthetic  # noqa: E402
import union_price  # noqa: E402

PY = sys.executable


def cfg_file(d, **kw):
    c = json.load(open(os.path.join(ROOT, "configs", "msmarco_1m_v3.json")))
    c = {k: v for k, v in c.items() if not k.startswith("_")}
    c.update(secondary_eval_qids=None, window=25, bm25_depth=30, traffic_ratio=[64.0],
             orders=["rand", "demand_then_rand"], probe_queries=4, strat_per_query=50, kprime=[25, 50])
    c.pop("kprime_frac", None)
    c.update(kw)
    path = os.path.join(d, f"cfg_{abs(hash(json.dumps(kw, sort_keys=True))) % 10 ** 8}.json")
    json.dump(c, open(path, "w"))
    return path


def run(args, cwd):
    r = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def replay(d, pair, cfg, out):
    run([os.path.join(ROOT, "scripts", "run_pair.py"), pair, "--config", cfg, "--name",
         os.path.basename(pair), "--out", out, "--allow-dirty"], d)


def test_static_winner_rule():
    sw = union_price.static_winner
    # inside the band the cheapest wins; S0 has no inline cost
    w, adm = sw({"S2": {"G": 0.001, "cov": 0.26}, "S4": {"G": -0.002, "cov": 0.85}}, 0.03, -0.1, 0.005, 0.01)
    assert w == "S2" and "S0" not in adm, (w, adm)
    # a union that beats the window by more than tau wins despite its cost
    w, _ = sw({"S2": {"G": 0.004, "cov": 0.26}, "S4": {"G": -0.002, "cov": 0.85}}, 0.03, -0.1, 0.005, 0.01)
    assert w == "S4", w
    # a gap inside the band: the rebuild wins (D1)
    w, _ = sw({"S2": {"G": 0.0, "cov": 0.26}}, 0.003, -0.1, 0.005, 0.01)
    assert w == "S0", w
    # nothing reaches target: no winner
    w, adm = sw({"S2": {"G": 0.02, "cov": 0.26}}, 0.05, -0.1, 0.005, 0.01)
    assert w is None and adm == [], (w, adm)
    # a bridge below the old model is never admissible, whatever its debt
    w, adm = sw({"S2": {"G": 0.02, "cov": 0.26}}, 0.009, -0.001, 0.005, 0.01)
    assert "S1" not in adm and w == "S0", (w, adm)


def test_union_price_bounds_replay_repairs(tmp):
    pair = os.path.join(tmp, "pairs", "X")
    make_synthetic.make_pair(pair, N=2500, Q=400, seed=5, bm25_depth=40)
    cfg = cfg_file(tmp)
    replay(tmp, pair, cfg, os.path.join(tmp, "res", "X.json"))
    a = union_price.build_parser().parse_args([pair, "--config", cfg, "--kprime", "25,50", "--workloads",
                                               "uniform,zipf:0.6", "--results", f"uniform={tmp}/res",
                                               "--out-dir", os.path.join(tmp, "up")])
    o = union_price.run(a)["X"]["workloads"]
    r = o["uniform"]["replay"]
    assert abs(r["replay_S0_debt"] - o["uniform"]["gap"]) < 1e-12, (r["replay_S0_debt"], o["uniform"]["gap"])
    for f, c in r["families"].items():
        # Static coverage unions candidates for *all* servable queries. At the first-arrival
        # endpoint the delayed reference replay may not have seen every traffic query yet, so its
        # inline repairs (and, while background is delayed, C_serve) can be smaller but not larger.
        assert c["replay_repairs_frac"] <= c["static_candidate_union_cov"] + 1e-12, (f, c)
        if c["replay_basis"] == "at_last_arrival":
            assert c["replay_C_serve_frac"] <= c["static_candidate_union_cov"] + 1e-12, (f, c)
    assert o["zipf:0.6"]["n_eff"] < o["uniform"]["n_eff"]
    # depths <= K are refused
    try:
        union_price.run(union_price.build_parser().parse_args([pair, "--config", cfg, "--kprime", "5,25",
                                                               "--out-dir", os.path.join(tmp, "up2")]))
        raise AssertionError("--kprime 5 accepted")
    except SystemExit as e:
        assert "> K" in str(e)


def test_zipf2_identity_and_scores(tmp):
    pair = os.path.join(tmp, "pairs", "Z")
    make_synthetic.make_pair(pair, N=2000, Q=400, seed=7, bm25_depth=40)
    cfg = cfg_file(tmp, window=50)
    cfgz = cfg_file(tmp, window=50, workload="zipf:0.6")
    run([os.path.join(ROOT, "scripts", "zipf_scan.py"), pair, "--config", cfg, "--kprime", "25,50,100",
         "--alpha", "0,0.6", "--pop-seeds", "2", "--out-dir", "zs"], tmp)
    run([os.path.join(ROOT, "scripts", "union_price.py"), pair, "--config", cfg, "--kprime", "25,50,100",
         "--workloads", "uniform,zipf:0.6", "--out-dir", "upz"], tmp)
    run([os.path.join(ROOT, "scripts", "zipf2.py"), "prereg", "--scan", "zs", "--price", "upz", "--plan", "0.6=Z",
         "--window", "50", "--md", "PRE.md"], tmp)
    replay(tmp, pair, cfgz, os.path.join(tmp, "rz0.6", "Z.json"))
    out = run([os.path.join(ROOT, "scripts", "zipf2.py"), "score", "--prereg", "PRE.md",
               "--results-pattern", "rz{A}", "--md", "SC.md"], tmp)
    assert "identity holds on 1" in out, out


def test_replay_prereg(tmp):
    pair = os.path.join(tmp, "pairs", "R")
    make_synthetic.make_pair(pair, N=2000, Q=400, seed=9, bm25_depth=40)
    cfg = cfg_file(tmp, window=50)
    run([os.path.join(ROOT, "scripts", "sweep.py"), pair, "--config", cfg, "--kprime", "25,50,100",
         "--out-dir", "sw", "--allow-dirty"] if "--allow-dirty" in open(os.path.join(ROOT, "scripts", "sweep.py")).read()
        else [os.path.join(ROOT, "scripts", "sweep.py"), pair, "--config", cfg, "--kprime", "25,50,100", "--out-dir", "sw"], tmp)
    run([os.path.join(ROOT, "scripts", "replay_prereg.py"), "prereg", "--sweeps", "sw/sweep_R.json", "--window", "50",
         "--label", "t", "--md", "PR.md"], tmp)
    replay(tmp, pair, cfg, os.path.join(tmp, "rr", "R.json"))
    out = run([os.path.join(ROOT, "scripts", "replay_prereg.py"), "score", "--prereg", "PR.md", "--results", "rr",
               "--md", "RS.md"], tmp)
    assert "1 of 1 replays" in out, out
    rows = json.loads(open(os.path.join(tmp, "RS.md")).read().split("```json\n")[1].split("```")[0])["rows"]
    r = rows[0]
    if r["s2_basis"] == "at_last_arrival":
        assert r["cost_ok"], r
    # A replay from the wrong corpus size must never be scored against this prediction.
    rf = os.path.join(tmp, "rr", "R.json")
    bad = json.load(open(rf)); bad["N"] += 1; json.dump(bad, open(rf, "w"))
    q = subprocess.run([PY, os.path.join(ROOT, "scripts", "replay_prereg.py"), "score",
                        "--prereg", "PR.md", "--results", "rr", "--md", "BAD.md"], cwd=tmp,
                       capture_output=True, text=True)
    assert q.returncode != 0 and "predicted from a sweep at N=" in (q.stdout + q.stderr), (q.stdout, q.stderr)


def test_scale_subset_and_diag(tmp):
    # a "full" corpus of 3,000 rows and a "1M" corpus of every judged row plus a fill, in order
    big = os.path.join(tmp, "pairs", "B")
    make_synthetic.make_pair(big, N=3000, Q=300, seed=11, bm25_depth=20)
    qrels = np.loadtxt(os.path.join(big, "qrels.tsv"), dtype=np.int64, ndmin=2)
    judged = np.unique(qrels[:, 1])
    rng = np.random.default_rng(0)
    rest = np.setdiff1d(np.arange(3000), judged)
    small_rows = np.sort(np.concatenate([judged, rng.choice(rest, 1000 - len(judged), replace=False)]))
    for name, rows in (("full", np.arange(3000)), ("one", small_rows)):
        d = os.path.join(tmp, name); os.makedirs(os.path.join(d, "query_sets"), exist_ok=True)
        with open(os.path.join(d, "corpus.tsv"), "w") as f:
            for i, r in enumerate(rows):
                f.write(f"{i}\tpid{r}\ttext {r}\n")
        pos = {int(r): i for i, r in enumerate(rows)}
        with open(os.path.join(d, "qrels.tsv"), "w") as f:
            for q, doc, rel in qrels:
                f.write(f"{q}\t{pos[int(doc)]}\t{rel}\n")
        with open(os.path.join(d, "queries.tsv"), "w") as f:
            for q in range(300):
                f.write(f"{q}\tq{q}\tdev\tquery {q}\n")
        np.savetxt(os.path.join(d, "query_sets", "trecdl.txt"), np.array([0, 1, 2]), fmt="%d")
    enc = os.path.join(tmp, "enc")
    for m, fn in (("src", "doc_src.npy"), ("tgt", "doc_tgt.npy")):
        os.makedirs(os.path.join(enc, m), exist_ok=True)
        np.save(os.path.join(enc, m, "doc.npy"), np.load(os.path.join(big, fn)))
        np.save(os.path.join(enc, m, "q.npy"), np.load(os.path.join(big, "q_" + fn.split("_")[1])))
    # An explicitly empty size list is a supported no-op (RUN_FINAL2.md documents SIZES=).
    out = run([os.path.join(ROOT, "scripts", "scale_subset.py"), "--full", "does-not-need-to-exist",
               "--small", "does-not-need-to-exist", "--enc", "does-not-need-to-exist",
               "--models", "src", "--sizes", "", "--out-root", tmp], tmp)
    assert "no intermediate sizes requested" in out

    run([os.path.join(ROOT, "scripts", "scale_subset.py"), "--full", "full", "--small", "one", "--enc", enc,
         "--models", "src", "tgt", "--sizes", "1500,2200", "--out-root", tmp, "--tag", "s"], tmp)
    r15, r22 = np.load(os.path.join(tmp, "s0.0015m", "rows.npy")), np.load(os.path.join(tmp, "s0.0022m", "rows.npy"))
    assert len(r15) == 1500 and len(r22) == 2200 and np.isin(r15, r22).all() and np.isin(small_rows, r15).all()
    X = np.load(os.path.join(enc, "tgt", "doc.npy"))
    assert np.array_equal(X[r22], np.load(os.path.join(tmp, "enc_s0.0022m", "tgt", "doc.npy")))
    # qrels point at the same passages
    full_pid = [l.split("\t")[1] for l in open(os.path.join(tmp, "full", "corpus.tsv"))]
    sub_pid = [l.split("\t")[1] for l in open(os.path.join(tmp, "s0.0022m", "corpus.tsv"))]
    fq = np.loadtxt(os.path.join(tmp, "full", "qrels.tsv"), dtype=np.int64, ndmin=2)
    sq = np.loadtxt(os.path.join(tmp, "s0.0022m", "qrels.tsv"), dtype=np.int64, ndmin=2)
    assert all(full_pid[a] == sub_pid[b] for a, b in zip(fq[:, 1], sq[:, 1]))
    # a size beyond the full corpus is refused
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "scale_subset.py"), "--full", "full", "--small", "one",
                        "--enc", enc, "--models", "src", "--sizes", "5000", "--out-root", tmp], cwd=tmp,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "must lie between" in (r.stderr + r.stdout)
    # assemble pairs at two sizes and diagnose: the gap at each size equals union_price's
    for lab in ("s0.0015m", "s0.0022m"):
        run([os.path.join(ROOT, "scripts", "assemble_pair.py"), "--src", os.path.join(tmp, f"enc_{lab}", "src"),
             "--tgt", os.path.join(tmp, f"enc_{lab}", "tgt"), "--data", lab, "--out", os.path.join("pa", lab, "Bf")], tmp)
    c = json.load(open(cfg_file(tmp, window=25)))
    for lab in ("s0.0015m", "s0.0022m"):
        c2 = dict(c); c2["secondary_eval_qids"] = os.path.join(tmp, lab, "query_sets", "trecdl.txt")
        json.dump(c2, open(os.path.join(tmp, f"cfg_{lab}.json"), "w"))
    out = run([os.path.join(ROOT, "scripts", "scale_diag.py"), "--pairs", "B",
               "--size", f"a=pa/s0.0015m/{{P}}f:cfg_s0.0015m.json", "--size", f"b=pa/s0.0022m/{{P}}f:cfg_s0.0022m.json",
               "--out", "sd.json", "--boot", "50"], tmp)
    sd = json.load(open(os.path.join(tmp, "sd.json")))["pairs"]["B"]
    up = union_price.run(union_price.build_parser().parse_args(
        [os.path.join(tmp, "pa", "s0.0022m", "Bf"), "--config", os.path.join(tmp, "cfg_s0.0022m.json"),
         "--kprime", "25", "--families", "S2", "--out-dir", os.path.join(tmp, "upd")]))["Bf"]["workloads"]["uniform"]
    assert abs(sd["b"]["gap_recall_dev"] - up["gap"]) < 1e-12, (sd["b"]["gap_recall_dev"], up["gap"])
    lo, hi = sd["b"]["dgap_recall_dev_ci95"]
    assert lo <= sd["b"]["dgap_recall_dev"] <= hi


TESTS = [test_static_winner_rule, test_union_price_bounds_replay_repairs, test_zipf2_identity_and_scores,
         test_replay_prereg, test_scale_subset_and_diag]

if __name__ == "__main__":
    n = 0
    for t in TESTS:
        d = tempfile.mkdtemp()
        try:
            if t.__code__.co_argcount:
                t(d)
            else:
                t()
            n += 1
            print(f"ok   {t.__name__}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"{n} final2 tests passed")
