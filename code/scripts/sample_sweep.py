"""How small can the planner's target-encoded sample be? (RUNBOOK 11.6 / 13.6)

    python scripts/sample_sweep.py pairs/P2 --config configs/msmarco_1m_v3.json \
        --results results --fracs 0.01,0.05,0.25 --seeds 3 --out-dir sample_sweep

D1 decides from the quality gap M_T - M_S. In deployment only a SAMPLE of the corpus is encoded
with the target model, so the planner sees the gap on a sub-corpus: every judged document (so
Recall@10 stays defined) plus a uniform random fraction f of the rest. Both models are scored on
the SAME sub-corpus, so the comparison is like for like; both are optimistic, since a sub-corpus
has fewer distractors. For each f and seed this reports the sub-corpus gap, the target encodings
it costs (|S|/N), and whether D1 (gap > tau) decides as it does on the full corpus (taken from
the replay result). The answer the paper needs is the smallest f at which D1 agrees on every seed.

D3's input G(K') is NOT estimated here: a window of K' documents in N has no counterpart in a 1%
sub-corpus, so G needs the frontier sweep over the full target encoding. That is reported as a
limit of the planner's sampling claim, not hidden.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap.metrics import weighted_quality  # noqa: E402
from reusemap.pair import Pair  # noqa: E402
from reusemap.runner import DEFAULT_CFG  # noqa: E402
from reusemap.search import topk_ip  # noqa: E402


def sweep(pair_dir, cfg, full_gap, fracs, seeds, log=print):
    K, tau = cfg["K"], cfg.get("tau", {}).get("debt", 0.005) if isinstance(cfg.get("tau"), dict) else 0.005
    p = Pair(pair_dir, cfg)
    metric, thr = cfg["primary"]["metric"], cfg["primary"].get("thr", 1)
    judged = np.array(sorted({int(d) for q in p.eval_q for d in p.qrels.get(int(q), {})}), np.int64)
    rest = np.setdiff1d(np.arange(p.N), judged)
    qs, qt = p.q_src[p.eval_q], p.q_tgt[p.eval_q]
    full_d1 = full_gap > tau
    rows = []
    for f in fracs:
        for s in range(seeds):
            rng = np.random.default_rng((cfg.get("seed", 0), s, int(round(f * 1e6))))
            S = np.sort(np.concatenate([judged, rng.choice(rest, int(round(f * len(rest))), replace=False)]))
            t0 = time.time()
            ts, _ = topk_ip(qs, np.asarray(p.doc_src[S]), K)
            tt, _ = topk_ip(qt, np.asarray(p.doc_tgt[S]), K)
            ms = {int(q): S[ts[i]] for i, q in enumerate(p.eval_q)}
            mt = {int(q): S[tt[i]] for i, q in enumerate(p.eval_q)}
            MS = weighted_quality(lambda qi: ms[int(qi)], p.eval_q, p.qrels, p.weights, K, metric, thr)
            MT = weighted_quality(lambda qi: mt[int(qi)], p.eval_q, p.qrels, p.weights, K, metric, thr)
            g = MT - MS
            rows.append({"frac": f, "seed": s, "n_docs": int(len(S)), "target_encodings_frac": len(S) / p.N,
                         "M_S": MS, "M_T": MT, "gap": g, "d1_migrate": bool(g > tau),
                         "agrees": bool((g > tau) == full_d1), "seconds": time.time() - t0})
            log(f"[{os.path.basename(os.path.normpath(pair_dir))}] f={f:g} seed={s}: |S|/N={len(S) / p.N:.4f} "
                f"gap={g:+.4f} (full {full_gap:+.4f}) D1 {'agrees' if rows[-1]['agrees'] else 'DIFFERS'}")
    ok = [f for f in fracs if all(r["agrees"] for r in rows if r["frac"] == f)]
    return {"pair": os.path.basename(os.path.normpath(pair_dir)), "N": int(p.N), "tau": tau,
            "n_judged": int(len(judged)), "full_gap": full_gap, "full_d1_migrate": bool(full_d1),
            "rows": rows, "smallest_frac_all_seeds_agree": min(ok) if ok else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True, help="directory of replay results (full-corpus gap)")
    ap.add_argument("--fracs", default="0.01,0.05,0.25")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if not os.path.isdir(a.results):
        sys.exit(f"--results must be a directory of replay results (got {a.results})")
    cfg = dict(DEFAULT_CFG); cfg.update(json.load(open(a.config)))
    os.makedirs(a.out_dir, exist_ok=True)
    for pd in a.pairs:
        name = os.path.basename(os.path.normpath(pd))
        R = json.load(open(os.path.join(a.results, name + ".json")))
        gap = R["reference"]["M_T"] - R["reference"]["M_S"]
        out = sweep(pd, cfg, gap, [float(x) for x in a.fracs.split(",")], a.seeds)
        json.dump(out, open(os.path.join(a.out_dir, name + ".json"), "w"), indent=1)
        print(f"{name}: smallest fraction at which D1 agrees on every seed: {out['smallest_frac_all_seeds_agree']}")
