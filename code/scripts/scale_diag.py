"""Why does a pair's quality gap move with the corpus size? Per-query evidence, same queries.

    python scripts/scale_diag.py --pairs P2 P3 P9 \
        --size 1M=pairs/{P}:configs/msmarco_1m_v3.json \
        --size 2M=pairs_s2m/{P}f:configs/msmarco_s2m_v3.json \
        --size 8.8M=pairs_full/{P}f:configs/msmarco_full_v3.json --out scale_diag.json

Every corpus holds every judged passage, so a passage that a larger corpus adds is unjudged by
construction, under the dev judgments and under TREC-DL alike. A larger corpus can therefore
lower measured quality in two ways: the added passages are better matches that are not relevant
(a real loss), or relevant passages nobody judged (an artefact of sparse judgments). This script
measures what separates the two, per model, on the same evaluation queries at every size:

  dev Recall@10            the primary metric, per query, so the change in the gap gets a
                           paired bootstrap interval (queries resampled, 2,000 draws)
  TREC-DL nDCG@10          deeply judged queries, raw
  TREC-DL condensed nDCG   the same with every unjudged passage removed from the ranking before
                           the cut (Sakai's condensed lists): it ranks judged passages only, so
                           unjudged added passages cannot lower it
  TREC-DL judged@10        the share of the top 10 that is judged: how many unjudged passages
                           each model pulls into its top 10 as the corpus grows

If the gap shrinks on dev and the raw TREC-DL gap, but not on the condensed one, and the target
pulls in more unjudged passages than the source, the shrinkage is the judgment artefact. If the
condensed gap shrinks too, it is a real change in the models' relative quality.
Exact search over the full corpus, three models: tens of minutes on CPU.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import runner  # noqa: E402
from reusemap.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from reusemap.pair import Pair  # noqa: E402
from reusemap.search import topk_ip  # noqa: E402

DEPTH = 1000   # TREC-DL condensed lists need judged passages from deep in the ranking


def parse_size(s):
    lab, rest = s.split("=", 1)
    pat, cfg = rest.rsplit(":", 1)
    return lab, pat, cfg


def condensed_ndcg(ranked, rels, k):
    judged = [d for d in ranked if int(d) in rels]
    return ndcg_at_k(judged, rels, k)


def boot(delta_fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([delta_fn(rng.integers(0, n, n)) for _ in range(B)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--size", action="append", required=True, help="LABEL=dir_pattern_with_{P}:config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    sizes = [parse_size(s) for s in a.size]
    cache = {}            # realpath of a doc matrix -> (dev top-10 rows, TREC-DL top-DEPTH rows)
    res = {"pairs": {}, "sizes": [lab for lab, _, _ in sizes]}
    for P in a.pairs:
        per = {}
        ev0 = sec0 = None
        for lab, pat, cfgp in sizes:
            d = pat.replace("{P}", P)
            if not os.path.exists(os.path.join(d, "doc_src.npy")):
                print(f"{P} {lab}: {d} missing, skipped"); continue
            cfg = dict(runner.DEFAULT_CFG); cfg.update(json.load(open(cfgp))); cfg["replay"] = False
            p = Pair(d, cfg)
            ev, sec = np.asarray(p.eval_q, np.int64), np.asarray(p.secondary_q, np.int64)
            if ev0 is None:
                ev0, sec0 = ev, sec
            elif not (np.array_equal(ev, ev0) and np.array_equal(sec, sec0)):
                sys.exit(f"{P} {lab}: evaluation queries differ from the first size; not comparable")
            K = cfg["K"]
            out = {"N": int(p.N)}
            for role, docs, qs in (("S", p.doc_src, p.q_src), ("T", p.doc_tgt, p.q_tgt)):
                key = os.path.realpath(docs.filename) if isinstance(docs, np.memmap) else f"{d}:{role}"
                if key not in cache:
                    top_dev, _ = topk_ip(qs[ev], docs, K)
                    top_sec, _ = topk_ip(qs[sec], docs, min(DEPTH, p.N)) if len(sec) else (None, None)
                    cache[key] = (top_dev, top_sec)
                top_dev, top_sec = cache[key]
                rec = np.array([recall_at_k(top_dev[i], p.qrels.get(int(q), {}), K, 1) if p.qrels.get(int(q)) else np.nan
                                for i, q in enumerate(ev)], float)
                m = {"recall_dev": rec}
                if top_sec is not None:
                    raw = np.array([ndcg_at_k(top_sec[i][:K], p.qrels.get(int(q), {}), K) if p.qrels.get(int(q)) else np.nan
                                    for i, q in enumerate(sec)], float)
                    cond = np.array([condensed_ndcg(top_sec[i], p.qrels.get(int(q), {}), K) if p.qrels.get(int(q)) else np.nan
                                     for i, q in enumerate(sec)], float)
                    jud = np.array([np.mean([int(x) in p.qrels.get(int(q), {}) for x in top_sec[i][:K]])
                                    for i, q in enumerate(sec)], float)
                    m.update(ndcg_trec=raw, cndcg_trec=cond, judged10_trec=jud)
                out[role] = m
            per[lab] = out
        if not per:
            continue
        labs = list(per)
        base = labs[0]
        summ = {}
        for lab in labs:
            o = per[lab]
            s = {"N": o["N"]}
            for met in ("recall_dev", "ndcg_trec", "cndcg_trec", "judged10_trec"):
                if met not in o["S"]:
                    continue
                vs, vt = o["S"][met], o["T"][met]
                s[f"S_{met}"] = float(np.nanmean(vs)); s[f"T_{met}"] = float(np.nanmean(vt))
                s[f"gap_{met}"] = s[f"T_{met}"] - s[f"S_{met}"]
                if lab != base:
                    b0s, b0t = per[base]["S"][met], per[base]["T"][met]
                    ok = ~(np.isnan(vs) | np.isnan(vt) | np.isnan(b0s) | np.isnan(b0t))
                    d = (vt - vs - (b0t - b0s))[ok]
                    s[f"dgap_{met}"] = float(d.mean())
                    s[f"dgap_{met}_ci95"] = boot(lambda idx, d=d: d[idx].mean(), len(d), a.boot) if len(d) else None
                    s[f"n_{met}"] = int(ok.sum())
            summ[lab] = s
        res["pairs"][P] = summ
        print(f"\n== {P}")
        for lab, s in summ.items():
            line = f"  {lab:>5s} N={s['N']:>9,d}  dev gap {s['gap_recall_dev']:+.4f}"
            if "dgap_recall_dev" in s:
                lo, hi = s["dgap_recall_dev_ci95"]
                line += f" (change {s['dgap_recall_dev']:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])"
            if "gap_ndcg_trec" in s:
                line += (f"  TREC nDCG gap {s['gap_ndcg_trec']:+.4f}  condensed {s['gap_cndcg_trec']:+.4f}"
                         f"  judged@10 S {s['S_judged10_trec']:.3f} T {s['T_judged10_trec']:.3f}")
            print(line)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
