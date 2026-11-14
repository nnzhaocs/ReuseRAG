"""Bridged retrieval quality for every adapter family Drift-Adapter evaluates, per pair.

    python scripts/adapter_families.py pairs/P2 pairs/P3 ... --config configs/msmarco_1m_v3.json \
        --out-dir adapters

For each pair: fit Procrustes, ridge affine, low-rank affine (rank 64, 128) and a residual MLP,
each on document pairs, on traffic-pool query pairs, and on both; map the target-model EVAL
queries into the source space; retrieve the top-K from the source index; and report M_A, the
relative recall M_A / M_T (the quantity Drift-Adapter reports as 95-99%), and M_A - M_S (the
comparison against not migrating). Eval queries are never used for fitting.

Cost: one shallow retrieval for M_S / M_T (window = K, tgt_depth = K, no ladder), then one
top-K retrieval of the 3,490 eval queries per fitted adapter. No replay, no encoding.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap.compat import FITTERS, relative_residual  # noqa: E402
from reusemap.metrics import weighted_quality  # noqa: E402
from reusemap.pair import Pair  # noqa: E402
from reusemap.provenance import data_fingerprint, git_version  # noqa: E402
from reusemap.runner import DEFAULT_CFG  # noqa: E402
from reusemap.search import topk_ip  # noqa: E402

FAMILIES = [("procrustes", {}), ("affine", {}), ("lowrank", {"rank": 64}), ("lowrank", {"rank": 128}),
            ("mlp", {})]


def run(pair_dir, cfg, fits=("docs", "queries", "both"), n_fit=10000, families=FAMILIES, log=print):
    c = dict(cfg)
    K = c["K"]
    # shallow everything: this script needs M_S, M_T and the fitted maps, nothing deeper
    c.update({"kprime": [K], "window": K, "tgt_depth": K, "probe_c": 1, "replay": False})
    c.pop("kprime_frac", None)
    t0 = time.time()
    p = Pair(pair_dir, c).precompute()
    metric, thr = c["primary"]["metric"], c["primary"].get("thr", 1)
    q = lambda f: weighted_quality(f, p.eval_q, p.qrels, p.weights, K, metric, thr)
    MS, MT = q(lambda qi: p.src_top[qi, :K]), q(lambda qi: p.tgt_top[qi, :K])
    out = {"pair": os.path.basename(os.path.normpath(pair_dir)), "N": int(p.N), "K": K,
           "n_eval": int(len(p.eval_q)), "M_S": MS, "M_T": MT, "precompute_s": time.time() - t0,
           "kappa": p.compat["kappa"], "adapters": []}
    rng = np.random.default_rng(c.get("seed", 0) + 7)
    dsel = rng.choice(p.N, min(n_fit, p.N), replace=False)
    pool = p.traffic_pool
    qsel = rng.choice(pool, min(n_fit, len(pool)), replace=False)
    assert not np.intersect1d(qsel, p.eval_q).size, "eval queries leaked into adapter fitting"
    ev = rng.choice(p.N, min(5000, p.N), replace=False)
    for fit_on in fits:
        Xs, Xt = [], []
        if fit_on in ("docs", "both"):
            Xs.append(np.asarray(p.doc_src[dsel])); Xt.append(np.asarray(p.doc_tgt[dsel]))
        if fit_on in ("queries", "both"):
            Xs.append(p.q_src[qsel]); Xt.append(p.q_tgt[qsel])
        Xs, Xt = np.vstack(Xs).astype(np.float32), np.vstack(Xt).astype(np.float32)
        for kind, kw in families:
            t = time.time()
            ad = FITTERS[kind](Xt, Xs, **kw)
            fit_s = time.time() - t
            qm = ad(p.q_tgt[p.eval_q])
            top, _ = topk_ip(qm, p.doc_src, K)
            rows = {int(qi): top[i] for i, qi in enumerate(p.eval_q)}
            MA = q(lambda qi: rows[int(qi)])
            rec = {"family": kind, **kw, "fit_on": fit_on, "n_fit": int(len(Xs)), "M_A": MA,
                   "rel_recall": MA / MT if MT else None, "vs_source": MA - MS,
                   "doc_residual": relative_residual(ad, np.asarray(p.doc_tgt[ev]), np.asarray(p.doc_src[ev])),
                   "fit_s": fit_s}
            for k_ in ("val_cos", "lr"):
                if hasattr(ad, k_):
                    rec[k_] = getattr(ad, k_)
            if getattr(ad, "ridge", None) is not None:
                rec["ridge"] = ad.ridge   # the penalty chosen on held-out pairs (affine, low-rank)
            out["adapters"].append(rec)
            log(f"[{out['pair']}] {kind}{kw or ''} fit={fit_on}: M_A={MA:.4f} "
                f"({(f'{100 * MA / MT:.1f}%' if MT else 'n/a')} of M_T, {MA - MS:+.4f} vs source)")
    best = max(out["adapters"], key=lambda r: r["M_A"])
    out["best"] = {k: best[k] for k in ("family", "fit_on", "M_A", "rel_recall", "vs_source")}
    out["any_beats_source"] = bool(best["M_A"] > MS)
    # affine/low-rank penalty rule this file was produced under; make_numbers refuses affine
    # columns from files without it (the first run used a fixed 1e-2 * n penalty that collapsed)
    out["affine_ridge"] = "held-out"
    out["git"] = git_version(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out["data_fingerprint"] = data_fingerprint(pair_dir)   # the exact input vectors (see sweep.py)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-fit", type=int, default=10000)
    ap.add_argument("--fits", default="docs,queries,both")
    a = ap.parse_args()
    cfg = dict(DEFAULT_CFG); cfg.update(json.load(open(a.config)))
    os.makedirs(a.out_dir, exist_ok=True)
    for pd in a.pairs:
        r = run(pd, cfg, tuple(a.fits.split(",")), a.n_fit)
        path = os.path.join(a.out_dir, f"adapters_{r['pair']}.json")
        json.dump(r, open(path, "w"), indent=1)
        print(f"wrote {path}: best {r['best']['family']}/{r['best']['fit_on']} M_A={r['best']['M_A']:.4f}, "
              f"M_S={r['M_S']:.4f}, beats source: {r['any_beats_source']}")
