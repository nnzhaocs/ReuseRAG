"""Window-mass statistics and expected cost-at-service curves for one source encoding.

    python scripts/window_mass.py --docs enc/MiniLM/doc.npy --queries enc/MiniLM/q.npy \
        --exclude-qids sub1m/query_sets/trecdl.txt --kprime 25,50,100,200,500 \
        --zipf 0.6,0.9,1.2 --sample-fracs 0.1,0.25,0.5 --out window_mass/MiniLM.json

One retrieval pass, the same one scripts/coverage.py makes, but it keeps per-document
MULTIPLICITY c(d) (how many servable queries have d in their top-K') instead of a 0/1 mask. From
it: the multiplicity histogram; the exact expected coverage curve under i.i.d. uniform arrivals
(reusemap.coverage_model) on an m-grid and at the replay's per-round arrivals; the independent
bound; the dispersion of window mass; the same curve under Zipf(alpha) popularity over the
queries (a seeded random popularity order, so no query is favoured by index); and, for each
sample fraction, the curve predicted from a random query sample alone against the full one.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap.coverage_model import (expected_coverage_from_hist, expected_coverage_from_mass,  # noqa: E402
                                     independent_bound, mass_dispersion)
from reusemap.pair import load_vectors  # noqa: E402
from reusemap.search import topk_ip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True); ap.add_argument("--queries", required=True)
ap.add_argument("--kprime", default="25,50,100,200,500")
ap.add_argument("--exclude-qids", default=None)
ap.add_argument("--zipf", default="0.6,0.9,1.2")
ap.add_argument("--sample-fracs", default="0.1,0.25,0.5")
ap.add_argument("--rounds", type=int, default=100, help="replay rounds; background budget b = N / rounds")
ap.add_argument("--traffic-ratios", default="4,16,64,256")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--block", type=int, default=512)
ap.add_argument("--out", required=True); ap.add_argument("--name", default=None)
a = ap.parse_args()

ks = sorted({int(k) for k in a.kprime.split(",") if k.strip()})
if not ks or min(ks) <= 0:
    sys.exit("--kprime must list at least one positive depth")
X = load_vectors(a.docs); Qv = load_vectors(a.queries); N = X.shape[0]
if N == 0 or Qv.shape[0] == 0:
    sys.exit(f"empty input: {N} documents, {Qv.shape[0]} queries")
qids = np.arange(Qv.shape[0])
if a.exclude_qids:   # ROW INDICES into q.npy, the same convention as coverage.py --exclude-qids
    drop = np.loadtxt(a.exclude_qids, dtype=np.int64, ndmin=1)
    if drop.size and (drop.min() < 0 or drop.max() >= Qv.shape[0]):
        sys.exit(f"{a.exclude_qids}: ids outside 0..{Qv.shape[0] - 1}; this file must hold row indices "
                 f"into {a.queries}, not dataset query ids")
    qids = np.setdiff1d(qids, drop)
    print(f"excluded {Qv.shape[0] - len(qids)} of {Qv.shape[0]} queries ({a.exclude_qids}); {len(qids)} remain")
Q = len(qids); rng = np.random.default_rng(a.seed)
alphas = [float(x) for x in a.zipf.split(",") if x.strip()]
pop = {al: None for al in alphas}
order = rng.permutation(Q)                         # popularity rank of each query, seeded
for al in alphas:
    w = 1.0 / np.power(np.arange(1, Q + 1), al); w = w / w.sum()
    pq = np.empty(Q); pq[order] = w; pop[al] = pq
fracs = [float(x) for x in a.sample_fracs.split(",") if x.strip()]
samples = {f: np.sort(rng.choice(Q, max(1, int(round(f * Q))), replace=False)) for f in fracs}
in_sample = {f: np.zeros(Q, bool) for f in fracs}
for f in fracs: in_sample[f][samples[f]] = True

cnt = {k: np.zeros(N, np.int32) for k in ks}
mass = {(al, k): np.zeros(N) for al in alphas for k in ks}
scnt = {(f, k): np.zeros(N, np.int32) for f in fracs for k in ks}
t0 = time.time(); kmax = min(max(ks), N)
for i0 in range(0, Q, a.block):
    pos = np.arange(i0, min(i0 + a.block, Q))
    top, _ = topk_ip(Qv[qids[pos]], X, kmax)
    for k in ks:
        w = top[:, :k]
        np.add.at(cnt[k], w.ravel(), 1)
        for al in alphas:
            np.add.at(mass[(al, k)], w.ravel(), np.repeat(pop[al][pos], k))
        for f in fracs:
            sel = in_sample[f][pos]
            if sel.any(): np.add.at(scnt[(f, k)], w[sel].ravel(), 1)
print(f"retrieval pass: {time.time() - t0:.0f}s for {Q} queries")

mgrid = np.unique(np.round(np.geomspace(100, 50 * Q, 60)).astype(int))
b = N / a.rounds
out = {"name": a.name or os.path.basename(os.path.dirname(os.path.abspath(a.docs))), "N": int(N),
       "n_queries": int(Q), "rounds": a.rounds, "seed": a.seed, "seconds": time.time() - t0,
       "m_grid": mgrid.tolist(), "rows": []}
for k in ks:
    c = cnt[k]; vals, freq = np.unique(c, return_counts=True)
    hist = {int(v): int(n) for v, n in zip(vals, freq)}
    P = c / Q
    assert abs(P.sum() - k) < 1e-6 * k, "window mass must sum to K'"
    row = {"kprime": k, "union": float((c > 0).mean()), "hist": hist,
           "dispersion": mass_dispersion(P), "max_multiplicity": int(c.max()),
           "curve_uniform": expected_coverage_from_hist(hist, Q, mgrid, N).tolist(),
           "curve_independent": independent_bound(k, N, mgrid).tolist(),
           "per_round": {}, "zipf": {}, "sample_forecast": {}}
    for tr in [float(x) for x in a.traffic_ratios.split(",")]:
        lam = tr * b / k
        ms = lam * np.arange(1, 8 * a.rounds + 1)
        row["per_round"][str(tr)] = {"arrivals_per_round": lam,
                                     "curve": expected_coverage_from_hist(hist, Q, ms, N).tolist()}
    for al in alphas:
        Pm = mass[(al, k)]
        row["zipf"][str(al)] = {"curve": expected_coverage_from_mass(Pm, mgrid, N).tolist(),
                                "dispersion": mass_dispersion(Pm)}
    full = np.array(row["curve_uniform"])
    for f in fracs:
        s = len(samples[f]); cs = scnt[(f, k)]
        vs, fs = np.unique(cs, return_counts=True)
        pred = expected_coverage_from_hist({int(v): int(n) for v, n in zip(vs, fs)}, s, mgrid, N)
        row["sample_forecast"][str(f)] = {"n": s, "curve": pred.tolist(),
                                          "max_abs_err": float(np.max(np.abs(pred - full))),
                                          "union_from_sample": float((cs > 0).mean())}
    out["rows"].append(row)
    d = row["dispersion"]
    print(f"K'={k:5d} union {row['union']:.4f}  max c {row['max_multiplicity']:5d}  mass CV {d['cv']:.2f}  "
          f"top-1% docs hold {100 * d['mass_in_top1pct_docs']:.1f}% of mass")
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
with open(a.out, "w") as fh:
    json.dump(out, fh)
print("wrote", a.out)
