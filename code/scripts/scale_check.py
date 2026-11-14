"""Check PREREG_SCALE.md's predictions: the planner's static inputs at 1M against the full corpus.

    python scripts/scale_check.py --one-m sweeps/sweep_P2.json sweeps_fine/sweep_P2.json ... \
        --full sweeps_full/sweep_P2f.json sweeps_full/sweep_P3f.json sweeps_full/sweep_P9f.json \
        --mass window_mass_full --md SCALE_CHECK.md

A full-corpus pair <P>f is compared with the 1M pair <P>; the 1M depths may be spread over
several sweep files (sweeps/ and sweeps_fine/), which are merged (a depth present twice must
agree). D3 is plan.decide's, with depths <= K dropped as in query_sweep.py, and no coverage
rows (D2 is not under test here).

  S1  gap(full) >= gap(1M), and D1 decides the same
  S2  G(full) >= G(1M) at K' = 25 and 50, so D3's depth at full >= its 1M depth
  S3  distinct documents in the servable windows: 1M count <= full count <= m K'
  S4  cov(K') <= 1 - (1 - K'/N)^m at every K'

No tolerance: "not smaller" is checked as >=. The script reports; it does not decide what the
paper says.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan as _plan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--one-m", nargs="+", required=True)
ap.add_argument("--full", nargs="+", required=True)
ap.add_argument("--mass", default=None, help="window_mass_full/ (optional; reported, not tested)")
ap.add_argument("--tau", type=float, default=0.005)
ap.add_argument("--eps", type=float, default=0.01)
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--md", default=None)
a = ap.parse_args()


def merged(files):
    by = {}
    for f in files:
        d = json.load(open(f)); p = d["pair"]
        e = by.setdefault(p, {"M_S": d["M_S"], "M_T": d["M_T"], "N": d["N"], "depths": {}, "n_servable": d.get("n_servable")})
        if abs(e["M_S"] - d["M_S"]) > 1e-9 or abs(e["M_T"] - d["M_T"]) > 1e-9:
            sys.exit(f"{f}: M_S/M_T differ from another sweep of {p}: not the same pair")
        for r in d["depths"]:
            k = int(r["kprime"]); old = e["depths"].get(k)
            if old and abs(old["G"] - r["G"]) > 1e-9:
                sys.exit(f"{f}: G({k}) = {r['G']} but another sweep of {p} has {old['G']}")
            e["depths"][k] = r
    return by


def d3(e):
    G = {k: r["G"] for k, r in e["depths"].items() if k > a.K}
    return _plan.decide("x", e["M_S"], e["M_T"], G, None, a.tau, a.eps)


one, full = merged(a.one_m), merged(a.full)
rows, verdicts, out = [], [], {}
for pf in sorted(full):
    p = pf[:-1] if pf.endswith("f") else pf
    if p not in one:
        sys.exit(f"no 1M sweep of {p} for {pf}")
    o, f = one[p], full[pf]
    go, gf = o["M_T"] - o["M_S"], f["M_T"] - f["M_S"]
    do, df = d3(o), d3(f)
    s1 = gf >= go and (do["family"] == "S0") == (df["family"] == "S0")
    s2_k = {k: (o["depths"][k]["G"], f["depths"][k]["G"]) for k in (25, 50) if k in o["depths"] and k in f["depths"]}
    ko = int(do["kprime"]) if do.get("kprime") not in (None, "", "-") else None
    kf = int(df["kprime"]) if df.get("kprime") not in (None, "", "-") else None
    s2 = all(gfk >= gok for gok, gfk in s2_k.values()) and (ko is None or kf is None or kf >= ko)
    m = f.get("n_servable") or o.get("n_servable")
    s3_k, s4_k = {}, {}
    for k in sorted(set(o["depths"]) & set(f["depths"])):
        co, cf = o["depths"][k]["covered"], f["depths"][k]["covered"]
        s3_k[k] = (co, cf, m * k if m else None)
    for k, r in sorted(f["depths"].items()):
        bound = 1 - (1 - k / f["N"]) ** m if m else None
        s4_k[k] = (r["coverage"], bound)
    s3 = all(co <= cf and (mk is None or cf <= mk) for co, cf, mk in s3_k.values())
    s4 = all(b is None or c <= b + 1e-12 for c, b in s4_k.values())
    out[pf] = {"N_1m": o["N"], "N_full": f["N"], "gap_1m": go, "gap_full": gf, "d3_1m": do, "d3_full": df,
               "G_25_50": {str(k): v for k, v in s2_k.items()},
               "covered": {str(k): v for k, v in s3_k.items()}, "cov_vs_bound": {str(k): v for k, v in s4_k.items()},
               "S1": bool(s1), "S2": bool(s2), "S3": bool(s3), "S4": bool(s4)}
    verdicts += [("S1", pf, s1), ("S2", pf, s2), ("S3", pf, s3), ("S4", pf, s4)]
    g = lambda k: " / ".join(f"{x:+.4f}" for x in s2_k[k]) if k in s2_k else "-"
    rows.append(f"| {pf} | {o['N']:,} -> {f['N']:,} | {go:+.4f} -> {gf:+.4f} | {g(25)} | {g(50)} | "
                f"{do['family']}@{do.get('kprime') or '-'} -> {df['family']}@{df.get('kprime') or '-'} | "
                f"{s3_k.get(50, (0, 0, 0))[0]:,} -> {s3_k.get(50, (0, 0, 0))[1]:,} | "
                f"{'pass' if s1 else 'FAIL'} {'pass' if s2 else 'FAIL'} {'pass' if s3 else 'FAIL'} {'pass' if s4 else 'FAIL'} |")

mass = {}
if a.mass and os.path.isdir(a.mass):
    for fn in sorted(os.listdir(a.mass)):
        if fn.endswith(".json"):
            mass[fn[:-5]] = json.load(open(os.path.join(a.mass, fn)))

hdr = ("| pair | N | gap | G(25) 1M / full | G(50) 1M / full | D3 | covered docs at K'=50 | S1 S2 S3 S4 |\n"
       "|---|---|---|---|---|---|---|---|\n")
n_pass = sum(v for _, _, v in verdicts)
summary = (f"{n_pass} of {len(verdicts)} pre-registered checks pass; failures: "
           + (", ".join(f"{s} on {p}" for s, p, v in verdicts if not v) or "none"))
print(hdr + "\n".join(rows) + "\n\n" + summary)
if a.md:
    with open(a.md, "w") as fh:
        fh.write("# Scale check: PREREG_SCALE.md's predictions at the full corpus\n\n" + hdr + "\n".join(rows)
                 + "\n\n" + summary + "\n\n```json\n" + json.dumps({"pairs": out, "mass_sources": sorted(mass)},
                                                                       indent=1, default=str) + "\n```\n")
