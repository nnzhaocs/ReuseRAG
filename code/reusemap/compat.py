"""Space compatibility between source and target encoders: adapters, residual, CKA.

The adapter maps TARGET-space vectors into the SOURCE space (as in space mapping:
new-model queries are mapped so the old index keeps serving).

Adapter families: semi-orthogonal Procrustes, ridge affine, low-rank affine and a compact
residual MLP. The first, third and fourth are the three families Drift-Adapter (EMNLP 2025)
evaluates. scripts/adapter_families.py measures bridged quality for every family; the replayed
S1/S3 arms use Procrustes on 10,000 document pairs unless the config says otherwise.
"""
import numpy as np
from .search import normalize_rows


class Adapter:
    def __init__(self, W, b):
        # checked at construction, not at the first call: a wrong-shaped bias broadcasts in
        # __call__ and produces an array of the wrong width rather than an error
        if getattr(W, "ndim", 0) != 2 or getattr(b, "shape", None) != (W.shape[1],):
            raise ValueError(f"Adapter takes W of shape (d_target, d_source) and b of shape "
                             f"(d_source,); got {getattr(W, 'shape', None)} and "
                             f"{getattr(b, 'shape', None)}")
        self.W, self.b = W, b

    def __call__(self, Xt, normalize=True):
        if Xt.ndim != 2 or Xt.shape[1] != self.W.shape[0]:
            raise ValueError(f"adapter expects (n, {self.W.shape[0]}) target-space rows, "
                             f"got {Xt.shape}. Source and target encodings are probably swapped: "
                             f"this adapter maps TARGET vectors into the SOURCE space.")
        Y = Xt @ self.W + self.b
        return normalize_rows(Y) if normalize else Y


def fit_procrustes(Xt, Xs):
    """Semi-orthogonal W (dt x ds) minimising ||Xt W - Xs||_F; works for unequal dims.

    Deliberately NOT mean-centred, unlike fit_affine. The adapter's output is compared with
    source embeddings in cosine space, so it has to approximate Xs in absolute terms including
    the mean direction, not just match its shape about the centroid. Centring here would fit a
    map that is right up to a translation the retrieval never applies.
    """
    U, _, Vt = np.linalg.svd(Xt.T @ Xs, full_matrices=False)
    W = (U @ Vt).astype(np.float32)
    return Adapter(W, np.zeros(Xs.shape[1], np.float32))


RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _ridge_solve(A, B, lam):
    d = A.shape[1]
    # eye in A's dtype: float64 would promote the whole solve, and at Qwen3's 4096 dims that is
    # a 134 MB temporary allocated on every fit
    return np.linalg.solve(A.T @ A + lam * np.eye(d, dtype=A.dtype), A.T @ B).astype(np.float32)


def _mean_cos(P, Y):
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return float((P * Y).sum(1).mean())


def fit_affine(Xt, Xs, ridge=None, grid=RIDGE_GRID, val_frac=0.1, seed=0):
    """Ridge-regularised affine map with bias.

    ridge=None (the default, and what every adapter family uses): the penalty is chosen on a
    held-out `val_frac` of the pairs, by mean cosine to the true source vector (the quantity
    retrieval uses), from `grid` x the mean eigenvalue of the centred Gram matrix, then the map
    is refitted on all pairs. The scale matters: embedding vectors share a large common
    direction, so after centring most of the variance sits in a few directions and the typical
    eigenvalue of A^T A is tiny. The fixed penalty this function used before (1e-2 * n, i.e.
    lambda = 100 at n = 10,000) exceeded nearly every eigenvalue, shrank the map onto the mean
    vector, and sent every query to the same documents (relative recall 0-19% on the 1M text
    pairs, against 37-89% for Procrustes). A numeric `ridge` keeps that old per-sample
    convention (penalty ridge * n * I) for callers that ask for it explicitly.
    """
    Xt = np.asarray(Xt, np.float32); Xs = np.asarray(Xs, np.float32)
    if ridge is not None:
        if ridge <= 0:
            raise ValueError(f"fit_affine needs ridge > 0 (got {ridge}): A.T @ A is singular "
                             f"whenever the fit sample is smaller than the target dimension, and "
                             f"np.linalg.solve would report that as a bare LinAlgError")
        mt, ms = Xt.mean(0), Xs.mean(0)
        W = _ridge_solve(Xt - mt, Xs - ms, ridge * len(Xt))
        ad = Adapter(W, (ms - mt @ W).astype(np.float32)); ad.ridge = float(ridge * len(Xt))
        return ad
    if not grid or min(grid) <= 0:
        raise ValueError(f"fit_affine needs a grid of positive ridge scales (got {grid})")
    n = len(Xt); nv = int(round(val_frac * n)) if n > 10 else 0
    mt, ms = Xt.mean(0), Xs.mean(0)
    A_all, B_all = Xt - mt, Xs - ms
    scale = max(float(np.einsum("ij,ij->", A_all, A_all)) / A_all.shape[1], 1e-12)  # mean eigenvalue
    lam = scale * float(np.median(grid))
    if nv:
        perm = np.random.default_rng(seed).permutation(n); va, tr = perm[:nv], perm[nv:]
        mtr, mstr = Xt[tr].mean(0), Xs[tr].mean(0)
        A, B = Xt[tr] - mtr, Xs[tr] - mstr
        AtA, AtB, d = A.T @ A, A.T @ B, A.shape[1]
        best = -np.inf
        for g in grid:
            W = np.linalg.solve(AtA + scale * g * np.eye(d, dtype=A.dtype), AtB).astype(np.float32)
            v = _mean_cos(Xt[va] @ W + (mstr - mtr @ W), Xs[va])
            if v > best:
                best, lam = v, scale * g
    W = _ridge_solve(A_all, B_all, lam)
    ad = Adapter(W, (ms - mt @ W).astype(np.float32)); ad.ridge = float(lam)
    return ad


def fit_lowrank(Xt, Xs, rank=64, ridge=None):
    """Low-rank affine map (Drift-Adapter's second family): reduced-rank ridge regression.

    Fit the full ridge-affine map, then project it onto the top-`rank` directions of its own
    fitted outputs (the classical reduced-rank regression solution), so W = W_full V_r V_r^T
    has rank <= `rank`. With rank >= min(d_target, d_source) it equals fit_affine exactly.
    """
    if rank < 1:
        raise ValueError(f"fit_lowrank needs rank >= 1 (got {rank})")
    full = fit_affine(Xt, Xs, ridge)
    mt = Xt.mean(0)
    fitted = (Xt - mt) @ full.W
    r = int(min(rank, *fitted.shape))
    _, _, Vt = np.linalg.svd(fitted, full_matrices=False)
    V = Vt[:r].T.astype(np.float32)
    W = (full.W @ V @ V.T).astype(np.float32)
    b = (Xs.mean(0) - mt @ W).astype(np.float32)
    ad = Adapter(W, b); ad.ridge = full.ridge
    return ad


class MLPAdapter(Adapter):
    """Residual MLP adapter: y = x W + b + relu(x W1 + b1) W2. The linear part is the
    Procrustes/affine starting point; the residual branch starts at zero, so an untrained
    MLP is exactly its linear initialisation."""

    def __init__(self, W, b, W1, b1, W2):
        super().__init__(W, b)
        self.W1, self.b1, self.W2 = W1, b1, W2

    def __call__(self, Xt, normalize=True):
        Y = super().__call__(Xt, normalize=False)
        H = np.maximum(np.asarray(Xt, np.float32) @ self.W1 + self.b1, 0.0)
        Y = Y + H @ self.W2
        return normalize_rows(Y) if normalize else Y


def fit_mlp(Xt, Xs, hidden=512, epochs=30, lr=(1e-3, 3e-3, 1e-2), batch=256, wd=1e-4, seed=0,
            init="affine", val_frac=0.1):
    """Compact residual MLP (Drift-Adapter's third family), trained in numpy with Adam on a
    cosine loss -- the quantity retrieval actually uses. The linear part is initialised from
    `init` ('affine' or 'procrustes') and trained jointly with the residual branch. A held-out
    `val_frac` of the pairs picks the best epoch AND the learning rate when `lr` is a sequence,
    so neither is tuned by hand per pair and more epochs cannot overfit silently.
    No torch: the sandbox that builds the paper has none, and 10k pairs train in minutes.
    With 10 pairs or fewer there is no held-out set (val_frac rounds to nothing); no epoch can
    then beat the start, so the result is the linear initialisation with a zero residual -- for
    every learning rate alike, so the "choice" of lr there is the first one and means nothing."""
    if isinstance(lr, (list, tuple)):
        fits = [fit_mlp(Xt, Xs, hidden, epochs, x, batch, wd, seed, init, val_frac) for x in lr]
        return max(fits, key=lambda a: a.val_cos)
    Xt = np.asarray(Xt, np.float32); Xs = np.asarray(Xs, np.float32)
    rng = np.random.default_rng(seed)
    n = len(Xt); perm = rng.permutation(n); nv = max(1, int(round(val_frac * n))) if n > 10 else 0
    va, tr = perm[:nv], perm[nv:]
    lin = (fit_affine if init == "affine" else fit_procrustes)(Xt[tr], Xs[tr])
    dt, ds = lin.W.shape
    P = {"W": lin.W.copy(), "b": lin.b.copy(),
         "W1": (rng.standard_normal((dt, hidden)) * np.sqrt(2.0 / dt)).astype(np.float32),
         "b1": np.zeros(hidden, np.float32), "W2": np.zeros((hidden, ds), np.float32)}
    M = {k: np.zeros_like(v) for k, v in P.items()}; V = {k: np.zeros_like(v) for k, v in P.items()}
    Tn = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)

    def forward(X):
        Hp = X @ P["W1"] + P["b1"]; H = np.maximum(Hp, 0.0)
        return X @ P["W"] + P["b"] + H @ P["W2"], H, Hp

    def val_cos():
        if nv == 0:
            return 0.0
        Y, _, _ = forward(Xt[va]); Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return float((Yn * Tn[va]).sum(1).mean())

    best, best_P, t = val_cos(), {k: v.copy() for k, v in P.items()}, 0
    for _ in range(epochs):
        order = rng.permutation(tr)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]; X, T = Xt[idx], Tn[idx]
            Y, H, Hp = forward(X)
            nrm = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12); Yn = Y / nrm
            c = (Yn * T).sum(1, keepdims=True)
            gY = -(T - c * Yn) / nrm / len(idx)          # d(-mean cos)/dY
            g = {"W": X.T @ gY, "b": gY.sum(0), "W2": H.T @ gY}
            gH = (gY @ P["W2"].T) * (Hp > 0)
            g["W1"] = X.T @ gH; g["b1"] = gH.sum(0)
            t += 1
            for k in P:
                if k in ("W", "W1", "W2"):
                    g[k] = g[k] + wd * P[k]
                M[k] = 0.9 * M[k] + 0.1 * g[k]; V[k] = 0.999 * V[k] + 0.001 * g[k] ** 2
                mh = M[k] / (1 - 0.9 ** t); vh = V[k] / (1 - 0.999 ** t)
                P[k] = (P[k] - lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        v = val_cos()
        if v > best:
            best, best_P = v, {k: x.copy() for k, x in P.items()}
    ad = MLPAdapter(best_P["W"], best_P["b"], best_P["W1"], best_P["b1"], best_P["W2"])
    ad.val_cos, ad.lr = best, lr
    return ad


FITTERS = {"procrustes": fit_procrustes, "affine": fit_affine, "lowrank": fit_lowrank,
           "mlp": fit_mlp}


def relative_residual(adapter, Xt, Xs):
    den = float(np.linalg.norm(Xs))
    if den == 0.0:
        raise ValueError("relative_residual: the source block has zero norm, so there is no "
                         "scale to be relative to. An all-zero encoding is a broken encode "
                         "(scripts/encode_check.py catches it), not a residual of 0 or 1.")
    R = adapter(Xt, normalize=False) - Xs
    return float(np.linalg.norm(R) / den)


def linear_cka(X, Y):
    """Linear CKA between two centred blocks.

    Refuses a zero-variance block rather than returning a number. The tempting guard -- den == 0
    so return 1.0 -- is the worst available answer: when X is constant, num is 0 too, so 1.0
    would report a COLLAPSED encoding as perfectly similar to anything and send it into the map
    table as the most compatible pair in the study. Undefined is undefined.

    Note for reading the results: linear CKA is invariant to rotation and isotropic scaling, so
    it is high whenever two spaces are linearly related however differently they rank documents.
    On the 2026-11-07 run, WITHIN the ten text pairs CKA and kappa are indistinguishable as
    predictors of adapter quality: r(CKA, M_A) = +0.754 against kappa's +0.755. Pooling the code
    pairs takes CKA to -0.186, because the two lowest-CKA pairs are both the code corpus and both
    have high M_A -- a corpus confound, not a property of CKA. An earlier version of this comment
    blamed the identity control; recomputation did not support that (CORRECTIONS.md, CLAIM 1).
    Neither measure transfers across corpora. Prefer kappa, for its cheaper interpretation rather
    than for better accuracy.
    """
    # the test is relative, not `== 0`: a block of identical float32 rows does not centre to
    # exactly zero (the mean of 64 copies is not bit-exact), and neither would a real collapsed
    # encoding, so an equality check would let the case it exists for straight through
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"linear_cka compares two views of the SAME rows, so both blocks need "
                         f"the same sample count; got {X.shape[0]} and {Y.shape[0]}")
    if X.shape[0] < 2:
        raise ValueError(f"linear_cka needs at least 2 samples to have a variance to compare; "
                         f"got {X.shape[0]}. (compatibility() reports cka=nan rather than "
                         f"calling this with a single row.)")
    flat_x, flat_y = _centred_or_none(X), _centred_or_none(Y)
    if flat_x is None or flat_y is None:
        both = flat_x is None and flat_y is None
        side = ("both blocks have" if both else
                "the first block has" if flat_x is None else "the second block has")
        raise ValueError(f"linear_cka is undefined: {side} no variance left after centring "
                         f"(every row effectively identical). Check the encoding for collapse -- "
                         f"scripts/encode_check.py tests for exactly this.")
    X, Y = flat_x, flat_y
    nx = float(np.linalg.norm(X.T @ X))
    ny = float(np.linalg.norm(Y.T @ Y))
    return float(np.linalg.norm(Y.T @ X) ** 2 / (nx * ny))


def _centred_or_none(X, rel_tol=1e-6):
    """X minus its column means, or None if that leaves nothing but rounding noise.

    `rel_tol` is calibrated for unit-norm embeddings, where a real pair's centred norm is a
    good fraction of its raw norm and 1e-6 is orders of magnitude below it. It is not tuned for
    data whose rows are nearly parallel with a large shared mean -- such a block has small but
    meaningful variance and would be reported here as collapsed.
    """
    raw = float(np.linalg.norm(X))
    C = X - X.mean(0)
    return None if float(np.linalg.norm(C)) <= rel_tol * max(raw, 1e-30) else C


def compatibility(doc_src, doc_tgt, kind="procrustes", n_fit=10000, n_eval=5000, seed=0):
    """Fit on one doc sample, evaluate on a disjoint one. Returns (adapter, report)."""
    if kind not in FITTERS:
        raise ValueError(f"unknown adapter kind {kind!r}; valid: {sorted(FITTERS)}")
    if doc_src.ndim != 2 or doc_tgt.ndim != 2:
        raise ValueError(f"doc_src and doc_tgt must be 2-D, got {doc_src.shape} and {doc_tgt.shape}")
    if doc_src.shape[0] != doc_tgt.shape[0]:
        raise ValueError(f"doc_src has {doc_src.shape[0]} rows and doc_tgt has "
                         f"{doc_tgt.shape[0]}; they must be the same corpus in the same order. "
                         f"Row counts this far apart mean one side was encoded from a different "
                         f"corpus file (scripts/assemble_pair.py refuses this).")
    rng = np.random.default_rng(seed)
    N = doc_src.shape[0]
    perm = rng.permutation(N)
    # the fit sample takes at most half the corpus and the eval sample takes what is left; the
    # only requirement here is that the two are disjoint, so n_eval has no N//2 cap of its own
    n_fit = min(n_fit, N // 2)
    n_eval = min(n_eval, N - n_fit)
    if n_fit < 1 or n_eval < 1:
        # at N < 2 the fit sample is empty and the SVD of the resulting zero matrix returns an
        # arbitrary W without complaining, which is worse than failing
        raise ValueError(f"compatibility needs at least 2 documents to split a fit set from a "
                         f"disjoint eval set; got N={N} (n_fit={n_fit}, n_eval={n_eval})")
    fi, ev = perm[:n_fit], perm[n_fit:n_fit + n_eval]
    ad = FITTERS[kind](doc_tgt[fi], doc_src[fi])
    r = relative_residual(ad, doc_tgt[ev], doc_src[ev])
    # the residual needs one eval row; CKA needs two, because a single centred row is the zero
    # vector. At N=2 (n_fit=n_eval=1) the residual is still meaningful, so report it and leave
    # cka undefined rather than failing the whole measurement with a collapse message.
    cka = linear_cka(doc_src[ev], doc_tgt[ev]) if n_eval >= 2 else float("nan")
    return ad, {
        "adapter": kind,
        "n_fit": int(n_fit),
        "n_eval": int(n_eval),
        "residual": r,
        "kappa": 1.0 - r,
        "cka": cka,
    }
