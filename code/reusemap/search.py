"""Exact inner-product top-k search (numpy). All measurements use exact search so
they carry no ANN error; ANN noise is measured separately (scripts/ann_noise.py)."""
import numpy as np


def _block_size(n_docs, budget_floats=2e8):
    return max(1, int(budget_floats // max(n_docs, 1)))


def topk_ip(Q, X, k, block=None):
    """Exact top-k of Q @ X.T per row, sorted by descending score.

    Returns (idx int64 [n,k], scores float32 [n,k]). Ties are broken by document
    index so results are deterministic.
    """
    Q = np.asarray(Q, dtype=np.float32)
    if not isinstance(X, np.memmap):
        X = np.asarray(X, dtype=np.float32)  # memmaps are read block-wise by BLAS via the page cache
    n, N = Q.shape[0], X.shape[0]
    k = int(min(k, N))
    block = block or _block_size(N)
    I = np.empty((n, k), np.int64)
    S = np.empty((n, k), np.float32)
    for a in range(0, n, block):
        sc = Q[a:a + block] @ X.T
        if k < N:
            part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        else:
            part = np.broadcast_to(np.arange(N), sc.shape).copy()
        ps = np.take_along_axis(sc, part, 1)
        # sort by (-score, index) for determinism
        order = np.lexsort((part, -ps), axis=1)
        I[a:a + block] = np.take_along_axis(part, order, 1)
        S[a:a + block] = np.take_along_axis(ps, order, 1)
    return I, S


def scores_for(q, X, ids):
    """Target scores of one query vector against a subset of documents."""
    return (X[ids] @ q).astype(np.float32)


def normalize_rows(A):
    A = np.asarray(A, dtype=np.float32)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n
