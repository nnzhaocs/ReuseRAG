"""Explicit per-model encoding contracts (configs/models.json) and the checks around them.

Why: three encoders in the first 1M run were mis-encoded silently (bge-m3, codesage-base
queries, Qwen3-8B). A substring table guessed prefixes; nothing recorded padding side, EOS,
pooling or library versions. Here every model must have an explicit contract, the loaded
model is compared against it, and every encoding records the contract, the checks and the
versions next to the vectors.

Pure-Python parts (load/validate/prefix/padding check) are unit-tested with a fake model;
load_model()/encode_texts() need sentence-transformers and a GPU box.
"""
import json
import os

import numpy as np

REQUIRED = ("revision", "query_prefix", "doc_prefix", "query_prompt_name", "doc_prompt_name",
            "trust_remote_code", "max_seq_len_cap", "pooling", "padding_side", "add_eos",
            "normalize", "status", "source")
STATUSES = ("verified", "unverified", "blocked")
POOLING_FLAGS = {  # sentence-transformers Pooling attributes
    "cls": "pooling_mode_cls_token", "mean": "pooling_mode_mean_tokens",
    "max": "pooling_mode_max_tokens", "lasttoken": "pooling_mode_lasttoken",
}
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "models.json")


class ContractError(Exception):
    pass


def load_contracts(path=DEFAULT_PATH):
    with open(path) as f:
        models = json.load(f)["models"]
    for mid, c in models.items():
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            raise ContractError(f"{mid}: contract missing fields {missing}")
        if c["status"] not in STATUSES:
            raise ContractError(f"{mid}: status {c['status']!r} not in {STATUSES}")
    return models


def get_contract(model_id, path=DEFAULT_PATH, allow_blocked=False, overrides=None):
    """The contract for model_id. Unknown models are refused: add an entry first."""
    models = load_contracts(path)
    if model_id not in models:
        raise ContractError(
            f"{model_id!r} has no encoding contract in {os.path.relpath(path)}. Add one "
            f"(prefixes, prompt names, pooling, padding side, EOS, revision) from the model "
            f"card; do not rely on name matching.")
    c = dict(models[model_id], model_id=model_id)
    if overrides:
        c.update(overrides)
        c["overridden"] = sorted(overrides)
    if c["status"] == "blocked" and not allow_blocked:
        raise ContractError(f"{model_id} is blocked: {c.get('notes', '')}  "
                            f"(resolve it, or pass --allow-blocked for a diagnostic run)")
    return c


def prefix_for(c, kind):
    return c["query_prefix"] if kind == "query" else c["doc_prefix"]


def prompt_name_for(c, kind):
    return c["query_prompt_name"] if kind == "query" else c["doc_prompt_name"]


def apply_contract(st_model, c, max_seq_len):
    """Make a loaded SentenceTransformer obey the contract; return a list of findings.
    Mismatches the contract can fix (padding side, EOS) are enforced and reported;
    mismatches it cannot fix (a Pooling module whose mode differs, a missing prompt) raise.
    A model that ships NO Pooling module cannot be compared and is recorded as a finding in the
    sidecar, not raised: codesage is such a model (SentenceTransformer falls back to mean
    pooling), and its contract is verified instead by encode_check's self-retrieval test."""
    findings = []
    if max_seq_len > c["max_seq_len_cap"]:
        raise ContractError(f"max_seq_len {max_seq_len} exceeds the contract cap {c['max_seq_len_cap']}")
    st_model.max_seq_length = max_seq_len
    tok = getattr(st_model, "tokenizer", None)
    if c["padding_side"] and tok is not None:
        had = getattr(tok, "padding_side", None)
        if had != c["padding_side"]:
            tok.padding_side = c["padding_side"]
            findings.append(f"padding_side enforced: tokenizer had {had!r}, contract {c['padding_side']!r}")
    if c["add_eos"] and tok is not None:
        findings.append(_enforce_eos(tok))
    if c["pooling"]:
        pool = _pooling_module(st_model)
        if pool is None:
            findings.append("pooling: no Pooling module found, could not compare")
        else:
            active = [k for k, attr in POOLING_FLAGS.items() if getattr(pool, attr, False)]
            if active != [c["pooling"]]:
                raise ContractError(f"pooling mismatch: model uses {active}, contract says {c['pooling']!r}")
    for kind in ("query", "doc"):
        pn = prompt_name_for(c, kind)
        if pn and pn not in (getattr(st_model, "prompts", None) or {}):
            raise ContractError(f"{kind} prompt_name {pn!r} not defined by the model "
                                f"(has {sorted((getattr(st_model, 'prompts', None) or {}))})")
        problem = double_instruction(c, kind, getattr(st_model, "prompts", None) or {},
                                     getattr(st_model, "default_prompt_name", None))
        if problem:
            raise ContractError(problem)
    return findings


def double_instruction(c, kind, prompts, default_prompt_name=None):
    """The text a model sees is prompts[prompt_name] + prefix + text. A contract that sets both a
    literal prefix and a prompt name whose prompt is non-empty sends the instruction twice (the
    Qwen3 contracts did: prompts.query IS the Instruct string). Return the problem, or None."""
    pre, pn = prefix_for(c, kind), prompt_name_for(c, kind)
    # with no prompt_name, sentence-transformers applies the model's default prompt, if it has one
    pn = pn or default_prompt_name
    if pre and pn and (prompts.get(pn) or ""):
        return (f"{kind}: contract sets prefix {pre[:40]!r} AND prompt_name {pn!r}, whose prompt is "
                f"{prompts[pn][:40]!r}: the model would see the instruction twice. Keep one.")
    return None


def stale_encoding(rec, c, kind):
    """Fields the encoding's sidecar recorded that the CURRENT contract no longer agrees with:
    vectors made under an earlier contract are not the vectors this contract would make."""
    out = []
    if rec.get("text_prefix") is not None and rec["text_prefix"] != prefix_for(c, kind):
        out.append(f"text_prefix {rec['text_prefix'][:40]!r} != contract {prefix_for(c, kind)[:40]!r}")
    if "prompt_name" in rec and rec["prompt_name"] != prompt_name_for(c, kind):
        out.append(f"prompt_name {rec['prompt_name']!r} != contract {prompt_name_for(c, kind)!r}")
    return out


def _pooling_module(st_model):
    try:
        mods = list(st_model) if hasattr(st_model, "__iter__") else []
    except TypeError:
        mods = []
    for m in mods:
        if any(hasattr(m, a) for a in POOLING_FLAGS.values()):
            return m
    return None


def _enforce_eos(tok):
    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is None:
        raise ContractError("contract requires add_eos but the tokenizer has no eos_token_id")
    if hasattr(tok, "add_eos_token"):
        try:
            tok.add_eos_token = True
        except Exception:  # some tokenizers expose a read-only property
            pass
    ids = tok("a")["input_ids"]
    if hasattr(ids, "tolist"):  # tensors / arrays
        ids = ids.tolist()
    ids = ids[0] if ids and isinstance(ids[0], list) else ids
    if ids and ids[-1] == eos_id:
        return "add_eos enforced via tokenizer.add_eos_token (verified on a probe string)"
    raise ContractError("contract requires add_eos but the tokenizer does not append EOS after "
                        "setting add_eos_token; add an explicit strategy for this model")


def padding_check(encode_fn, texts, n=64, warn_below=0.999, fail_below=0.99):
    """Batched vs one-at-a-time encodings of the same texts. With correct masking and
    pooling they agree up to float noise; disagreement means padding leaks into the
    embedding (the suspected codesage-base / Qwen failure). encode_fn(texts, batch_size)
    -> array of unit vectors. Left-padded rotary models may differ slightly; hence a
    warn and a fail threshold."""
    sample = list(texts[:n])
    if not sample:
        return {"n": 0, "min_cos": None, "status": "skipped"}
    # mix short and long texts in one batch so padding actually happens
    order = np.argsort([len(t) for t in sample])
    mixed = [sample[i] for i in np.concatenate([order[::2], order[1::2][::-1]])]
    batched = np.asarray(encode_fn(mixed, len(mixed)), dtype=np.float32)
    single = np.vstack([np.asarray(encode_fn([t], 1), dtype=np.float32) for t in mixed])
    cos = np.sum(batched * single, axis=1) / (
        np.linalg.norm(batched, axis=1) * np.linalg.norm(single, axis=1))
    m = float(cos.min())
    status = "ok" if m >= warn_below else ("warn" if m >= fail_below else "fail")
    return {"n": len(mixed), "min_cos": m, "mean_cos": float(cos.mean()), "status": status,
            "warn_below": warn_below, "fail_below": fail_below}


def self_retrieval_check(doc_emb, query_fn, texts, idx, pool, batch=32,
                         warn_below=0.8, fail_below=0.3):
    """Encode a sample of DOCUMENT texts through the QUERY path and check that each retrieves
    its own document vector, against a pool of documents.

    This is the end-to-end check the first 1M run did not have. The contract and padding checks
    compare the loaded model against its own card; they cannot see that the query side was
    pooled the wrong way, that the weights never actually downloaded (a 3.3 MB "model"), or that
    the rows are misaligned. A working bi-encoder puts a document's own text first by a wide
    margin; a collapsed or garbage encoding scores at chance, 1 / |pool|.

    `warn_below` / `fail_below` are judgement, not measured: record them with the result and do
    not tune them per model. Asymmetric models (a strong query prefix) sit lower than symmetric
    ones, which is why the failure threshold is far below any working encoder rather than close
    to the warning one.
    """
    idx = np.unique(np.asarray(idx, np.int64))
    pool = np.unique(np.concatenate([np.asarray(pool, np.int64), idx]))
    if len(idx) == 0 or len(pool) < 2:
        return {"n": 0, "top1_rate": None, "status": "skipped"}
    qv = np.asarray(query_fn([texts[int(i)] for i in idx], batch), dtype=np.float32)
    # not `qv /=`: np.asarray returns the SAME object when given an ndarray, so an encoder
    # handing back a read-only view makes in-place division raise. encode.py already avoids
    # this; this path did not.
    qv = qv / np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
    D = np.asarray(doc_emb[pool], dtype=np.float32)
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    S = qv @ D.T
    where = {int(d): j for j, d in enumerate(pool)}
    want = np.array([where[int(i)] for i in idx])
    own = S[np.arange(len(idx)), want]
    ranks = 1 + (S > own[:, None]).sum(axis=1)
    top1 = float(np.mean(ranks == 1))
    status = "ok" if top1 >= warn_below else ("warn" if top1 >= fail_below else "fail")
    return {"n": int(len(idx)), "pool": int(len(pool)), "top1_rate": top1,
            "mrr": float(np.mean(1.0 / ranks)), "median_rank": float(np.median(ranks)),
            "chance": 1.0 / len(pool), "status": status,
            "warn_below": warn_below, "fail_below": fail_below}


def env_versions():
    out = {}
    for mod in ("torch", "transformers", "sentence_transformers", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # not installed
            out[mod] = None
    import sys as _sys
    out["shims"] = [m for m in ("reusemap.tfcompat",) if m in _sys.modules]
    sc = _sys.modules.get("sitecustomize")   # compat/sitecustomize.py, when on PYTHONPATH (codesage)
    if sc is not None and str(getattr(sc, "__file__", "")).replace("\\", "/").endswith("compat/sitecustomize.py"):
        out["shims"].append("compat/sitecustomize")
    return out


def load_model(c, device=None, fp16=False):  # pragma: no cover - needs sentence-transformers
    from sentence_transformers import SentenceTransformer
    kw = {"trust_remote_code": bool(c["trust_remote_code"])}
    if c["revision"]:
        kw["revision"] = c["revision"]
    m = SentenceTransformer(c["model_id"], device=device, **kw)
    if fp16:
        m = m.half()
    return m


def make_encode_fn(st_model, c, kind, normalize=True):
    """encode_fn(texts, batch_size) applying the contract's prefix and prompt name."""
    pre, pn = prefix_for(c, kind), prompt_name_for(c, kind)
    kw = {"prompt_name": pn} if pn else {}

    def fn(texts, batch_size):
        return st_model.encode([pre + t for t in texts], batch_size=batch_size,
                               normalize_embeddings=normalize and c["normalize"],
                               convert_to_numpy=True, show_progress_bar=False, **kw)
    return fn


def encoding_record(c, kind, max_seq_len, findings, pad, extra=None):
    """What goes into the sidecar next to the vectors."""
    rec = {"model_id": c["model_id"], "kind": kind, "contract": {k: c[k] for k in REQUIRED},
           "contract_status": c["status"], "overridden": c.get("overridden", []),
           "text_prefix": prefix_for(c, kind), "prompt_name": prompt_name_for(c, kind),
           "max_seq_len": max_seq_len, "apply_findings": findings, "padding_check": pad,
           "versions": env_versions()}
    rec.update(extra or {})
    return rec
