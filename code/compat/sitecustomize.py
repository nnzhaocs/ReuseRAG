"""API-drift aliases so codesage's remote code (written for transformers ~4.2x / torch ~2.0)
loads under this machine's stack. Each alias restores an API that was moved or removed; none
touches weights, pooling, prefixes or numerics.
  1) Conv1D: moved from transformers.modeling_utils to transformers.pytorch_utils -> re-export
     the SAME class object.
  2) PreTrainedModel.all_tied_weights_keys: read by transformers 5.x during from_pretrained;
     pre-5.x remote models never define it. Default {} (= no tied weights; codesage's lm_head
     is unused for embeddings).
  3) nn.Module.get_head_mask: removed from torch 2.x (deprecated since 1.9). Copied from the
     last torch release that shipped it; codesage only calls it with head_mask=None.
     Only that path is restored: a non-None head_mask needs _convert_head_mask_to_5d, which torch
     2.x also removed, and raises AttributeError. This is a shim for one known caller, not the API.
"""
try:  # imported at every interpreter start once compat/ is on PYTHONPATH: never fail without torch
    import torch as _torch
except ImportError:
    _torch = None
if _torch is not None and not hasattr(_torch.nn.Module, "get_head_mask"):
    import torch.nn as _nn

    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    _nn.Module.get_head_mask = _get_head_mask

try:
    import transformers.modeling_utils as _mu
    if not hasattr(_mu, "Conv1D"):
        from transformers.pytorch_utils import Conv1D as _Conv1D
        _mu.Conv1D = _Conv1D
    if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
        _mu.PreTrainedModel.all_tied_weights_keys = {}
except Exception:
    pass
