import functools
import math

import matplotlib
import numpy
import torch

import mlable.shapes

# GENERATE #######################################################################

@functools.lru_cache(maxsize=32)
def compute_raw_logits(
    indices_arr: object,
    model_obj: object,
) -> tuple:
    # single forward pass
    with torch.no_grad():
        __outputs = model_obj(
            input_ids=indices_arr,
            attention_mask=torch.ones_like(indices_arr),
            output_hidden_states=False,
            output_attentions=False,
            output_scores=False,
            output_logits=True,
            return_dict=True,
            use_cache=True)
    # (B, T, V)
    return __outputs.logits

# RANK #########################################################################

def compute_rank_metrics(
    indices_arr: object,
    logits_arr: object,
) -> object:
    # the first token cannot be rated => (T-1,) and (T-1, V)
    __indices = indices_arr[0, 1:].detach().int()
    __logits = logits_arr[0, :-1].detach().float()
    # fetch the logits of the tokens chosen in the actual output
    __chosen = __logits.gather(dim=-1, index=__indices.unsqueeze(-1))
    # count the tokens with higher logits
    return (__logits > __chosen).int().sum(dim=-1)

# POSTPROCESS ##################################################################

def postprocess_score_cls(
    score_arr: object,
    scale_val: float=1.0,
) -> list:
    # prepend a 0 because the first token cannot be rated
    __scores = [0] + score_arr.numpy().tolist()
    # rescale and output str labels
    return [str(int(__s * scale_val)) for __s in __scores]
