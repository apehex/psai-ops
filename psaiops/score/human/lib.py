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
    lower_val: int=100,
    upper_val: int=201088, # size of the vocabulary used by gpt-oss
) -> object:
    # the first token cannot be rated => (B, T-1, 1) and (B, T-1, V)
    __indices = indices_arr[:, 1:].detach().int().unsqueeze(-1)
    __logits = logits_arr[:, :-1].detach().float()
    # fetch the logits of the tokens chosen in the actual output
    __chosen = __logits.gather(dim=-1, index=__indices)
    # count the tokens with higher logits
    __ranks = (__logits > __chosen).float().sum(dim=-1, keepdim=True)
    # normalization factors
    __llower = math.log(1 + lower_val)
    __lupper = math.log(1 + upper_val)
    # the metric is in [0.5; 1] with shape (B, T-1, 1)
    return 0.5 * (1.0 + torch.clamp((torch.log(1 + __ranks) - __llower) / (__lupper - __llower), min=0.0, max=1.0))

# ENTROPY ######################################################################

def compute_entropy_metrics(
    logits_arr: object,
) -> object:
    # the first token can be rated actually (B, T, V)
    __outputs = logits_arr.detach().float()
    # compute the log probs
    __outputs = torch.log_softmax(__outputs, dim=-1)
    # reduce the last axis
    return -(torch.exp(__outputs) * __outputs).sum(dim=-1)

# PERPLEXITY ###################################################################

def compute_perplexity_metrics(
    indices_arr: object,
    logits_arr: object,
) -> object:
    # the first token cannot be rated => (B, T-1, 1) and (B, T-1, V)
    __indices = indices_arr[:, 1:].detach().int().unsqueeze(-1)
    __logits = logits_arr[:, :-1].detach().float()
    # compute the log probs
    __outputs = torch.log_softmax(__logits, dim=-1)
    # fetch the logprobs of the tokens chosen in the actual output
    __outputs = __outputs.gather(dim=-1, index=__indices)
    # compute the perplexity exp(E(-log(p(t))))
    return torch.exp(-torch.mean(__outputs, dim=-1))

# POSTPROCESS ##################################################################

def postprocess_score_cls(
    score_arr: object,
    scale_val: float=1.0,
) -> list:
    # remove the orphan axes => flat sequence
    __scores = score_arr.squeeze().numpy().tolist()
    # rescale and output str labels
    return [str(int(__s * scale_val)) for __s in __scores]
