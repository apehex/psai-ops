import functools
import math

import matplotlib
import numpy
import torch
import torch.nn.functional

import mlable.shapes

# META ###########################################################################

VOCABULARY_DIM = 201088 # size of the vocabulary used by gpt-oss

RANK_DIM_MIN = 1
RANK_DIM_MAX = 1

ENTROPY_DIM_MIN = 1
ENTROPY_DIM_MAX = 33

PERPLEXITY_DIM_MIN = 5
PERPLEXITY_DIM_MAX = 33

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

# POOLING ########################################################################

def compute_average_pooling(
    data_arr: object,
    pool_dim: int,
    axis_idx: int=1,
) -> object:
    # always take an odd kernel size, so that the number of elements taken on the left and right are the same
    __kernel = 2 * (pool_dim // 2) + 1
    # interpret negative indices
    __axis = axis_idx % data_arr.ndim
    # move the pooling axis to the end
    __data = data_arr.movedim(source=__axis, destination=-1)
    # save the shape before flattening
    *__prefix, __dim = tuple(__data.shape)
    # flatten to match the expected shape of pool1d
    __data = __data.reshape(math.prod(__prefix), 1, __dim)
    # actually pool the data
    __data = torch.nn.functional.avg_pool1d(
        __data,
        kernel_size=__kernel,
        stride=1,
        padding=__kernel // 2,
        ceil_mode=False,
        count_include_pad=False)
    # restore the batch axes
    __data = __data.reshape((*__prefix, int(__data.shape[-1])))
    # move the axis back
    return __data.movedim(source=-1, destination=__axis)

# RANK #########################################################################

def compute_ranks(
    indices_arr: object,
    logits_arr: object,
) -> object:
    # the first token cannot be rated => (B, T-1, 1) and (B, T-1, V)
    __indices = indices_arr[:, 1:].detach().int().unsqueeze(-1)
    __logits = logits_arr[:, :-1].detach().float()
    # fetch the logits of the tokens chosen in the actual output (B, T-1, 1)
    __chosen = __logits.gather(dim=-1, index=__indices)
    # count the tokens with higher logits (B, T-1)
    return (__logits > __chosen).float().sum(dim=-1, keepdim=False)

def postprocess_ranks(
    ranks_arr: object,
    lower_val: int=100,
    upper_val: int=VOCABULARY_DIM,
    scope_dim: int=RANK_DIM_MIN,
) -> object:
    # prevent the pooling from flattening the ranks
    __dim = max(RANK_DIM_MIN, min(RANK_DIM_MAX, scope_dim))
    # normalization factors ()
    __llower = math.log(1 + lower_val)
    __lupper = math.log(1 + upper_val)
    # the metric is in [0.5; 1] with shape (B, T-1)
    __outputs = 0.5 * (1.0 + torch.clamp((torch.log(1 + ranks_arr) - __llower) / (__lupper - __llower), min=0.0, max=1.0))
    # compute the average in the scope to smooth the output (B, T-1)
    return compute_average_pooling(__outputs, pool_dim=__dim, axis_idx=1)

def compute_rank_metrics(
    indices_arr: object,
    logits_arr: object,
    lower_val: int=100,
    upper_val: int=VOCABULARY_DIM,
    scope_dim: int=RANK_DIM_MIN,
) -> object:
    # compute the raw ranks (B, T-1)
    __outputs = compute_ranks(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # and normalize them (B, T-1)
    return postprocess_ranks(
        ranks_arr=__outputs,
        lower_val=lower_val,
        upper_val=upper_val,
        scope_dim=scope_dim)

# ENTROPY ######################################################################

def compute_entropies(
    logits_arr: object
) -> object:
    # the first token can be rated actually (B, T-1, V)
    __outputs = logits_arr[:, :-1].detach().float()
    # compute the log probs (B, T-1, V)
    __outputs = torch.log_softmax(__outputs, dim=-1)
    # reduce the last axis (B, T-1)
    return -(torch.exp(__outputs) * __outputs).sum(dim=-1, keepdim=False)

def postprocess_entropies(
    entropies_arr: object,
    scope_dim: int=ENTROPY_DIM_MIN,
) -> object:
    # prevent the entropy from just accumulating over time (when the window is the length of the sample)
    __dim = max(ENTROPY_DIM_MIN, min(ENTROPY_DIM_MAX, scope_dim))
    # normalize (B, T-1)
    __outputs = entropies_arr / math.log(VOCABULARY_DIM)
    # and average over the scope (B, T-1)
    return compute_average_pooling(__outputs, pool_dim=__dim, axis_idx=1)

def compute_entropy_metrics(
    logits_arr: object,
    scope_dim: int=ENTROPY_DIM_MIN,
) -> object:
    # compute the raw entropies (B, T-1)
    __outputs = compute_entropies(
        logits_arr=logits_arr)
    # and normalize them (B, T-1)
    return postprocess_entropies(
        entropies_arr=__outputs,
        scope_dim=scope_dim)

# PERPLEXITY ###################################################################

def compute_nllikelihoods(
    indices_arr: object,
    logits_arr: object,
) -> object:
    # the first token cannot be rated => (B, T-1, 1) and (B, T-1, V)
    __indices = indices_arr[:, 1:].detach().int().unsqueeze(-1)
    __logits = logits_arr[:, :-1].detach().float()
    # compute the log probs (B, T-1, V)
    __outputs = torch.log_softmax(__logits, dim=-1)
    # fetch the logprobs of the tokens chosen in the actual output (B, T-1)
    return __outputs.gather(dim=-1, index=__indices).squeeze(-1)

def postprocess_nllikelihoods(
    nlls_arr: object,
    lower_val: float=math.log(2), # perplexity 2 => average probability of 0.5, values below are rare and considered the extrem of LLM sampling
    upper_val: float=math.log(800), # perplexity 800 => computed so the 0.5 * (L + U) = log(40) => perplexity 40 is undecided between LLM / human
    scope_dim: int=PERPLEXITY_DIM_MIN,
) -> object:
    # prevent the perplexity from just accumulating over time, and avoid computing the perplexity of a single token
    __dim = max(PERPLEXITY_DIM_MIN, min(PERPLEXITY_DIM_MAX, scope_dim))
    # compute the log of the perplexity E(-log(p(t)))
    __outputs = compute_average_pooling(-nlls_arr, pool_dim=__dim, axis_idx=1)
    # rescale the metric to cover [0; 1] (B, T-1)
    return torch.clamp((__outputs - lower_val) / (upper_val - lower_val), min=0.0, max=1.0)

def compute_perplexity_metrics(
    indices_arr: object,
    logits_arr: object,
    lower_val: float=math.log(2), # perplexity 2 => average probability of 0.5, values below are rare and considered the extrem of LLM sampling
    upper_val: float=math.log(800), # perplexity 800 => computed so the 0.5 * (L + U) = log(40) => perplexity 40 is undecided between LLM / human
    scope_dim: int=PERPLEXITY_DIM_MIN,
) -> object:
    # compute the negative log likelihoods (B, T-1)
    __outputs = compute_nllikelihoods(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # and normalize them (B, T-1)
    return postprocess_nllikelihoods(
        nlls_arr=__outputs,
        lower_val=lower_val,
        upper_val=upper_val,
        scope_dim=scope_dim)

# FINAL SCORES #################################################################

def postprocess_score_cls(
    score_arr: object,
    scale_val: float=1.0,
) -> list:
    # remove the orphan axes => flat sequence
    __scores = score_arr.squeeze().numpy().tolist()
    # rescale and output str labels
    return [str(int(__s * scale_val)) for __s in __scores]
