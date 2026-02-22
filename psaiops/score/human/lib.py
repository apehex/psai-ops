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

SURPRISAL_DIM_MIN = 1
SURPRISAL_DIM_MAX = 33

# UNICODE ########################################################################

UNICODE_RANGES = [
    (0x0100, 0x017f), # latin extended A: used to offset special characters of ASCII
    (0x0180, 0x024f), # latin extended B: used to offset special characters of ASCII
    (0x0250, 0x02af), # IPA extension: used to offset special characters of ASCII
    (0x02b0, 0x02ff), # spacing modifier letters: fancy glyphs used by LLMS, like the prime \u02b9
    (0x0300, 0x036f), # combining diacritical marks: more fancy glyphs like the tilde \u0334
    (0x0360, 0x03ff), # Greek and Coptic: used by LLMs in equations, while we use named glyphs (cf LaTeX)
    (0x2000, 0x206f), # general punctuation: fancy non ASCII puntuation like the double quote \u201c
    (0x2070, 0x209f), # superscripts and subscripts: fancy exponents like the parenthesis \u207d
    (0x20d0, 0x20ff), # combining diacritical marks for symbols: never seen those
    (0x2100, 0x214f), # letterlike symbols, like the struck N for the natural numbers \u2115
    (0x2150, 0x218f), # number forms, like the symbol for VII \u2166
    (0x2190, 0x21ff), # arrows, fancy arrows like the symbol for <=> \u21d4
    (0x2200, 0x22ff), # mathematical operators, like the double integral \u222c
    (0x2300, 0x2d7f), # a whole bunch of fancy symbols, if you care partition this
    (0x010000, 0xffffff),] # anything outside the basic multilingual plane is suspicious

# GENERATE #######################################################################

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
    __data = __data.contiguous().view(math.prod(__prefix), 1, __dim)
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

# UNICODE ######################################################################

def _is_char_in_blacklist(
    char_str: str,
    unicode_arr: list=UNICODE_RANGES
) -> bool:
    return any([
        (ord(char_str) >= __s) and (ord(char_str) <= __e)
        for (__s, __e) in unicode_arr])

def _is_token_in_blacklist(
    token_str: str,
    unicode_arr: list=UNICODE_RANGES
) -> bool:
    return any([
        _is_char_in_blacklist(char_str=__c, unicode_arr=unicode_arr)
        for __c in token_str])

def compute_unicode_metrics(
    tokens_arr: list, # list of token strings, without escaping the special characters
    unicode_arr: list=UNICODE_RANGES,
) -> object:
    return torch.Tensor([[
        0.0 if _is_token_in_blacklist(token_str=__t, unicode_arr=unicode_arr) else 0.5
        for __t in tokens_arr]])

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
    return -__outputs.gather(dim=-1, index=__indices).squeeze(-1)

def postprocess_nllikelihoods(
    nlls_arr: object,
    lower_val: float=math.log(2), # perplexity 2 => average probability of 0.5, values below are rare and considered the extrem of LLM sampling
    upper_val: float=math.log(800), # perplexity 800 => computed so the 0.5 * (L + U) = log(40) => perplexity 40 is undecided between LLM / human
    scope_dim: int=PERPLEXITY_DIM_MIN,
) -> object:
    # prevent the perplexity from just accumulating over time, and avoid computing the perplexity of a single token
    __dim = max(PERPLEXITY_DIM_MIN, min(PERPLEXITY_DIM_MAX, scope_dim))
    # compute the log of the perplexity E(-log(p(t)))
    __outputs = compute_average_pooling(nlls_arr, pool_dim=__dim, axis_idx=1)
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

# SURPRISAL ####################################################################

def postprocess_surprisals(
    surprisals_arr: object,
    scope_dim: int=SURPRISAL_DIM_MIN,
) -> object:
    # prevent the surprisal from just accumulating over time (when the window is the length of the sample)
    __dim = max(SURPRISAL_DIM_MIN, min(SURPRISAL_DIM_MAX, scope_dim))
    # normalize (B, T-1)
    __outputs = torch.clamp(0.5 + (surprisals_arr / math.log(VOCABULARY_DIM)), min=0.0, max=1.0)
    # and average over the scope (B, T-1)
    return compute_average_pooling(__outputs, pool_dim=__dim, axis_idx=1)

def compute_surprisal_metrics(
    indices_arr: object,
    logits_arr: object,
    scope_dim: int=SURPRISAL_DIM_MIN,
) -> object:
    # compute the raw entropies (B, T-1)
    __expectations = compute_entropies(
        logits_arr=logits_arr)
    # compute the negative log likelihoods (B, T-1)
    __realizations = compute_nllikelihoods(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # the surprisal is defined as the delta between entropy and NLL (B, T-1)
    __outputs = __realizations - __expectations
    # normalized like the entropy (B, T-1)
    return postprocess_surprisals(
        surprisals_arr=__outputs,
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
