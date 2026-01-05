import functools
import math

import matplotlib
import numpy
import torch

import mlable.shapes

# GENERATE #######################################################################

@functools.lru_cache(maxsize=32)
def generate_token_ids(
    model_obj: object,
    input_ids: torch.Tensor,
    token_num: int,
    topk_num: int = 4,
    topp_num: float = 0.9,
    attention_mask: torch.Tensor=None,
) -> tuple:
    # generate completion
    with torch.no_grad():
        __outputs = model_obj.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=token_num,
            do_sample=(0.0 < topp_num < 1.0) or (topk_num > 0),
            top_k=topk_num if (topk_num > 0) else None,
            top_p=topp_num if (0.0 < topp_num < 1.0) else None,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=False,
            output_scores=False,
            # early_stopping=True,
            use_cache=True)
    # ((B, T), O * L * (B, I, E))
    return __outputs.sequences, __outputs.hidden_states

# MERGE ########################################################################

def merge_hidden_states(
    hidden_data: torch.Tensor,
) -> torch.Tensor:
    # parse the inputs
    __token_dim = len(hidden_data)
    __layer_dim = len(hidden_data[0])
    # stack the data for each layer => (B, L, I + O, E)
    return torch.stack(
        [
            # concatenate the data for all the tokens => (B, I + O, E)
            torch.concatenate([hidden_data[__t][__l] for __t in range(__token_dim)], dim=1)
            for __l in range(__layer_dim)],
        dim=1)

# REDUCE #######################################################################

def reduce_hidden_states(
    hidden_data: torch.Tensor, # (B, L, T, E)
    layer_idx: int, # -1 => select all layers
    token_idx: int, # -1 => select all tokens
    axes_idx: int=2, # token sequence axis
) -> torch.Tensor:
    # parse the hidden states (B, L, T, E)
    __batch_dim, __layer_dim, __token_dim, __hidden_dim = tuple(hidden_data.shape)
    __layer_idx = min(layer_idx, __layer_dim - 1)
    __token_idx = min(token_idx, __token_dim - 1)
    # select the relevant data along each axis
    __layer_slice = slice(0, __layer_dim) if (__layer_idx < 0) else slice(__layer_idx, __layer_idx + 1)
    __token_slice = slice(0, __token_dim) if (__token_idx < 0) else slice(__token_idx, __token_idx + 1)
    # filter the data
    __data = hidden_data[slice(None), __layer_slice, __token_slice, slice(None)]
    # reduce the token axis => (B, L, E)
    return __data.mean(dim=axes_idx, keepdim=False)

# RESCALE ######################################################################

def rescale_hidden_states(
    hidden_data: torch.Tensor, # (B, L, E) or (B, E)
) -> torch.Tensor:
    # compute the scale of the data, layer by layer
    __s = torch.quantile(hidden_data.abs(), q=0.9, dim=-1, keepdim=True)
    # log scaling on large values and linear near 0
    __a = torch.asinh(hidden_data / (__s + torch.finfo().eps))
    # clip and map to [-1; 1]
    return 0.33 * __a.clamp(min=-3, max=3)

# RESHAPE ######################################################################

def reshape_hidden_states(
    hidden_data: torch.Tensor, # (B, L, E) or (B, E)
    layer_idx: int=1,
) -> torch.Tensor:
    # parse the shape
    __shape = tuple(hidden_data.shape)
    # factor the hidden dimension
    __factor = 2 ** round(0.5 * math.log2(__shape[-1]))
    # compute the shape with the last axis split
    __shape = mlable.shapes.divide(shape=__shape, axis=-1, factor=__factor, insert=True, right=True)
    # move the layer axis at the end
    __perm = mlable.shapes.move(shape=range(len(__shape)), before=layer_idx, after=-1)
    # reshape into (B, W, H, L) or (B, W, H)
    return hidden_data.reshape(__shape).permute(*__perm)

# MASK #########################################################################

def mask_hidden_states(
    hidden_data: torch.Tensor, # (B, L, E)
    topk_num: int=128,
) -> torch.Tensor:
    # sanitize
    __k = min(topk_num, int(hidden_data.shape[-1]))
    # indices of the topk values
    __indices = hidden_data.abs().topk(__k, dim=-1, largest=True, sorted=False).indices
    # initialize the mask with False
    __mask = torch.zeros_like(hidden_data, dtype=torch.bool)
    # (B, L, E) mask of the topk values
    return __mask.scatter_(dim=-1, index=__indices, value=True)

# FORMAT #######################################################################

def color_hidden_states(
    hidden_data: numpy.array, # (B, H, W, L)
    color_map: callable=matplotlib.colormaps['coolwarm'],
) -> list:
    # [-1; 1] => [0; 1]
    __data = 0.5 * (hidden_data + 1.0)
    # (B, W, H, L) => (B, W, H, L, 4)
    __rgba = color_map(__data)
    # (B, W, H, L, 3) in [0; 1]
    return __rgba[..., :3]

def size_hidden_states(
    hidden_data: numpy.array, # (B, H, W, L)
    area_min: float=0.01,
    area_max: float=16.0,
    gamma_val: float=1.6,
) -> list:
    # [-1; 1] => [0; 1]
    __data = numpy.abs(hidden_data)
    # gamma < 1 will boost small values and > 1 emphasize larger values
    __data = (__data + numpy.finfo(numpy.float32).eps) ** gamma_val
    # map to point area
    return area_min + (area_max - area_min) * __data

# KL SCORES ####################################################################

def kl_from_logprobs(
    p_log: torch.Tensor,
    q_log: torch.Tensor,
) -> torch.Tensor:
    # compute the KL div from log probabilities (B, T, E) or (T, E)
    return (p_log.exp() * (p_log - q_log)).sum(dim=-1)

def jsd_from_logits(
    final_logits: torch.Tensor,
    prefix_logits: torch.Tensor,
) -> torch.Tensor:
    # compute the log probs from logits (B, T, E) or (T, E)
    __p = torch.log_softmax(final_logits.float(), dim=-1)
    __q = torch.log_softmax(prefix_logits.float(), dim=-1)
    # m = 0.5(p+q) in log-space (logsumexp trick)
    __m = torch.logsumexp(torch.stack([__p, __q], dim=0), dim=0) - math.log(2.0)
    # compute the JSD metric
    __jsd = 0.5 * kl_from_logprobs(__p, __m) + 0.5 * kl_from_logprobs(__q, __m)
    # scale to [0; 1]
    return (__jsd / math.log(2.0)).clamp(0.0, 1.0)

# POSTPROCESS ##################################################################

def postprocess_focus_cls(
    left_idx: int,
    right_idx: int,
    token_dim: int,
) -> list:
    __left_idx = max(-1, min(token_dim, left_idx))
    __right_idx = max(-1, min(token_dim, right_idx))
    # class 1 for the token(s) focused on the left, 0 for the rest
    __left_cls = token_dim * [1] if (__left_idx < 0) else [int(__i == __left_idx) for __i in range(token_dim)]
    # class 2 for the token(s) focused on the right, 0 for the rest
    __right_cls = token_dim * [2] if (__right_idx < 0) else [2 * int(__i == __right_idx) for __i in range(token_dim)]
    # sum the classes so that the overlap has class 3
    return [str(__l + __r) for __l, __r in zip(__left_cls, __right_cls)]

def postprocess_score_cls(
    score_data: torch.Tensor
) -> list:
    return [str(__s) for __s in score_data.numpy().tolist()]
