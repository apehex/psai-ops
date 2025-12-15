import functools
import math

import matplotlib
import numpy
import torch

# GENERATE #######################################################################

@functools.lru_cache(maxsize=32)
def generate_token_ids(
    model_obj: object,
    input_ids: torch.Tensor,
    token_num: int,
    topk_num: int = 4,
    topp_num: float = 0.9,
) -> tuple:
    # generate completion
    with torch.no_grad():
        __outputs = model_obj.generate(
            input_ids=input_ids,
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
    # ((B, T), O * L * (B, I, H))
    return __outputs.sequences, __outputs.hidden_states

# MERGE ########################################################################

def merge_hidden_states(
    hidden_data: torch.Tensor,
) -> torch.Tensor:
    # parse the inputs
    __token_dim = len(hidden_data)
    __layer_dim = len(hidden_data[0])
    # stack the data for each layer => (B, L, I + O, H)
    return torch.stack(
        [
            # concatenate the data for all the tokens => (B, I + O, H)
            torch.concatenate([hidden_data[__t][__l] for __t in range(__token_dim)], dim=1)
            for __l in range(__layer_dim)],
        dim=1)

# REDUCE #######################################################################

def reduce_hidden_states(
    hidden_data: torch.Tensor,
    token_idx: int, # -1 => avg over all tokens
) -> torch.Tensor:
    # parse the hidden states (B, L, T, H)
    __batch_dim, __layer_dim, __token_dim, __hidden_dim = tuple(hidden_data.shape)
    __token_idx = min(token_idx, __token_dim - 1)
    # select the relevant data along each axis
    __token_slice = slice(0, __token_dim) if (__token_idx < 0) else slice(__token_idx, __token_idx + 1)
    # filter the data
    __data = hidden_data[slice(None), slice(None), __token_slice, slice(None)]
    # reduce the token axis => (B, L, H)
    return __data.mean(dim=2, keepdim=False)

# RESCALE ######################################################################

def rescale_hidden_states(
    hidden_data: torch.Tensor, # (B, L, H)
) -> torch.Tensor:
    # compute the scale of the data, layer by layer
    __s = torch.quantile(hidden_data.abs(), q=0.9, dim=-1, keepdim=True)
    # log scaling on large values and linear near 0
    __a = torch.asinh(hidden_data / (__s + torch.finfo().eps))
    # clip and map to [-1; 1]
    return 0.33 * __a.clamp(min=-3, max=3)

# RESHAPE ######################################################################

def reshape_hidden_states(
    hidden_data: torch.Tensor, # (B, L, H)
) -> torch.Tensor:
    # parse the hidden states (B, L, H)
    __batch_dim, __layer_dim, __hidden_dim = tuple(hidden_data.shape)
    # factor the hidden dimension
    __width_dim = math.gcd(__hidden_dim, 2 ** int(math.log2(__hidden_dim))) # greatest power of 2 that divides H
    __height_dim = __hidden_dim // __width_dim
    # reshape into (B, W, H, L)
    return hidden_data.reshape((__batch_dim, __layer_dim, __width_dim, __height_dim)).permute(0, 2, 3, 1)

# MASK #########################################################################

def mask_hidden_states(
    hidden_data: torch.Tensor, # (B, L, H)
    topk_num: int=128,
) -> torch.Tensor:
    # sanitize
    __k = min(topk_num, int(hidden_data.shape[-1]))
    # indices of the topk values
    __indices = hidden_data.abs().topk(__k, dim=-1, largest=True, sorted=False).indices
    # initialize the mask with False
    __mask = torch.zeros_like(hidden_data, dtype=torch.bool)
    # (B, L, H) mask of the topk values
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
    __data = (__data + torch.finfo().eps) ** gamma_val
    # map to point area
    return area_min + (area_max - area_min) * __data

# POSTPROCESS ##################################################################

def postprocess_token_cls(
    token_idx: int,
    token_dim: int,
) -> list:
    __token_idx = max(-1, min(token_dim, token_idx))
    # class 1 for the focused token(s) 0 for the rest
    __token_cls = [str(int(__i == token_idx)) for __i in range(token_dim)]
    # average on all the tokens when the idx is negative
    return token_dim * ['1'] if (token_idx < 0) else __token_cls
