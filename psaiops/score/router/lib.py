import functools

import torch

# COMPUTE ########################################################################

def compute_router_weights(
    model_obj: object,
    token_data: torch.Tensor,
) -> torch.Tensor:
    # process the full sequence
    with torch.no_grad():
        __outputs = model_obj(
            input_ids=token_data,
            output_attentions=False,
            output_router_logits=True,
            return_dict=True)
    # stack all the layer outputs L * (T, E) => (L, T, E)
    __logits = torch.stack(__outputs.router_logits, dim=0)
    # turn the logits into expert probabilities
    return torch.softmax(__logits, dim=-1)

# REDUCE #######################################################################

def reduce_router_weights(
    router_data: torch.Tensor,
    token_idx: int, # -1 => avg over all tokens
) -> torch.Tensor:
    # parse
    __layer_dim, __token_dim, __expert_dim = tuple(router_data.shape) # L, T, E
    __token_idx = min(token_idx, __token_dim - 1)
    # select the relevant data along each axis
    __token_slice = slice(0, __token_dim) if (__token_idx < 0) else slice(__token_idx, __token_idx + 1)
    # filter the data
    __data = router_data[slice(None), __token_slice, slice(None)]
    # reduce all the axes but the last
    return __data.mean(dim=1, keepdim=False)

# FORMAT #########################################################################

def postprocess_router_weights(
    router_data: torch.Tensor, # (L, E)
) -> list:
    # the averaging over tokens may have broken the scaling
    __probs = torch.softmax(router_data, dim=-1)
    # enforce the output range [0; 1] with 1 included
    return __probs / __probs.amax(dim=-1, keepdim=True)

# POSTPROCESS ####################################################################

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
