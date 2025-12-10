import functools

import torch
import torch.nn.modules

import mlable.shapes
import psaiops.common.tokenizer

# HOOK #########################################################################

def capture_hidden_activation(
    module: torch.nn.modules.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    index: int,
    captured: dict,
) -> None:
    captured[index] = outputs # (B, S, E)

# MASKS ########################################################################

def compute_sequence_mask(
    tokens: torch.Tensor, # (B, S)
    # masks: torch.Tensor, # (B, S)
) -> torch.Tensor:
    __shape = mlable.shapes.divide(tokens.shape, axis=0, factor=2, insert=True)
    # group the samples two by two
    __data = tokens.reshape(__shape)
    # compare each sample with its neighbor
    __masks = __data[:, :1]  != __data[:, 1:]
    # apply the same mask to both samples
    return __masks.expand(__shape).reshape(tokens.shape)

# REDUCTION ####################################################################

def compute_delta_activation(
    data: torch.Tensor, # (B, S, E)
    masks: torch.Tensor, # (B, S,)
    signs: torch.Tensor, # (B,)
    keepdim: bool=True,
) -> torch.Tensor:
    __dtype = data.dtype
    __device = data.device
    __dim0, __dim1, __dim2 = tuple(data.shape)
    # sign each sample along the batch axis
    __shape = tuple(mlable.shapes.filter(data.shape, axes=[0]))
    __signs = signs.to(dtype=__dtype, device=__device).view(__shape)
    # combine along the batch axis to keep the shortest mask on the sequence axis
    __shape = tuple(mlable.shapes.filter(data.shape, axes=[0, 1]))
    __masks = masks.to(dtype=__dtype, device=__device).view(__shape)
    # mean factor: half the signs size along the batch axis and the number of positions kept along the sequence axis
    __factor = (0.5 * float(__dim0) * __masks.sum(dim=1, keepdim=True)).clamp(min=1.0)
    # take the difference along the batch axis and the average along the sequence axis
    return (data * __signs * __masks / __factor).sum(dim=[0, 1], keepdim=keepdim)

# DELTA ########################################################################

def add_delta_activation(
    module: torch.nn.modules.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    delta: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    # expand the single feature axis of the delta
    __shape = mlable.shapes.filter(outputs.shape, axes=[-1])
    # rescale the delta
    return alpha * outputs + beta * delta.view(__shape)

# MAIN #########################################################################

def steer_model_output(
    positive_str: str,
    negative_str: str,
    prompt_str: str,
    positive_rate: float,
    negative_rate: float,
    prompt_rate: float,
    token_num: int,
    topk_num: int,
    topp_num: float,
    layer_idx: int,
    device_str: str,
    model_obj: object,
    tokenizer_obj: object,
) -> str:
    # parse & sanitize
    __prompt0 = positive_str.strip()
    __prompt1 = negative_str.strip()
    __prompt2 = prompt_str.strip()
    __alpha0 = max(0.0, float(positive_rate))
    __alpha1 = max(0.0, float(negative_rate))
    __alpha2 = max(0.0, float(prompt_rate))
    __count = max(1, int(token_num))
    __topk = max(1, int(topk_num))
    __topp = max(0.0, float(topp_num))
    __index = max(0, int(layer_idx))
    # store hidden states
    __captured = {}
    # stop if inputs are missing
    if not (__prompt0 and __prompt1 and __prompt2):
        return ''
    # tokenize the 2 prompts and pad to same length
    __inputs = psaiops.common.tokenizer.preprocess_token_ids(tokenizer=tokenizer_obj, prompts=(__prompt0, __prompt1), device=device_str)
    # forward hook to capture output hidden state
    __hook = functools.partial(capture_hidden_activation, index=__index, captured=__captured)
    # attach to the model
    __handle = model_obj.model.layers[__index].register_forward_hook(__hook)
    with torch.no_grad():
        # inference mode
        model_obj.eval().to(device_str)
        # prefill with a single forward
        __outputs = model_obj(**__inputs, use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True)
    # stop capturing activations
    __handle.remove()
    # select only the positions where the tokens differ
    __masks = compute_sequence_mask(tokens=__inputs['input_ids'])
    # activation delta at layer L
    __delta = compute_delta_activation(data=__captured[__index], masks=__masks, signs=torch.Tensor([1, -1]), keepdim=False)
    # add the delta on every forward pass
    __hook = functools.partial(add_delta_activation, alpha=__alpha2, beta=0.5 * (__alpha0 + __alpha1), delta=__delta)
    # attach to the model
    __handle = model_obj.model.layers[__index].register_forward_hook(__hook)
    # now process the user input
    __inputs = psaiops.common.tokenizer.preprocess_token_ids(tokenizer=tokenizer_obj, prompts=(prompt_str,), device=device_str)
    # generate the new with tampered activations
    with torch.no_grad():
        __outputs = model_obj.generate(
            **__inputs,
            max_new_tokens=__count,
            do_sample=(0.0 < __topp < 1.0) or (__topk > 0),
            top_k=__topk if (__topk > 0) else None,
            top_p=__topp if (0.0 < __topp <= 1.0) else None,
            return_dict_in_generate=True,
            output_hidden_states=False,
            output_attentions=False,
            output_scores=False,
            use_cache=True)
    # stop altering the activations
    __handle.remove()
    # single string
    return tokenizer_obj.decode(__outputs.sequences[0], skip_special_tokens=True)
