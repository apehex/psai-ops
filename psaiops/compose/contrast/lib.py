import functools

import torch
import torch.nn
import torch.nn.modules
import transformers

import deformers.models.openai.gptoss
import mlable.shapes

# LOAD #########################################################################

@functools.lru_cache(maxsize=4)
def get_tokenizer(name: str, device: str='cpu'):
    return transformers.AutoTokenizer.from_pretrained(
        name,
        use_fast=True,
        dtype='auto',
        device_map=device)

@functools.lru_cache(maxsize=2)
def get_model(name: str, device: str='cpu'):
    __model = deformers.models.openai.gptoss.GptOssForCausalInference.from_pretrained(
        name,
        dtype='auto',
        device_map=device)
    # toggle the inference mode (not training)
    __model.eval()
    # transformers model
    return __model

# PREPROCESS #####################################################################

@functools.lru_cache(maxsize=4)
def preprocess_token_ids(
    tokenizer: object,
    prompts: list,
    device: str='cpu'
) -> dict:
    # tokenize
    __inputs = tokenizer(prompts, return_tensors='pt', padding=True)
    # move to the main device
    return {__k: __v.to(device) for __k, __v in __inputs.items()}

# HOOK #########################################################################

def capture_hidden_activation(
    module: torch.nn.modules.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    index: int,
    captured: dict,
) -> None:
    captured[index] = outputs # (B, S, E)

# REDUCTION ####################################################################

def compute_delta_activation(
    data: torch.Tensor, # (B, S, E)
    masks: torch.Tensor, # (B, S)
    signs: torch.Tensor, # (B,)
    keepdim: bool=True,
) -> torch.Tensor:
    __dtype = data.dtype
    __device = data.device
    # sign each sample along the batch axis
    __shape = tuple(mlable.shapes.filter(data.shape, axes=[0]))
    __signs = signs.to(dtype=__dtype, device=__device).view(__shape)
    # combine along the batch axis to keep the shortest mask on the sequence axis
    __shape = tuple(mlable.shapes.filter(data.shape, axes=[1]))
    __masks = torch.prod(masks, dim=0, keepdim=True).to(dtype=__dtype, device=__device).view(__shape)
    # mean factor: half the signs size along the batch axis and the number of positions kept along the sequence axis
    __factor = (0.5 * __signs.abs().sum() * __masks.sum()).clamp(min=1e-8)
    # take the difference along the batch axis and the average along the sequence axis
    return (data * __signs * __masks).sum(dim=[0, 1], keepdim=keepdim) / __factor

# DELTA ########################################################################

def add_delta_activation(
    module: torch.nn.modules.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    delta: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    # expand the single feature axis of the delta
    __shape = mlable.shapes.filter(outputs.shape, axes=[-1])
    # rescale the delta
    return outputs + alpha * delta.view(__shape)

# MAIN #########################################################################

def steer_model_output(
    positive_str: str,
    negative_str: str,
    prompt_str: str,
    token_num: int,
    topk_num: int,
    topp_num: float,
    alpha_num: float,
    layer_idx: int,
    device_str: str,
    model_obj: object,
    tokenizer_obj: object,
) -> str:
    # parse
    __index = max(0, int(layer_idx))
    __alpha = max(0.0, float(alpha_num))
    __limit = max(1, int(token_num))
    __topk = max(1, int(topk_num))
    __topp = max(0.0, float(topp_num))
    # store hidden states
    __captured = {}
    # tokenize the 2 prompts and pad to same length
    __inputs = preprocess_token_ids(tokenizer=tokenizer_obj, prompts=(positive_str, negative_str), device=device_str)
    # forward hook to capture output hidden state
    __hook = functools.partial(capture_hidden_activation, index=__index, captured=__captured)
    # attach to the model
    __handle = model_obj.model.layers[__index].register_forward_hook(__hook)
    with torch.no_grad():
        # inference mode
        model_obj.eval().to(device_str)
        # prefill with a single forward
        __outputs = model(**__inputs, use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True)
    # stop capturing activations
    __handle.remove()
    # activation delta at layer L
    __delta = compute_delta_activation(data=__captured[__index], masks=__inputs['attention_mask'], signs=torch.Tensor([1, -1]), keepdim=False)
    # add the delta on every forward pass
    __hook = functools.partial(add_delta_activation, alpha=__alpha, delta=__delta)
    # attach to the model
    __handle = model_obj.model.layers[index].register_forward_hook(__hook)
    # now process the user input
    __inputs = preprocess_token_ids(tokenizer=tokenizer_obj, prompts=(prompt_str,), device=device_str)
    # generate the new with tampered activations
    with torch.no_grad():
        __outputs = model_obj.generate(
            **__inputs,
            max_new_tokens=__limit,
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
