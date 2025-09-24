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
    tokenizer_obj: object,
    prompt_str: str,
    device_str: str='cpu'
) -> dict:
    # tokenize
    __inputs = tokenizer_obj(prompt_str, return_tensors='pt')
    # move to the main device
    return {__k: __v.to(device_str) for __k, __v in __inputs.items()}

# REDUCTION ####################################################################

def masked_mean(
    data: torch.Tensor,
    mask: torch.Tensor,
    axis: int=-2,
    keepdim: bool=True,
) -> torch.Tensor:
    # keep only the sequence axis
    __shape = tuple(mlable.shapes.filter(data.shape, axes=[axis]))
    # match the shape of the input
    __mask = mask.to(dtype=data.dtype, device=data.device).view(__shape)
    # average over the sequence axis
    return (data * __mask).sum(dim=axis, keepdim=keepdim) / __mask.sum().clamp(min=1e-8)

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
