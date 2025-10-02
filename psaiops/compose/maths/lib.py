import functools

import torch
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
