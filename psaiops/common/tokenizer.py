import functools

import torch
import transformers

# LOAD #########################################################################

@functools.lru_cache(maxsize=4)
def get_tokenizer(name: str, device: str='cpu'):
    return transformers.AutoTokenizer.from_pretrained(
        name,
        use_fast=True,
        dtype='auto',
        device_map=device)

# PREPROCESS #####################################################################

@functools.lru_cache(maxsize=32)
def preprocess_token_ids(
    tokenizer_obj: object,
    prompt_str: str,
    device_str: str='cpu'
) -> dict:
    # tokenize
    __inputs = tokenizer_obj(prompt_str, return_tensors='pt')
    # move to the main device
    return {__k: __v.to(device_str) for __k, __v in __inputs.items()}

# POSTPROCESS ####################################################################

@functools.lru_cache(maxsize=32)
def postprocess_token_ids(
    tokenizer_obj: object,
    token_data: torch.Tensor,
) -> list:
    # remove the batch axis
    __indices = token_data.squeeze().tolist()
    # back to token strings
    __tokens = tokenizer_obj.convert_ids_to_tokens(__indices)
    # normalize the tokens
    return [__t.replace(chr(0x0120), ' ').replace(chr(0x010a), '\n') for __t in __tokens]
