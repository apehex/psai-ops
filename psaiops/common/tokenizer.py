import torch
import transformers

# LOAD #########################################################################

def get_tokenizer(name: str, device: str='cpu'):
    return transformers.AutoTokenizer.from_pretrained(
        name,
        use_fast=True,
        dtype='auto',
        device_map=device)

# PREPROCESS #####################################################################

def preprocess_token_ids(
    tokenizer_obj: object,
    prompt_str: str,
    device_str: str='cpu'
) -> dict:
    # tokenize
    __data = tokenizer_obj(prompt_str, return_tensors='pt', padding='longest')
    # move to the main device
    return {__k: __v.to(device_str) for __k, __v in __data.items()}

def preprocess_token_str(
    tokenizer_obj: object,
    prompt_str: str,
) -> list:
    # tokenize
    __data = tokenizer_obj(prompt_str, return_offsets_mapping=True, add_special_tokens=False)
    # partition the original string (avoid escaping special characters)
    return [prompt_str[__s:__e] for (__s, __e) in __data['offset_mapping']]

# POSTPROCESS ####################################################################

def postprocess_token_ids(
    tokenizer_obj: object,
    token_arr: torch.Tensor,
) -> list:
    # remove the batch axis
    __indices = token_arr.squeeze().tolist()
    # back to token strings
    __tokens = tokenizer_obj.convert_ids_to_tokens(__indices)
    # normalize the tokens
    return [__t.replace(chr(0x0120), ' ').replace(chr(0x010a), '\n') for __t in __tokens]
