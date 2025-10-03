import functools

import torch
import transformers

import deformers.models.openai.gptoss

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

# GENERATE #######################################################################

def generate_token_ids(
    model_obj: object,
    input_args: dict,
    token_num: int,
    topk_num: int = 4,
    topp_num: float = 0.9,
) -> torch.Tensor:
    # generate completion
    with torch.no_grad():
        __outputs = model_obj.generate(
            **input_args,
            max_new_tokens=token_num,
            do_sample=(0.0 < topp_num < 1.0) or (topk_num > 0),
            top_k=topk_num if (topk_num > 0) else None,
            top_p=topp_num if (0.0 < topp_num < 1.0) else None,
            return_dict_in_generate=True,
            output_hidden_states=False,
            output_attentions=False,
            output_scores=False,
            # early_stopping=True,
            use_cache=True)
    # full sequence
    return __outputs.sequences # (1, T)
