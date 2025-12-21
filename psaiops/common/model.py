import functools

import torch

import deformers.models.openai.gptoss

# LOAD #########################################################################

@functools.lru_cache(maxsize=1)
def get_model(name: str, device: str='cpu'):
    __model = deformers.models.openai.gptoss.GptOssForCausalInference.from_pretrained(
        name,
        dtype='auto',
        device_map=device)
    # toggle the inference mode (not training)
    __model.eval()
    # transformers model
    return __model

# GENERATE #######################################################################

@functools.lru_cache(maxsize=32)
def generate_token_ids(
    model_obj: object,
    input_ids: torch.Tensor,
    token_num: int,
    topk_num: int = 4,
    topp_num: float = 0.9,
    attention_mask: torch.Tensor=None,
) -> torch.Tensor:
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
            output_hidden_states=False,
            output_attentions=False,
            output_scores=False,
            # early_stopping=True,
            use_cache=True)
    # full sequence
    return __outputs.sequences # (1, T)
