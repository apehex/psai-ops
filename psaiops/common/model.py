import functools

import torch
import torch.nn

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

# PREFIX #######################################################################

def get_prefix(
    parent_obj: object,
    layer_num: int
) -> object:
    # init from the config
    __child = parent_obj.__class__(parent_obj.config)
    # share decoder core
    __child.model.embed_tokens = parent_obj.model.embed_tokens
    __child.model.norm = parent_obj.model.norm
    # share prefix layers (same objects)
    __child.model.layers = torch.nn.ModuleList(parent_obj.model.layers[:layer_num])
    # keep LM head
    __child.lm_head = parent_obj.lm_head
    # config hygiene
    __child.model.config.num_hiddelayer_num = layer_num
    __child.config.num_hiddelayer_num = layer_num
    # layer types
    if getattr(__child.config, "layer_types", None) is not None:
        __child.config.layer_types = __child.config.layer_types[:layer_num]
        __child.model.config.layer_types = __child.config.layer_types
    # wrapper with the first N hidden layers, pointing at the parent weights
    return __child

# GENERATE #####################################################################

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
