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

# COMPUTE ########################################################################

def compute_router_weights(
    model_obj: object,
    token_obj: torch.Tensor,
) -> torch.Tensor:
    # process the full sequence
    with torch.no_grad():
        __outputs = model_obj(
            input_ids=token_obj,
            output_attentions=False,
            output_router_logits=True,
            return_dict=True)
    # stack all the layer outputs L * (B, T, E) => (L, B, T, E)
    __logits = torch.stack(__outputs.router_logits, dim=0)
    # turn the logits into expert probabilities
    return torch.softmax(__logits, dim=-1)

# REDUCE #######################################################################

def reduce_router_weights(
    router_data: torch.Tensor,
    token_idx: int, # -1 => avg over all tokens
) -> torch.Tensor:
    # parse
    __layer_dim, __batch_dim, __token_dim, __expert_dim = tuple(router_data.shape) # L, B, T, E
    __token_idx = min(token_idx, __token_dim - 1)
    # select the relevant data along each axis
    __token_slice = slice(0, __token_dim) if (__token_idx < 0) else slice(__token_idx, __token_idx + 1)
    # filter the data
    __data = router_data[slice(None), slice(None), __token_slice, slice(None)]
    # reduce all the axes but the last
    return __data.mean(dim=(1, 2), keepdim=False)

# FORMAT #########################################################################

def postprocess_router_weights(
    router_data: torch.Tensor, # (L, E)
) -> list:
    return 100.0 * torch.softmax(router_data, dim=-1)

# POSTPROCESS ####################################################################

def postprocess_token_ids(
    tokenizer_obj: object,
    token_obj: torch.Tensor,
) -> list:
    # remove the batch axis
    __indices = token_obj.squeeze().tolist()
    # back to token strings
    __tokens = tokenizer_obj.convert_ids_to_tokens(__indices)
    # normalize the tokens
    return [__t.replace(chr(0x0120), ' ').replace(chr(0x010a), '\n') for __t in __tokens]
