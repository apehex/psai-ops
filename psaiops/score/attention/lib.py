import torch

import psaiops.common.model
import psaiops.common.tokenizer

# COMPUTE ########################################################################

def compute_attention_weights(
    model_obj: object,
    token_obj: torch.Tensor,
) -> torch.Tensor:
    # process the full sequence
    with torch.no_grad():
        __outputs = model_obj(
            input_ids=token_obj,
            output_attentions=True,
            return_dict=True)
    # parse the outputs
    return torch.stack(__outputs.attentions, dim=0)

# REDUCE #######################################################################

def reduce_attention_weights(
    attention_data: torch.Tensor,
    token_idx: int, # -1 => avg over all tokens
    layer_idx: int, # -1 => avg over layers
    head_idx: int, # -1 => avg over heads
    input_dim: int,
) -> torch.Tensor:
    # parse
    __layer_dim, __batch_dim, __head_dim, __output_dim, __output_dim = tuple(attention_data.shape) # L, B, H, T, T
    __layer_idx = min(layer_idx, __layer_dim - 1)
    __head_idx = min(head_idx, __head_dim - 1)
    __token_idx = min(token_idx, __output_dim - input_dim - 1) # T = I + O
    # select the relevant data along each axis
    __layer_slice = slice(None) if (__layer_idx < 0) else slice(__layer_idx, __layer_idx + 1)
    __sample_slice = slice(None)
    __head_slice = slice(None) if (__head_idx < 0) else slice(__head_idx, __head_idx + 1)
    __token_slice = slice(input_dim - 1, __output_dim) if (__token_idx < 0) else slice(input_dim + __token_idx - 1, input_dim + __token_idx)
    # filter the data
    __data = attention_data[__layer_slice, __sample_slice, __head_slice, __token_slice, slice(None)]
    # reduce all the axes but the last
    return __data.mean(dim=tuple(range(len(__data.shape) - 1)))

# FORMAT #########################################################################

def postprocess_attention_scores(
    attention_data: torch.Tensor, # (T,)
    input_dim: int,
    token_idx: int,
) -> list:
    __output_dim = int(attention_data.shape[-1])
    # isolate the scores of the input prompt
    __input_slice = slice(0, input_dim)
    # mask the token that were used to compute the scores
    __token_idx = min(token_idx, __output_dim - input_dim - 1) # T = I + O
    __output_range = list(range(__output_dim - input_dim)) if (__token_idx < 0) else [__token_idx]
    __output_mask = torch.BoolTensor([__i in __output_range for __i in range(__output_dim - input_dim)])
    # normalize the scores
    __input_scores = attention_data[__input_slice] / (attention_data[__input_slice].max() + 1e-5)
    # round to obtain integer labels from 0 to 100
    __input_scores = torch.round(100.0 * __input_scores, decimals=0).type(torch.int32)
    # the generated tokens are not scored
    __output_scores = torch.where(__output_mask, -1, 0).type(torch.int32)
    # native list of serialized integers
    return [str(__i) for __i in __input_scores.tolist() + __output_scores.tolist()] # (I,) + (O,) = (T,)

# COMPUTE ########################################################################

def score_tokens(
    prompt_str: str,
    token_num: int,
    topk_num: int,
    topp_num: float,
    token_idx: int,
    layer_idx: int,
    head_idx: int,
    device_str: str,
    model_obj: object,
    tokenizer_obj: object,
) -> list:
    # dictionary {'input_ids': _, 'attention_mask': _}
    __inputs = psaiops.common.tokenizer.preprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        prompt_str=prompt_str,
        device_str=device_str)
    # parse the inputs
    __input_dim = int(__inputs['input_ids'].shape[-1])
    # tensor (1, T)
    __outputs = psaiops.common.tokenizer.model.generate_token_ids(
        model_obj=model_obj,
        input_ids=__inputs['input_ids'],
        attention_mask=__inputs['attention_mask'],
        token_num=token_num,
        topk_num=topk_num,
        topp_num=topp_num)
    # tensor (L, S, H, T, T)
    __attentions = compute_attention_weights(
        model_obj=model_obj,
        token_obj=__outputs)
    # reduce the layer, sample, head and output token axes => tensor (T,)
    __scores = reduce_attention_weights(
        __attentions,
        token_idx=token_idx,
        layer_idx=layer_idx,
        head_idx=head_idx,
        input_dim=__input_dim)
    # translate the scores into integer labels
    __labels = postprocess_attention_scores(
        __scores,
        input_dim=__input_dim,
        token_idx=token_idx)
    # detokenize the IDs
    __tokens = psaiops.common.tokenizer.postprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        token_obj=__outputs)
    # match tokens and labels for the HighlightedText field
    return list(zip(__tokens, __labels))
