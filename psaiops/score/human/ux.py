import os
import random

import gradio
import numpy
import torch
import matplotlib.pyplot

import psaiops.common.tokenizer as _tok
import psaiops.common.utils as _utils
import psaiops.score.human.lib as _lib
import psaiops.score.human.ui as _ui

# META #########################################################################

_PATH = os.path.dirname(__file__)

# IO ###########################################################################

def save_to_disk(data: object, name: str, path: str=os.path.join(_PATH, 'data', 'state')) -> None:
    torch.save(data, os.path.join(path, name))

# BUTTONS ######################################################################

def enable_button() -> dict:
    return gradio.update(interactive=True)

def disable_button() -> dict:
    return gradio.update(interactive=False)

# SAMPLES ######################################################################

@_utils.typecheck(inputs=True, outputs=True, returns=_ui.TUTO)
def sample_input_text(
    dataset_str: str,
    type_str: str,
    samples_arr: dict=_ui.SAMPLES,
) -> str:
    # return the documentation by default
    __dataset = samples_arr.get(dataset_str, {}).get(type_str, [_ui.TUTO])
    # return a single string
    return random.choice(__dataset)

# WINDOW #######################################################################

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_window_range(
    current_val: float,
    indices_arr: torch.Tensor,
) -> dict:
    # take the generated tokens into account
    __max = max(1, int(indices_arr.shape[-1]))
    # keep the previous value if possible
    __val = min(int(current_val), __max)
    # return a gradio update dictionary
    return gradio.update(value=__val, maximum=__max)

# TOKENS #######################################################################

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_tokens_state(
    prompt_str: str,
    export_str: str,
    tokenizer_obj: torch.Tensor,
) -> list:
    # list of token strings, without escaping
    __tokens = _tok.preprocess_token_str(
        tokenizer_obj=tokenizer_obj,
        prompt_str=prompt_str.strip(),)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__tokens, name='tokens.pt', path=export_str)
    # the token partition is used to highlight the sample token by token
    return __tokens

# INDICES ######################################################################

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_indices_state(
    prompt_str: str,
    export_str: str,
    tokenizer_obj: torch.Tensor,
) -> torch.Tensor:
    # dictionary {'input_ids': _, 'attention_mask': _}
    __inputs = _tok.preprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        prompt_str=prompt_str.strip(),
        device_str='cpu')
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__inputs['input_ids'], name='indices.pt', path=export_str)
    # discard the mask, which is all ones
    return __inputs['input_ids'].cpu()

# LOGITS #######################################################################

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_logits_state(
    indices_arr: torch.Tensor,
    export_str: str,
    model_obj: object,
) -> torch.Tensor:
    # move the output back to the CPU
    __logits = _lib.compute_raw_logits(
        indices_arr=indices_arr.to(device=model_obj.device),
        model_obj=model_obj).cpu()
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__logits, name='logits.pt', path=export_str)
    # used to compute all the indicators from the critic LLM
    return __logits

# METRICS ######################################################################

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_unicode_state(
    tokens_arr: list,
    export_str: str,
) -> torch.Tensor:
    # one score per token (B, T)
    __unicodes = _lib.compute_unicode_metrics(
        tokens_arr=tokens_arr,)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__unicodes, name='unicodes.pt', path=export_str)
    # identify rare glyphs as LLM outputs
    return __unicodes

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_rank_state(
    indices_arr: torch.Tensor,
    logits_arr: torch.Tensor,
    export_str: str,
) -> torch.Tensor:
    # one score per token (B, T)
    __ranks = _lib.compute_rank_metrics(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__ranks, name='ranks.pt', path=export_str)
    # rank of each token in the LLM predictions, log-scaled
    return __ranks

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_entropy_state(
    logits_arr: torch.Tensor,
    export_str: str,
) -> torch.Tensor:
    # one score per token (B, T)
    __entropies = _lib.compute_entropy_metrics(
        logits_arr=logits_arr)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__entropies, name='entropies.pt', path=export_str)
    # measures the spread of the predictions probabilities, over the vocabulary
    return __entropies

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_perplexity_state(
    indices_arr: torch.Tensor,
    logits_arr: torch.Tensor,
    export_str: str,
) -> torch.Tensor:
    # one score per token (B, T)
    __perplexities = _lib.compute_perplexity_metrics(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__perplexities, name='perplexities.pt', path=export_str)
    # measures how surprising the whole neighbordhood of each token is
    return __perplexities

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_surprisal_state(
    indices_arr: torch.Tensor,
    logits_arr: torch.Tensor,
    export_str: str,
) -> torch.Tensor:
    # one score per token (B, T)
    __surprisals = _lib.compute_surprisal_metrics(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__surprisals, name='surprisals.pt', path=export_str)
    # measures how surprising each token is, comparent to the model's predictions
    return __surprisals

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_sampling_state(
    indices_arr: torch.Tensor,
    logits_arr: torch.Tensor,
    export_str: str,
    topk_val: float=0.0,
    topp_val: float=1.0,
    repp_val: float=1.0,
    temp_val: float=1.0,
) -> torch.Tensor:
    # one score per token (B, T)
    __samplings = _lib.compute_sampling_metrics(
        indices_arr=indices_arr,
        logits_arr=logits_arr,
        topk_val=topk_val,
        topp_val=topp_val,
        repp_val=repp_val,
        temp_val=temp_val)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__samplings, name='samplings.pt', path=export_str)
    # measures how surprising each token is, comparent to the model's predictions
    return __samplings

# HIGHLIGHTS ###################################################################

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_token_highlights(
    tokens_arr: list,
    unicode_arr: torch.Tensor,
    surprisal_arr: torch.Tensor,
    perplexity_arr: torch.Tensor,
    sampling_arr: torch.Tensor,
    selection_arr: list,
    window_dim: float,
    export_str: str,
) -> list:
    # normalize and force an odd window size
    __window_dim = 2 * (int(window_dim) // 2) + 1
    # common arguments
    __args = {'axis_idx': -1, 'step_val': 0.25, 'rate_val': 10.0, 'neutral_val': 0.5}
    # downplay the scores near the begining because the model is lacking context
    __surprisal_arr = _lib.apply_time_ramp(data_arr=surprisal_arr, **__args) if (_ui.RAMPING in selection_arr) else surprisal_arr
    __perplexity_arr = _lib.apply_time_ramp(data_arr=perplexity_arr, **__args) if (_ui.RAMPING in selection_arr) else perplexity_arr
    __sampling_arr = _lib.apply_time_ramp(data_arr=sampling_arr, **__args) if (_ui.RAMPING in selection_arr) else sampling_arr
    # compute the scores on a sliding window
    __surprisal_arr = _lib.compute_topk_pooling(
        data_arr=__surprisal_arr,
        pool_dim=__window_dim,
        topk_dim=(__window_dim + 1) // 2,
        axis_idx=-1)
    __perplexity_arr = _lib.compute_average_pooling(
        data_arr=__perplexity_arr,
        pool_dim=max(9, __window_dim),
        axis_idx=-1)
    __sampling_arr = _lib.compute_topk_pooling(
        data_arr=__sampling_arr,
        pool_dim=__window_dim,
        topk_dim=(__window_dim + 1) // 2,
        axis_idx=-1)
    # toggle the metrics according to the selection
    __token_cls = (
        [0.5 * torch.ones_like(unicode_arr, device=unicode_arr.device, dtype=unicode_arr.dtype)]
        + (_ui.UNICODE in selection_arr) * [unicode_arr]
        + (_ui.SURPRISAL in selection_arr) * [__surprisal_arr]
        + (_ui.PERPLEXITY in selection_arr) * [__perplexity_arr]
        + (_ui.SAMPLING in selection_arr) * [__sampling_arr])
    # there is an extra neutral score in case all other have been toggled off (so that the conflation doesn't error)
    __token_cls = _lib.compute_probability_conflation(
        metrics_arr=__token_cls,
        axis_idx=-1)
    # scale into a [0; 100] label
    __token_cls = _lib.postprocess_score_cls(
        score_arr=__token_cls,
        scale_val=100.0)
    # color each token according to its rank in the LLM's predictions
    __labels = list(zip(tokens_arr, __token_cls))
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__labels, name='labels.pt', path=export_str)
    # list of (token, label), where each label is mapped to a color
    return __labels

# PLOTS ########################################################################

@_utils.typecheck(inputs=True, outputs=True, returns=gradio.update())
def update_metric_plots(
    unicode_arr: torch.Tensor,
    rank_arr: torch.Tensor,
    entropy_arr: torch.Tensor,
    surprisal_arr: torch.Tensor,
    perplexity_arr: torch.Tensor,
    sampling_arr: torch.Tensor,
    selection_arr: list,
    window_dim: float,
) -> object:
    # normalize and force an odd window size
    __window_dim = 2 * (int(window_dim) // 2) + 1
    # time ramp to downplay the first few tokens because they have no context
    __yt = _lib.sigmoid_ramp(
        time_dim=int(unicode_arr.shape[-1]),
        step_val=0.25,
        rate_val=10.0)
    # smooth the curves
    __ye = _lib.compute_average_pooling(
        data_arr=entropy_arr,
        pool_dim=__window_dim,
        axis_idx=-1)
    __yp = _lib.compute_average_pooling(
        data_arr=perplexity_arr,
        pool_dim=__window_dim,
        axis_idx=-1)
    __ys = _lib.compute_topk_pooling(
        data_arr=surprisal_arr,
        pool_dim=__window_dim,
        topk_dim=(__window_dim + 1) // 2,
        axis_idx=-1)
    __ya = _lib.compute_topk_pooling(
        data_arr=sampling_arr,
        pool_dim=__window_dim,
        topk_dim=(__window_dim + 1) // 2,
        axis_idx=-1)
    # combine all the metrics into a final score
    __yf = _lib.compute_probability_conflation(
        metrics_arr=[unicode_arr, __yp, __ys, __ya],
        axis_idx=-1)
    # keep only the first sample and rescale as a percentage
    __yu = 100.0 * unicode_arr.squeeze(dim=0).numpy()
    __yr = 100.0 * rank_arr.squeeze(dim=0).numpy()
    __ye = 100.0 * __ye.squeeze(dim=0).numpy()
    __yp = 100.0 * __yp.squeeze(dim=0).numpy()
    __ys = 100.0 * __ys.squeeze(dim=0).numpy()
    __ya = 100.0 * __ya.squeeze(dim=0).numpy()
    __yf = 100.0 * __yf.squeeze(dim=0).numpy()
    __yt = 100.0 * __yt.squeeze(dim=0).numpy()
    # match the metrics with their token position
    __x = numpy.arange(len(__yr))
    # prepare a wide canvas
    __figure = matplotlib.pyplot.figure(figsize=(16, 4), dpi=120)
    __axes = __figure.add_subplot(1, 1, 1)
    __axes.plot(__x, __yf, linestyle='-', label='final', color='#FF5555') # vibrant coral
    # toggle each curve on / off
    if _ui.RAMPING in selection_arr:
        __axes.plot(__x, __yt, linestyle='--', label='unicode', color='#555500') # olive leaf
    if _ui.UNICODE in selection_arr:
        __axes.plot(__x, __yu, linestyle='--', label='unicode', color='#005555') # slate grey
    if _ui.PERPLEXITY in selection_arr:
        __axes.plot(__x, __yp, linestyle='--', label='perplexity', color='#5555FF') # full blue
    if _ui.SURPRISAL in selection_arr:
        __axes.plot(__x, __ys, linestyle='--', label='surprisal', color='#550055') # deep purple
    if _ui.SAMPLING in selection_arr:
        __axes.plot(__x, __ys, linestyle='--', label='sampling', color='#55FF55') # electric green
    if _ui.INTERMEDIATE in selection_arr:
        __axes.plot(__x, __yr, linestyle=':', label='rank', color='#442222') # expresso
        __axes.plot(__x, __ye, linestyle=':', label='entropy', color='#222244') # indigo
    # display the legend and remove the extra padding
    __axes.legend()
    __figure.tight_layout()
    # remove the figure for the pyplot register for garbage collection
    matplotlib.pyplot.close(__figure)
    # update each component => (highlight, plot) states
    return __figure
