import os
import random

import gradio
import numpy
import torch
import matplotlib.pyplot

import psaiops.common.tokenizer as _tok
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

def sample_input_text(
    dataset_str: str,
    type_str: str,
    samples_arr: dict=_ui.SAMPLES,
) -> str:
    # exit if some values are missing
    if (not dataset_str) or (not type_str):
        return _ui.TUTO
    # return the documentation by default
    __dataset = samples_arr.get(dataset_str, {}).get(type_str, [_ui.TUTO])
    # return a single string
    return random.choice(__dataset)

# WINDOW #######################################################################

def update_window_range(
    current_val: float,
    indices_arr: object,
) -> dict:
    # exit if some values are missing
    if (current_val is None) or (indices_arr is None) or (len(indices_arr) == 0):
        return gradio.update()
    # take the generated tokens into account
    __max = max(1, int(indices_arr.shape[-1]))
    # keep the previous value if possible
    __val = min(int(current_val), __max)
    # return a gradio update dictionary
    return gradio.update(value=__val, maximum=__max)

# TOKENS #######################################################################

def update_tokens_state(
    prompt_str: str,
    export_str: str,
    tokenizer_obj: object,
) -> object:
    # exit if some values are missing
    if (prompt_str is None) or (tokenizer_obj is None):
        return None
    # list of token strings, without escaping
    __tokens = _tok.preprocess_token_str(
        tokenizer_obj=tokenizer_obj,
        prompt_str=prompt_str.strip(),)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__tokens, name='tokens.pt', path=export_str)
    # the token partition is used to highlight the sample token by token
    return __tokens

# INDICES ######################################################################

def update_indices_state(
    prompt_str: str,
    export_str: str,
    tokenizer_obj: object,
) -> object:
    # exit if some values are missing
    if (prompt_str is None) or (tokenizer_obj is None):
        return None
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

def update_logits_state(
    indices_arr: object,
    export_str: str,
    model_obj: object,
) -> object:
    # exit if some values are missing
    if (indices_arr is None) or (model_obj is None):
        return None
    # move the output back to the CPU
    __logits = _lib.compute_raw_logits(
        indices_arr=indices_arr.to(device=model_obj.device),
        model_obj=model_obj).cpu()
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__logits, name='logits.pt', path=export_str)
    # used to compute all the indicators from the critic LLM
    return __logits

# METRICS ######################################################################

def update_unicode_state(
    tokens_arr: list,
    export_str: str,
) -> object:
    # exit if some values are missing
    if (tokens_arr is None) or (len(tokens_arr) == 0):
        return None
    # one score per token (B, T)
    __unicodes = _lib.compute_unicode_metrics(
        tokens_arr=tokens_arr,)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__unicodes, name='unicodes.pt', path=export_str)
    # identify rare glyphs as LLM outputs
    return __unicodes

def update_rank_state(
    indices_arr: object,
    logits_arr: object,
    export_str: str,
) -> object:
    # exit if some values are missing
    if (indices_arr is None) or (len(indices_arr) == 0) or (logits_arr is None) or (len(logits_arr) == 0):
        return None
    # one score per token (B, T)
    __ranks = _lib.compute_rank_metrics(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__ranks, name='ranks.pt', path=export_str)
    # rank of each token in the LLM predictions, log-scaled
    return __ranks

def update_entropy_state(
    logits_arr: object,
    export_str: str,
) -> object:
    # exit if some values are missing
    if (logits_arr is None) or (len(logits_arr) == 0):
        return None
    # one score per token (B, T)
    __entropies = _lib.compute_entropy_metrics(
        logits_arr=logits_arr)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__entropies, name='entropies.pt', path=export_str)
    # measures the spread of the predictions probabilities, over the vocabulary
    return __entropies

def update_perplexity_state(
    indices_arr: object,
    logits_arr: object,
    export_str: str,
) -> object:
    # exit if some values are missing
    if (indices_arr is None) or (len(indices_arr) == 0) or (logits_arr is None) or (len(logits_arr) == 0):
        return None
    # one score per token (B, T)
    __perplexities = _lib.compute_perplexity_metrics(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__perplexities, name='perplexities.pt', path=export_str)
    # measures how surprising the whole neighbordhood of each token is
    return __perplexities

def update_surprisal_state(
    indices_arr: object,
    logits_arr: object,
    export_str: str,
) -> object:
    # exit if some values are missing
    if (indices_arr is None) or (len(indices_arr) == 0) or (logits_arr is None) or (len(logits_arr) == 0):
        return None
    # one score per token (B, T)
    __surprisals = _lib.compute_surprisal_metrics(
        indices_arr=indices_arr,
        logits_arr=logits_arr)
    # save the data to pre-fill the UI on startup
    if export_str: save_to_disk(__surprisals, name='surprisals.pt', path=export_str)
    # measures how surprising each token is, comparent to the model's predictions
    return __surprisals

# HIGHLIGHTS ###################################################################

def update_token_highlights(
    tokens_arr: list,
    unicode_arr: object,
    surprisal_arr: object,
    perplexity_arr: object,
    selection_arr: list,
    window_dim: float,
    export_str: str,
) -> list:
    # exit if some values are missing
    if (tokens_arr is None) or (len(tokens_arr) == 0) or (unicode_arr is None) or (len(unicode_arr) == 0) or (surprisal_arr is None) or (len(surprisal_arr) == 0) or (perplexity_arr is None) or (len(perplexity_arr) == 0) or (selection_arr is None) or (window_dim is None):
        return None
    # normalize and force an odd window size
    __window_dim = 2 * (int(window_dim) // 2) + 1
    # common arguments
    __args = {'axis_idx': -1, 'step_val': 0.25, 'rate_val': 10.0, 'neutral_val': 0.5}
    # downplay the scores near the begining because the model is lacking context
    __surprisal_arr = _lib.apply_time_ramp(data_arr=surprisal_arr, **__args) if (_ui.RAMPING in selection_arr) else surprisal_arr
    __perplexity_arr = _lib.apply_time_ramp(data_arr=perplexity_arr, **__args) if (_ui.RAMPING in selection_arr) else perplexity_arr
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
    # toggle the metrics according to the selection
    __token_cls = (
        [0.5 * torch.ones_like(unicode_arr, device=unicode_arr.device, dtype=unicode_arr.dtype)]
        + (_ui.UNICODE in selection_arr) * [unicode_arr]
        + (_ui.SURPRISAL in selection_arr) * [__surprisal_arr]
        + (_ui.PERPLEXITY in selection_arr) * [__perplexity_arr])
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

def update_metric_plots(
    unicode_arr: object,
    rank_arr: object,
    entropy_arr: object,
    surprisal_arr: object,
    perplexity_arr: object,
    selection_arr: list,
    window_dim: float,
) -> object:
    # exit if some values are missing
    if (unicode_arr is None) or (len(unicode_arr) == 0) or (rank_arr is None) or (len(rank_arr) == 0) or (entropy_arr is None) or (len(entropy_arr) == 0) or (surprisal_arr is None) or (len(surprisal_arr) == 0) or (perplexity_arr is None) or (len(perplexity_arr) == 0) or (selection_arr is None) or (window_dim is None):
        return None
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
    # combine all the metrics into a final score
    __yf = _lib.compute_probability_conflation(
        metrics_arr=[unicode_arr, __yp, __ys],
        axis_idx=-1)
    # keep only the first sample and rescale as a percentage
    __yu = 100.0 * unicode_arr.squeeze(dim=0).numpy()
    __yr = 100.0 * rank_arr.squeeze(dim=0).numpy()
    __ye = 100.0 * __ye.squeeze(dim=0).numpy()
    __yp = 100.0 * __yp.squeeze(dim=0).numpy()
    __ys = 100.0 * __ys.squeeze(dim=0).numpy()
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
