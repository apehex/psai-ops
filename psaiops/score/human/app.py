import copy
import functools
import json
import os
import random

import gradio
import numpy
import torch
import torch.cuda
import matplotlib.pyplot

import psaiops.common.model
import psaiops.common.style
import psaiops.common.tokenizer
import psaiops.score.human.lib

# META #########################################################################

_PATH = os.path.dirname(__file__)
_LEGEND = {'AI': '0', 'human': '100',}

MODELS = ['qwen/qwen3.5-9b', 'qwen/qwen3.5-27b']

TITLE = '''De-Generate 🔎 🤖'''
INTRO = list((__t, _LEGEND.get(__t, '50')) for __t in '''Detect AI generated text sections and tell them apart from human written text, using an open source LLM as critic.'''.split(' '))
TUTO = open(os.path.join(_PATH, 'docs', 'readme.md'), 'r').read()
DOCS = open(os.path.join(_PATH, 'docs', 'scoring.md'), 'r').read()

# SAMPLES ######################################################################

SAMPLES = {
    'degen': {
        'readme': TUTO,
        'documentation': DOCS,},
    'hc3': json.load(open(os.path.join(_PATH, 'data', 'samples', 'hc3.json'), 'r')),
    'known': json.load(open(os.path.join(_PATH, 'data', 'samples', 'known.json'), 'r')),
    'system': json.load(open(os.path.join(_PATH, 'data', 'samples', 'system.json'), 'r')),
    'trace': json.load(open(os.path.join(_PATH, 'data', 'samples', 'trace.json'), 'r')),}

# ENUMS ########################################################################

# metric types
RAMPING = 1
UNICODE = 2
SURPRISAL = 4
PERPLEXITY = 8
INTERMEDIATE = 16

# IO ###########################################################################

def save_to_disk(data: object, name: str, path: str=os.path.join(_PATH, 'data', 'state')) -> None:
    torch.save(data, os.path.join(path, name))

def load_from_disk(name: str, path: str=os.path.join(_PATH, 'data', 'state')) -> object:
    __path = os.path.join(path, name)
    return torch.load(__path) if os.path.exists(__path) else None

# COLORS #######################################################################

def create_selection_cmap() -> dict:
    return {
        '0': '#000000',
        '1': '#004444',
        '2': '#444400',
        '3': '#440044',}

def create_score_cmap() -> dict:
    return {
        str(__i): '#{:02x}{:02x}00'.format(
            int(2.55 * 2 * max(0, 50 - __i)), # red: decreasing prob of LLM from 0 to 50
            int(2.55 * 2 * max(0, __i - 50))) # green: increasing prob of human from 50 to 100
        for __i in range(101)}

# INTRO ########################################################################

def create_text_block(text: str) -> dict:
    __text = gradio.Markdown(text, line_breaks=True, latex_delimiters=[{'left': '$$', 'right': '$$', 'display': True}, {'left': '$', 'right': '$', 'display': False}])
    return {'text_block': __text}

# MODEL ########################################################################

def create_model_block(models: list=MODELS) -> dict:
    __model = gradio.Dropdown(label='Model', value=models[0], choices=models, scale=1, allow_custom_value=False, multiselect=False, interactive=True) # 'openai/gpt-oss-120b'
    __load = gradio.Button('Load', variant='primary', size='lg', scale=1, interactive=False)
    return {
        'model_block': __model,
        'load_block': __load,}

# SAMPLING #####################################################################

def create_sampling_block() -> dict:
    __reps = gradio.Slider(label='Penalty', value=1.1, minimum=0.0, maximum=2.0, step=0.05, scale=1, interactive=True)
    __temp = gradio.Slider(label='Temperature', value=0.8, minimum=0.0, maximum=2.0, step=0.05, scale=1, interactive=True)
    __topk = gradio.Slider(label='Top K', value=24, minimum=1, maximum=2048, step=1, scale=1, interactive=True)
    __topp = gradio.Slider(label='Top P', value=0.95, minimum=0.0, maximum=1.0, step=0.01, scale=1, interactive=True)
    return {
        'reps_block': __reps,
        'temp_block': __temp,
        'topk_block': __topk,
        'topp_block': __topp,}

# DATASETS #####################################################################

def create_dataset_block(options: list, prefix: str='') -> dict:
    return {
        prefix + __o.lower() + '_block': gradio.Button(__o, variant='primary', size='lg', scale=1, interactive=True)
        for __o in options}

# INPUTS #######################################################################

def create_inputs_block(label: str='Prompt', prefix: str='', value: str='') -> dict:
    __input = gradio.Textbox(label=label, value=value, placeholder='A text sample to score.', lines=4, scale=1, interactive=True)
    return {prefix + 'input_block': __input}

# PLOTS ########################################################################

def create_plot_block(label: str='Plot', prefix: str='') -> dict:
    __plot = gradio.Plot(label=label, scale=1)
    return {prefix + 'plot_block': __plot,}

# HIGHLIGHT ####################################################################

def create_highlight_block(label: str='Score', prefix: str='', value: list=[], cmap: dict=create_selection_cmap(), show_label: bool=True) -> dict:
    __highlight = gradio.HighlightedText(label=label, value=value, scale=1, interactive=False, combine_adjacent=False, show_legend=False, show_inline_category=False, show_label=show_label, color_map=cmap, elem_classes='white-text')
    return {prefix + 'highlight_block': __highlight}

# REDUCTION ####################################################################

def create_selection_block(label: str='Selection', prefix: str='') -> dict:
    __metrics = gradio.CheckboxGroup(label=label, type='value', value=[UNICODE, SURPRISAL, PERPLEXITY], choices=[('Ramping', RAMPING), ('Unicode', UNICODE), ('Surprisal', SURPRISAL), ('Perplexity', PERPLEXITY), ('Intermediate', INTERMEDIATE)], interactive=True)
    return {prefix + 'selection_block': __metrics,}

def create_window_block(label: str='Scope', prefix: str='', value: int=5) -> dict:
    __window = gradio.Slider(label=label, value=value, minimum=1, maximum=32, step=1, scale=1, interactive=True)
    return {prefix + 'window_block': __window,}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __process = gradio.Button('Process', variant='primary', size='lg', scale=1, interactive=True)
    return {'process_block': __process,}

# STATE ########################################################################

def create_state(export_str: str='') -> dict:
    return {
        'export_state': gradio.State(export_str),
        'tokens_state': gradio.State(load_from_disk('tokens.pt')),
        'indices_state': gradio.State(load_from_disk('indices.pt')),
        'logits_state': gradio.State(None), # too large and not useful on startup
        'unicode_state': gradio.State(load_from_disk('unicodes.pt')),
        'rank_state': gradio.State(load_from_disk('ranks.pt')),
        'entropy_state': gradio.State(load_from_disk('entropies.pt')),
        'perplexity_state': gradio.State(load_from_disk('perplexities.pt')),
        'surprisal_state': gradio.State(load_from_disk('surprisals.pt')),}

# LAYOUT #######################################################################

def create_layout(title: str=TITLE, intro: str=INTRO, tuto: str=TUTO, docs: str=DOCS, models: list=MODELS) -> dict:
    __fields = {}
    with gradio.Row(equal_height=True):
        __fields.update(create_text_block(text='# ' + title))
    with gradio.Row(equal_height=True):
        __fields.update(create_highlight_block(label='', prefix='intro_', value=intro, cmap=create_score_cmap(), show_label=False))
    with gradio.Tabs():
        with gradio.Tab('Scores') as __main_tab:
            __fields.update({'main_tab': __main_tab})
            with gradio.Accordion(label='Inputs', open=True, visible=True):
                with gradio.Row(equal_height=True):
                    __fields.update(create_inputs_block(label='Prompt', prefix='', value=tuto))
            with gradio.Accordion(label='Outputs', open=True, visible=True):
                with gradio.Row(equal_height=True):
                    __fields.update(create_highlight_block(label='Results', prefix='', value=load_from_disk('labels.pt'), cmap=create_score_cmap()))
        with gradio.Tab('Graphs') as __graphs_tab:
            __fields.update({'graphs_tab': __graphs_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_plot_block(label='Metrics', prefix=''))
        with gradio.Tab('Samples') as __samples_tab:
            __fields.update({'samples_tab': __samples_tab})
            with gradio.Accordion(label='De-Generate App', open=True, visible=True):
                with gradio.Row(equal_height=True):
                    __fields.update(create_dataset_block(options=['Readme', 'Documentation'], prefix='degen_'))
            with gradio.Accordion(label='Known Texts', open=True, visible=True):
                with gradio.Row(equal_height=True):
                    __fields.update(create_dataset_block(options=['Wikipedia', 'License', 'Cookies', 'Contract'], prefix='known_'))
            with gradio.Accordion(label='System Prompts', open=True, visible=True):
                with gradio.Row(equal_height=True):
                    __fields.update(create_dataset_block(options=['ChatGPT', 'Opus', 'Gemini', 'Grok', 'Soul'], prefix='system_'))
            with gradio.Accordion(label='HC3 Dataset', open=True, visible=True):
                with gradio.Row(equal_height=True):
                    __fields.update(create_dataset_block(options=['Human', 'AI', ], prefix='hc3_'))
            with gradio.Accordion(label='LLM Trace Dataset', open=True, visible=True):
                with gradio.Row(equal_height=True):
                    __fields.update(create_dataset_block(options=['Human', 'AI', ], prefix='trace_'))
        with gradio.Tab('Settings') as __settings_tab:
            __fields.update({'settings_tab': __settings_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_model_block(models=models))
            with gradio.Row(equal_height=True):
                __fields.update(create_sampling_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_selection_block(label='Metrics', prefix=''))
                __fields.update(create_window_block(label='Window', prefix='', value=3))
        with gradio.Tab('Docs') as __docs_tab:
            __fields.update({'docs_tab': __docs_tab})
            __fields.update(create_text_block(text=docs))
    with gradio.Row(equal_height=True):
        __fields.update(create_actions_block())
    return __fields

# BUTTONS ######################################################################

def enable_button() -> dict:
    return gradio.update(interactive=True)

def disable_button() -> dict:
    return gradio.update(interactive=False)

# SAMPLES ######################################################################

def sample_input_text(
    dataset_str: str,
    type_str: str,
    samples_arr: dict=SAMPLES,
) -> str:
    global DOCS
    # return the documentation by default
    __dataset = samples_arr.get(dataset_str, {}).get(type_str, [DOCS])
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
    __tokens = psaiops.common.tokenizer.preprocess_token_str(
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
    __inputs = psaiops.common.tokenizer.preprocess_token_ids(
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
    __logits = psaiops.score.human.lib.compute_raw_logits(
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
    __unicodes = psaiops.score.human.lib.compute_unicode_metrics(
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
    __ranks = psaiops.score.human.lib.compute_rank_metrics(
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
    __entropies = psaiops.score.human.lib.compute_entropy_metrics(
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
    __perplexities = psaiops.score.human.lib.compute_perplexity_metrics(
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
    __surprisals = psaiops.score.human.lib.compute_surprisal_metrics(
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
    __surprisal_arr = psaiops.score.human.lib.apply_time_ramp(data_arr=surprisal_arr, **__args) if (RAMPING in selection_arr) else surprisal_arr
    __perplexity_arr = psaiops.score.human.lib.apply_time_ramp(data_arr=perplexity_arr, **__args) if (RAMPING in selection_arr) else perplexity_arr
    # compute the scores on a sliding window
    __surprisal_arr = psaiops.score.human.lib.compute_topk_pooling(
        data_arr=__surprisal_arr,
        pool_dim=__window_dim,
        topk_dim=(__window_dim + 1) // 2,
        axis_idx=-1)
    __perplexity_arr = psaiops.score.human.lib.compute_average_pooling(
        data_arr=__perplexity_arr,
        pool_dim=max(9, __window_dim),
        axis_idx=-1)
    # toggle the metrics according to the selection
    __token_cls = (
        [0.5 * torch.ones_like(unicode_arr, device=unicode_arr.device, dtype=unicode_arr.dtype)]
        + (UNICODE in selection_arr) * [unicode_arr]
        + (SURPRISAL in selection_arr) * [__surprisal_arr]
        + (PERPLEXITY in selection_arr) * [__perplexity_arr])
    # there is an extra neutral score in case all other have been toggled off (so that the conflation doesn't error)
    __token_cls = psaiops.score.human.lib.compute_probability_conflation(
        metrics_arr=__token_cls,
        axis_idx=-1)
    # scale into a [0; 100] label
    __token_cls = psaiops.score.human.lib.postprocess_score_cls(
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
    __yt = psaiops.score.human.lib.sigmoid_ramp(
        time_dim=int(unicode_arr.shape[-1]),
        step_val=0.25,
        rate_val=10.0)
    # smooth the curves
    __ye = psaiops.score.human.lib.compute_average_pooling(
        data_arr=entropy_arr,
        pool_dim=__window_dim,
        axis_idx=-1)
    __yp = psaiops.score.human.lib.compute_average_pooling(
        data_arr=perplexity_arr,
        pool_dim=__window_dim,
        axis_idx=-1)
    __ys = psaiops.score.human.lib.compute_topk_pooling(
        data_arr=surprisal_arr,
        pool_dim=__window_dim,
        topk_dim=(__window_dim + 1) // 2,
        axis_idx=-1)
    # combine all the metrics into a final score
    __yf = psaiops.score.human.lib.compute_probability_conflation(
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
    if RAMPING in selection_arr:
        __axes.plot(__x, __yt, linestyle='--', label='unicode', color='#555500') # olive leaf
    if UNICODE in selection_arr:
        __axes.plot(__x, __yu, linestyle='--', label='unicode', color='#005555') # slate grey
    if PERPLEXITY in selection_arr:
        __axes.plot(__x, __yp, linestyle='--', label='perplexity', color='#5555FF') # full blue
    if SURPRISAL in selection_arr:
        __axes.plot(__x, __ys, linestyle='--', label='surprisal', color='#550055') # deep purple
    if INTERMEDIATE in selection_arr:
        __axes.plot(__x, __yr, linestyle=':', label='rank', color='#442222') # expresso
        __axes.plot(__x, __ye, linestyle=':', label='entropy', color='#222244') # indigo
    # display the legend and remove the extra padding
    __axes.legend()
    __figure.tight_layout()
    # remove the figure for the pyplot register for garbage collection
    matplotlib.pyplot.close(__figure)
    # update each component => (highlight, plot) states
    return __figure

# APP ##########################################################################

def create_app(
    switch: callable,
    current: callable,
    partition: callable,
    convert: callable,
    compute: callable,
    title: str=TITLE,
    intro: str=INTRO,
    tuto: str=TUTO,
    docs: str=DOCS,
    export: str='',
    models: list=MODELS,
    samples: dict=SAMPLES,
) -> gradio.Blocks:
    # holds all the UI widgets
    __fields = {}
    with gradio.Blocks(title=title) as __app:
        # create the UI
        __fields.update(create_layout(title=title, intro=intro, tuto=tuto, docs=docs, models=current().get('choices', models)))
        # init the state
        __fields.update(create_state(export_str=export))
        # split the string into token sub-strings
        __fields['process_block'].click(
            fn=partition,
            inputs=[__fields[__k] for __k in ['input_block', 'export_state']],
            outputs=__fields['tokens_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # translate the string into token indices
            fn=convert,
            inputs=[__fields[__k] for __k in ['input_block', 'export_state']],
            outputs=__fields['indices_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # then compute the associated logits
            fn=compute,
            inputs=[__fields[__k] for __k in ['indices_state', 'export_state']],
            outputs=__fields['logits_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the unicode scores
            fn=update_unicode_state,
            inputs=[__fields[__k] for __k in ['tokens_state', 'export_state']],
            outputs=__fields['unicode_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the rank scores
            fn=update_rank_state,
            inputs=[__fields[__k] for __k in ['indices_state', 'logits_state', 'export_state']],
            outputs=__fields['rank_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the entropy scores
            fn=update_entropy_state,
            inputs=[__fields[__k] for __k in ['logits_state', 'export_state']],
            outputs=__fields['entropy_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the perplexity scores
            fn=update_perplexity_state,
            inputs=[__fields[__k] for __k in ['indices_state', 'logits_state', 'export_state']],
            outputs=__fields['perplexity_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the surprisal scores
            fn=update_surprisal_state,
            inputs=[__fields[__k] for __k in ['indices_state', 'logits_state', 'export_state']],
            outputs=__fields['surprisal_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # then compute the scores
            fn=update_token_highlights,
            inputs=[__fields[__k] for __k in ['tokens_state', 'unicode_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block', 'export_state']],
            outputs=__fields['highlight_block'],
            queue=True,
            show_progress='full'
        ).then(
        # and plot the metrics
            fn=update_metric_plots,
            inputs=[__fields[__k] for __k in ['unicode_state', 'rank_state', 'entropy_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block']],
            outputs=__fields['plot_block'],
            queue=True,
            show_progress='full')
        # update the plots when the metric selection changes
        __fields['selection_block'].change(
            fn=update_token_highlights,
            inputs=[__fields[__k] for __k in ['tokens_state', 'unicode_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block']],
            outputs=__fields['highlight_block'],
            queue=True,
            show_progress='full'
        ).then(
            fn=update_metric_plots,
            inputs=[__fields[__k] for __k in ['unicode_state', 'rank_state', 'entropy_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block']],
            outputs=__fields['plot_block'],
            queue=True,
            show_progress='full')
        # update the plots when the window changes
        __fields['window_block'].change(
            fn=update_token_highlights,
            inputs=[__fields[__k] for __k in ['tokens_state', 'unicode_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block']],
            outputs=__fields['highlight_block'],
            queue=True,
            show_progress='full'
        ).then(
            fn=update_metric_plots,
            inputs=[__fields[__k] for __k in ['unicode_state', 'rank_state', 'entropy_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block']],
            outputs=__fields['plot_block'],
            queue=True,
            show_progress='full')
        # activate the loading button when selecting another model
        __fields['model_block'].change(
            fn=enable_button,
            inputs=[],
            outputs=__fields['load_block'],
            queue=True,
            show_progress='full')
        # require an explicit button press to load the selected model
        __fields['load_block'].click(
            fn=switch,
            inputs=__fields['model_block'],
            outputs=__fields['model_block'],
            queue=True,
            show_progress='full')
        # make sure the loaded model is selected (in case the user changed the selection without loading the model)
        __fields['settings_tab'].select(
            fn=current,
            inputs=[],
            outputs=__fields['model_block'],
            queue=True,
            show_progress='hidden'
        ).then(
        # disable the loading button
            fn=disable_button,
            inputs=[],
            outputs=__fields['load_block'],
            queue=True,
            show_progress='hidden')
        # buttons to fill the input with random samples
        for __dataset in samples.keys():
            for __label in samples[__dataset].keys():
                # fill the input textbox when the user clicks on a given sample
                __fields[f'{__dataset}_{__label}_block'].click(
                    fn=sample_input_text,
                    inputs=[gradio.State(__dataset), gradio.State(__label)],
                    outputs=__fields['input_block'],
                    queue=True,
                    show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    # load the model
    __device = 'cuda' if torch.cuda.is_available() else 'cpu'
    __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=MODELS[0], device=__device)
    __model = psaiops.common.model.get_model(name=MODELS[0], device=__device)
    # adapt the event handlers
    __current = lambda: gradio.update(value=MODELS[0], choices=MODELS) # list of the models with the current model at index 0
    __switch = lambda: __current() # disable model switching here
    __partition = functools.partial(update_tokens_state, tokenizer_obj=__tokenizer)
    __convert = functools.partial(update_indices_state, tokenizer_obj=__tokenizer)
    __compute = functools.partial(update_logits_state, model_obj=__model)
    # the event handlers are created outside so that they can be wrapped with `spaces.GPU` if necessary
    __app = create_app(switch=__switch, current=__current, partition=__partition, convert=__convert, compute=__compute)
    __app.launch(theme=gradio.themes.Soft(), css=psaiops.common.style.ALL, share=True, debug=True)
