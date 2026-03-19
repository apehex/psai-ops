import json
import os

import gradio
import torch

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
    'jailbreak': json.load(open(os.path.join(_PATH, 'data', 'samples', 'jailbreak.json'), 'r')),
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
                    __fields.update(create_dataset_block(options=['ChatGPT', 'Claude', 'Gemini', 'Grok', 'Soul'], prefix='system_'))
            with gradio.Accordion(label='Jailbreak Prompts', open=True, visible=True):
                with gradio.Row(equal_height=True):
                    __fields.update(create_dataset_block(options=['ChatGPT', 'Claude', 'Gemini', 'Grok'], prefix='jailbreak_'))
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
