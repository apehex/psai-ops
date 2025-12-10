import functools

import gradio
import torch
import torch.cuda

import psaiops.common.model
import psaiops.common.tokenizer

# META #########################################################################

STYLE = '''.white-text span { color: white; }'''
TITLE = '''Shapley Scoring'''
INTRO = '''Score each token according to their [Shapley value](https://en.wikipedia.org/wiki/Shapley_value).\nUnder construction, only "openai/gpt-oss-20b" is available for now.'''

MODEL = 'openai/gpt-oss-20b'

# COLORS #######################################################################

def create_color_map() -> dict:
    return {
        '-1': '#004444',
        **{str(__i): '#{:02x}0000'.format(int(2.55 * __i)) for __i in range(101)}}

# INTRO ########################################################################

def create_intro_block(intro: str) -> dict:
    __intro = gradio.Markdown(intro, line_breaks=True)
    return {'intro_block': __intro}

# MODEL ########################################################################

def create_model_block() -> dict:
    __model = gradio.Dropdown(label='Model', value='openai/gpt-oss-20b', choices=['openai/gpt-oss-20b'], scale=1, allow_custom_value=False, multiselect=False, interactive=True) # 'openai/gpt-oss-120b'
    return {'model_block': __model,}

# SAMPLING #####################################################################

def create_sampling_block() -> dict:
    __tokens = gradio.Slider(label='Tokens', value=16, minimum=1, maximum=128, step=1, scale=1, interactive=True)
    __topk = gradio.Slider(label='Top K', value=4, minimum=1, maximum=8, step=1, scale=1, interactive=True)
    __topp = gradio.Slider(label='Top P', value=0.9, minimum=0.0, maximum=1.0, step=0.1, scale=1, interactive=True)
    return {
        'tokens_block': __tokens,
        'topk_block': __topk,
        'topp_block': __topp,}

# SAMPLING #####################################################################

def create_computation_block() -> dict:
    __count = gradio.Slider(label='Sample count', value=4, minimum=1, maximum=16, step=1, scale=1, interactive=True)
    __min = gradio.Slider(label='Min size', value=50, minimum=0, maximum=100, step=1, scale=1, interactive=True)
    __max = gradio.Slider(label='Max size', value=100, minimum=0, maximum=100, step=1, scale=1, interactive=True)
    return {
        'count_block': __count,
        'min_block': __min,
        'max_block': __max,}

# INPUTS #######################################################################

def create_inputs_block() -> dict:
    __input = gradio.Textbox(label='Prompt', value='', placeholder='A string of tokens to score.', lines=4, scale=1, show_copy_button=True, interactive=True)
    return {'input_block': __input}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.HighlightedText(label='Scores', value='', scale=1, interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=False, color_map=create_color_map(), elem_classes='white-text')
    return {'output_block': __output,}

# SELECT #######################################################################

def create_selection_block() -> dict:
    __position = gradio.Slider(label='Token Position', value=-1, minimum=-1, maximum=15, step=1, scale=1, interactive=True) # info='-1 to average on all tokens'
    __layer = gradio.Slider(label='Layer Depth', value=12, minimum=-1, maximum=23, step=1, scale=1, interactive=True) # info='-1 to average on all layers'
    return {
        'position_block': __position,
        'layer_block': __layer,}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __process = gradio.Button('Process', variant='primary', size='lg', scale=1, interactive=True)
    return {'process_block': __process,}

# STATE ########################################################################

def create_state() -> dict:
    return {}

# LAYOUT #######################################################################

def create_layout(intro: str=INTRO) -> dict:
    __fields = {}
    __fields.update(create_intro_block(intro=intro))
    with gradio.Tabs():
        with gradio.Tab('Score Tokens') as __main_tab:
            __fields.update({'main_tab': __main_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_inputs_block())
                __fields.update(create_outputs_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_selection_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_actions_block())
        with gradio.Tab('Settings') as __settings_tab:
            __fields.update({'settings_tab': __settings_tab})
            with gradio.Column(scale=1):
                with gradio.Row(equal_height=True):
                    __fields.update(create_model_block())
                with gradio.Row(equal_height=True):
                    __fields.update(create_sampling_block())
                with gradio.Row(equal_height=True):
                    __fields.update(create_computation_block())
    return __fields

# EVENTS #######################################################################

def update_layer_range(value: float, model: str) -> dict:
    return gradio.update(maximum=35, value=min(35, int(value))) if '120b' in model else gradio.update(maximum=23, value=min(23, int(value)))

def update_position_range(value: float, tokens: float) -> dict:
    return gradio.update(maximum=int(tokens) - 1, value=min(int(tokens) - 1, int(value)))

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, style: str=STYLE, model: str=MODEL) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        # load the model
        __device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # __model = psaiops.common.model.get_model(name=model, device=__device)
        __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=model, device=__device)
        # create the UI
        __fields.update(create_layout(intro=intro))
        # init the state
        __fields.update(create_state())
        # wire the input fields
        __fields['tokens_block'].change(
            fn=update_position_range,
            inputs=[__fields[__k] for __k in ['position_block', 'tokens_block']],
            outputs=__fields['position_block'],
            queue=False,
            show_progress='hidden')
        __fields['model_block'].change(
            fn=update_layer_range,
            inputs=[__fields[__k] for __k in ['layer_block', 'model_block']],
            outputs=__fields['layer_block'],
            queue=False,
            show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_app()
    __app.launch(share=True, debug=True)
