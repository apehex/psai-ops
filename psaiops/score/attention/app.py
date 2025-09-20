import functools

import gradio
import torch
import torch.cuda

import psaiops.score.attention.lib

# META #########################################################################

TITLE = '''Attention Scoring'''
INTRO = '''Score each token according to the weights of the attention layers.\nThe model is fixed to "openai/gpt-oss-20b" for now.'''
STYLE = '''.white-text span { color: white; }'''

MODEL = 'openai/gpt-oss-20b'

# COLORS #######################################################################

def create_color_map() -> dict:
    return {
        '-1': '#00ff00',
        **{str(__i): '#{:02x}0000'.format(int(2.55 * __i)) for __i in range(101)}}

# INTRO ########################################################################

def create_intro_block(intro: str) -> dict:
    __intro = gradio.Markdown(intro)
    return {'intro_block': __intro}

# MODEL ########################################################################

def create_model_block() -> dict:
    __model_dd = gradio.Dropdown(label='Model', value='openai/gpt-oss-20b', choices=['openai/gpt-oss-20b'], allow_custom_value=False, multiselect=False, interactive=True) # 'openai/gpt-oss-120b'
    __layer_sl = gradio.Slider(label='Layer Depth', value=12, minimum=-1, maximum=23, step=1, interactive=True) # info='-1 to average on all layers'
    __head_sl = gradio.Slider(label='Attention Head', value=-1, minimum=-1, maximum=63, step=1, interactive=True) # info='-1 to average on all heads'
    __model_dd.change(fn=update_layer_range, inputs=[__layer_sl, __model_dd], outputs=__layer_sl, queue=False, show_progress='hidden')
    return {
        'model_block': __model_dd,
        'layer_block': __layer_sl,
        'head_block': __head_sl}

# SAMPLING #####################################################################

def create_sampling_block() -> dict:
    __tokens = gradio.Slider(label='Tokens', value=32, minimum=0, maximum=128, step=1, interactive=True)
    __topk = gradio.Slider(label='Top K', value=4, minimum=1, maximum=8, step=1, interactive=True)
    __topp = gradio.Slider(label='Top P', value=0.8, minimum=0.0, maximum=1.0, step=0.1, interactive=True)
    return {
        'tokens_block': __tokens,
        'topk_block': __topk,
        'topp_block': __topp}

# DISPLAY ######################################################################

def create_display_block() -> dict:
    __display = gradio.Radio(label='Display', value='Tokens', choices=['Tokens', 'Indexes'], interactive=True)
    return {'display_block': __display}

# INPUTS #######################################################################

def create_inputs_block() -> dict:
    __input = gradio.Textbox(label='Prompt', value='', placeholder='A string of tokens to score.', lines=4, show_copy_button=True, interactive=True)
    return {'input_block': __input}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.HighlightedText(label='Scores', value='', interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=True, color_map=create_color_map(), elem_classes='white-text')
    return {'output_block': __output}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __process = gradio.Button('Process', variant='primary', size='lg', interactive=True)
    __position = gradio.Slider(label='Position', value=-1, minimum=-1, maximum=128, step=1, interactive=True) # info='-1 to average on all tokens'
    return {
        'process_block': __process,
        'position_block': __position}

# STATE ########################################################################

def create_state() -> dict:
    return {'attention_state': gradio.State(None), 'token_state': gradio.State(None)}

# LAYOUT #######################################################################

def create_layout(intro: str=INTRO) -> dict:
    __fields = {}
    __fields.update(create_intro_block(intro=intro))
    with gradio.Tabs():
        with gradio.Tab('Score Tokens') as __main_tab:
            __fields.update({'main_tab': __main_tab})
            with gradio.Row():
                with gradio.Column(scale=1):
                    __fields.update(create_inputs_block())
                with gradio.Column(scale=1):
                    __fields.update(create_outputs_block())
            with gradio.Row():
                __fields.update(create_actions_block())
        with gradio.Tab('Settings') as __settings_tab:
            __fields.update({'settings_tab': __settings_tab})
            with gradio.Column(scale=1):
                with gradio.Row():
                    __fields.update(create_model_block())
                with gradio.Row():
                    __fields.update(create_sampling_block())
                with gradio.Row():
                    __fields.update(create_display_block())
    return __fields

# EVENTS #######################################################################

def update_layer_range(value: int, model: str) -> dict:
    return gradio.update(maximum=35, value=min(35, int(value))) if '120b' in model else gradio.update(maximum=23, value=min(23, int(value)))

def update_position_range(value: int, dimension: int) -> dict:
    return gradio.update(maximum=dimension - 1, value=min(dimension - 1, value))

def update_output_value(
    attention_data: torch.Tensor=None,
    token_data: torch.Tensor=None,) -> torch.Tensor:
    return

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, style: str=STYLE, model: str=MODEL) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        # load the model
        __device = 'cuda' if torch.cuda.is_available() else 'cpu'
        __model = psaiops.score.attention.lib.get_model(name=model, device=__device)
        __tokenizer = psaiops.score.attention.lib.get_tokenizer(name=model, device=__device)
        # adapt the scoring function
        __score = functools.partial(psaiops.score.attention.lib.score_tokens, model_obj=__model, tokenizer_obj=__tokenizer, device_str=__device)
        # create the UI
        __fields.update(create_layout(intro=intro))
        # init the state
        __fields.update(create_state())
        # fetch the relevant fields
        __button = __fields['process_block']
        # wire the input fields
        __button.click(
            fn=__score,
            inputs=[__fields[__k] for __k in ['input_block', 'tokens_block', 'topk_block', 'topp_block', 'position_block', 'layer_block', 'head_block']],
            outputs=__fields['output_block'],
            queue=False,
            show_progress='full')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_app()
    __app.launch(share=True, debug=True)
