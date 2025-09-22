import functools

import gradio
import torch
import torch.cuda

# META #########################################################################

TITLE = '''Contrastive Steering'''
INTRO = '''Add a delta of activation to a prompt to steer the model output in a specific latent direction.'''
STYLE = '''.white-text span { color: white; }'''

MODEL = 'openai/gpt-oss-20b'

# COLORS #######################################################################

def create_color_map() -> dict:
    return {
        '-1': '#004444',
        **{str(__i): '#{:02x}0000'.format(int(2.55 * __i)) for __i in range(101)}}

# INTRO ########################################################################

def create_intro_block(intro: str) -> dict:
    __intro = gradio.Markdown(intro)
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
        'topp_block': __topp}

# TARGET #######################################################################

def create_target_block() -> dict:
    __target = gradio.Radio(label='Score', value='Inputs', choices=['Inputs', 'Everything'], scale=1, interactive=True)
    return {'target_block': __target}

# INPUTS #######################################################################

def create_inputs_block() -> dict:
    __input = gradio.Textbox(label='Prompt', value='', placeholder='A string of tokens to score.', lines=4, scale=1, show_copy_button=True, interactive=True)
    return {'input_block': __input}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.HighlightedText(label='Scores', value='', scale=1, interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=False, color_map=create_color_map(), elem_classes='white-text')
    return {'output_block': __output}

# SELECT #######################################################################

def create_selection_block() -> dict:
    __position = gradio.Slider(label='Token Position', value=-1, minimum=-1, maximum=15, step=1, scale=1, interactive=True) # info='-1 to average on all tokens'
    __layer = gradio.Slider(label='Layer Depth', value=12, minimum=-1, maximum=23, step=1, scale=1, interactive=True) # info='-1 to average on all layers'
    __head = gradio.Slider(label='Attention Head', value=-1, minimum=-1, maximum=63, step=1, scale=1, interactive=True) # info='-1 to average on all heads'
    return {
        'position_block': __position,
        'layer_block': __layer,
        'head_block': __head,}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __process = gradio.Button('Process', variant='primary', size='lg', scale=1, interactive=True)
    return {'process_block': __process,}

# STATE ########################################################################

def create_state() -> dict:
    return {
        'input_state': gradio.State(None),
        'output_state': gradio.State(None),
        'attention_state': gradio.State(None),}

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
                    __fields.update(create_target_block())
                    # __fields.update(create_display_block())
    return __fields

# EVENTS #######################################################################

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, style: str=STYLE, model: str=MODEL) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        # load the model
        __device = 'cuda' if torch.cuda.is_available() else 'cpu'
        __model = psaiops.score.attention.lib.get_model(name=model, device=__device)
        __tokenizer = psaiops.score.attention.lib.get_tokenizer(name=model, device=__device)
        # adapt the computing function
        __compute = functools.partial(update_computation_state, model_obj=__model, tokenizer_obj=__tokenizer, device_str=__device)
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
        __fields['process_block'].click(
            fn=__compute,
            inputs=[__fields[__k] for __k in ['tokens_block', 'topk_block', 'topp_block', 'position_block', 'layer_block', 'head_block', 'input_block']],
            outputs=[__fields[__k] for __k in ['output_block', 'input_state', 'output_state', 'attention_state']],
            queue=False,
            show_progress='full')
        __fields['position_block'].change(
            fn=update_text_highlight,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'head_block', 'input_state', 'output_state', 'attention_state']],
            outputs=__fields['output_block'],
            queue=False,
            show_progress='hidden')
        __fields['layer_block'].change(
            fn=update_text_highlight,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'head_block', 'input_state', 'output_state', 'attention_state']],
            outputs=__fields['output_block'],
            queue=False,
            show_progress='hidden')
        __fields['head_block'].change(
            fn=update_text_highlight,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'head_block', 'input_state', 'output_state', 'attention_state']],
            outputs=__fields['output_block'],
            queue=False,
            show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_app()
    __app.launch(share=True, debug=True)
