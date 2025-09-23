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
    __model = gradio.Dropdown(label='Model ID', value='openai/gpt-oss-20b', choices=['openai/gpt-oss-20b'], scale=1, allow_custom_value=False, multiselect=False, interactive=True) # 'openai/gpt-oss-120b'
    __layer = gradio.Slider(label='Layer Depth', value=12, minimum=0, maximum=23, step=1, scale=1, interactive=True)
    return {
        'model_block': __model,
        'layer_block': __layer,}

# SAMPLING #####################################################################

def create_sampling_block() -> dict:
    __tokens = gradio.Slider(label='Tokens', value=16, minimum=1, maximum=128, step=1, scale=1, interactive=True)
    __topk = gradio.Slider(label='Top K', value=4, minimum=1, maximum=8, step=1, scale=1, interactive=True)
    __topp = gradio.Slider(label='Top P', value=0.9, minimum=0.0, maximum=1.0, step=0.1, scale=1, interactive=True)
    return {
        'tokens_block': __tokens,
        'topk_block': __topk,
        'topp_block': __topp,}

# DISPLAY ######################################################################

def create_display_block() -> dict:
    __display = gradio.Radio(label='Intermediate Results', value='Show', choices=['Show', 'Hide'], scale=1, interactive=True)
    return {'display_block': __display}

# INPUTS #######################################################################

def create_prompts_row(operation: str='', index: int=0, show: bool=True) -> dict:
    __operation = gradio.Dropdown(label=f'operation-{index}', value='', choices=['', '+ (add)', '- (sub)', '. (dot)', '= (rev)'], scale=1, show_label=False, allow_custom_value=False, multiselect=False, interactive=False)
    __input = gradio.Textbox(label=f'input-{index}', value='', placeholder='Some text.', lines=1, scale=4, show_label=False, show_copy_button=True, interactive=True)
    __arrow = gradio.Markdown(label=f'arrow-{index}', value=' => ')
    __output = gradio.Textbox(label=f'output-{index}', value='', placeholder='Some text.', lines=1, scale=4, show_label=False, show_copy_button=True, interactive=False)
    return {
        f'operation_{index}_block': __operation,
        f'input_{index}_block': __input,
        f'arrow_{index}_block': __arrow,
        f'output_{index}_block': __output,}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.Textbox(label='= Total', value='', placeholder='Some text.', lines=1, scale=4, show_label=True, show_copy_button=True, interactive=False)
    return {'output_block': __output}

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
        with gradio.Tab('Equation') as __main_tab:
            __fields.update({'main_tab': __main_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_prompts_row(operation='', index=0))
            with gradio.Row(equal_height=True):
                __fields.update(create_prompts_row(operation='- (sub)', index=1))
            with gradio.Row(equal_height=True):
                __fields.update(create_prompts_row(operation='+ (add)', index=2))
            with gradio.Row(equal_height=True):
                __fields.update(create_outputs_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_actions_block())
        with gradio.Tab('Details') as __details_tab:
            with gradio.Row(equal_height=True):
                pass
        with gradio.Tab('Settings') as __settings_tab:
            __fields.update({'settings_tab': __settings_tab})
            with gradio.Column(scale=1):
                with gradio.Row(equal_height=True):
                    __fields.update(create_model_block())
                with gradio.Row(equal_height=True):
                    __fields.update(create_sampling_block())
                with gradio.Row(equal_height=True):
                    __fields.update(create_display_block())
                    # __fields.update(create_display_block())
    return __fields

# EVENTS #######################################################################

def update_layer_range(value: float, model: str) -> dict:
    return gradio.update(maximum=35, value=min(35, int(value))) if '120b' in model else gradio.update(maximum=23, value=min(23, int(value)))

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
