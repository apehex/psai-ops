import functools

import gradio
import pandas
import torch
import torch.cuda

import psaiops.compose.contrast.lib

# META #########################################################################

TITLE = '''Activation Maths'''
INTRO = '''Compose prompts in the latent space.'''
STYLE = '''.giga-text input { font-size: 32px; }'''

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
    __tokens = gradio.Slider(label='Tokens', value=32, minimum=1, maximum=128, step=1, scale=1, interactive=True)
    __topk = gradio.Slider(label='Top K', value=4, minimum=1, maximum=8, step=1, scale=1, interactive=True)
    __topp = gradio.Slider(label='Top P', value=0.9, minimum=0.0, maximum=1.0, step=0.1, scale=1, interactive=True)
    return {
        'tokens_block': __tokens,
        'topk_block': __topk,
        'topp_block': __topp,}

# REDUCTION ####################################################################

def create_reduction_block() -> dict:
    __from = gradio.Slider(label='Average From', value=0, minimum=0, maximum=256, step=1, scale=1, interactive=True)
    __to = gradio.Slider(label='Average To', value=256, minimum=0, maximum=256, step=1, scale=1, interactive=True)
    return {
        'from_block': __from,
        'to_block': __to,}

# INPUTS #######################################################################

def create_inputs_row(operation: str='', index: int=0) -> dict:
    # __operation = gradio.Button(value=operation, variant='primary', size='lg', elem_classes='white-text', scale=1, interactive=False)
    __operation = gradio.Dropdown(
        label=f'Operation',
        value='' if (index == 0) else operation,
        choices=(index == 0) * [''] + ['+', '-', 'x', '.', '='],
        elem_classes='giga-text',
        scale=1,
        show_label=(index == 0),
        allow_custom_value=False,
        multiselect=False,
        interactive=(index != 0))
    __alpha = gradio.Slider(
        label='Factor',
        value=1.0,
        minimum=0.0,
        maximum=8.0,
        step=0.1,
        scale=1,
        show_label=(index == 0),
        interactive=True)
    __input = gradio.Textbox(
        label=f'Prompt',
        value='',
        placeholder='Some text.',
        lines=2,
        max_lines=2,
        scale=8,
        show_label=(index == 0),
        show_copy_button=True,
        interactive=True)
    __delete = gr.Button(
        value='✖',
        variant='secondary',
        size='lg',
        scale=1,
        interactive=(index != 0))
    return {
        f'operation_{index}_block': __operation,
        f'factor_{index}_block': __alpha,
        f'prompt_{index}_block': __input,
        f'delete_{index}_block': __delete,}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.Textbox(label='= Total', value='', placeholder='Some text.', lines=2, max_lines=8, scale=1, show_label=True, show_copy_button=True, interactive=False)
    return {'output_block': __output}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __add = gradio.Button(value='Add', variant='primary', size='lg', scale=1, interactive=True)
    __process = gradio.Button(value='Process', variant='primary', size='lg', scale=1, interactive=True)
    return {
        'add_block': __add,
        'process_block': __process,}

# TABLE ########################################################################

def create_table_block() -> dict:
    __table = gradio.DataFrame(label='Summary', type='numpy', headers=None,  row_count=4, col_count=256, scale=1, interactive=False)
    return {'table_block': __table,}

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
                __fields.update(create_inputs_row(operation='', index=0, label=True))
            with gradio.Row(equal_height=True):
                __fields.update(create_inputs_row(operation='-', index=1, label=False))
            with gradio.Row(equal_height=True):
                __fields.update(create_inputs_row(operation='+', index=2, label=False))
            with gradio.Row(equal_height=True):
                __fields.update(create_outputs_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_actions_block())
        with gradio.Tab('Details') as __details_tab:
            __fields.update({'details_tab': __details_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_table_block())
        with gradio.Tab('Settings') as __settings_tab:
            __fields.update({'settings_tab': __settings_tab})
            with gradio.Column(scale=1):
                with gradio.Row(equal_height=True):
                    __fields.update(create_model_block())
                with gradio.Row(equal_height=True):
                    __fields.update(create_sampling_block())
                with gradio.Row(equal_height=True):
                    __fields.update(create_reduction_block())
                    # __fields.update(create_display_block())
    return __fields

# DYNAMIC ######################################################################

def render_rows(rows: list) -> list:
    updates = []
    for __i, __row in enumerate(rows):
        updates.append(gradio.update(visible=True, value=__row.get('operation', '')))
        updates.append(gradio.update(visible=True, value=__row.get('alpha', 1.0)))
        updates.append(gradio.update(visible=True, value=__row.get('prompt', '')))
        updates.append(gradio.update(visible=True))
    return updates

def add_row(rows: list) -> tuple:
    rows.append({'operation': '+', 'alpha': 1.0, 'prompt': ''})
    return rows, *render_rows(rows)

def remove_row(rows: list, index: int) -> tuple:
    if 0 <= index < len(rows):
        rows.pop(index)
    return rows, *render_rows(rows)

# EVENTS #######################################################################

def update_layer_range(value: float, model: str) -> dict:
    return gradio.update(maximum=35, value=min(35, int(value))) if '120b' in model else gradio.update(maximum=23, value=min(23, int(value)))

def update_table_data(positive: str, negative: str, prompt: str, output: str, tokenizer: object) -> pandas.DataFrame:
    # array of token IDs
    __outputs = tokenizer([positive, negative, prompt, output], return_tensors='pt', padding=True)
    # array of token strings
    __tokens = [tokenizer.convert_ids_to_tokens(__s) for __s in __outputs['input_ids']]
    # shift the special characters
    __tokens = [[__t.replace(chr(0x0120), ' ').replace(chr(0x010a), '\\n') for __t in __s] for __s in __tokens]
    # mask the tokens that differ between positive and negative prompts
    __masks = psaiops.compose.contrast.lib.compute_sequence_mask(tokens=__outputs['input_ids'])
    # convert into a data frame
    __data = pandas.DataFrame(__tokens)
    # color the background in red for the positions marked by the mask
    return __data.style.apply(update_table_style, masks=pandas.DataFrame(__masks), axis=None)

def update_table_style(data: pandas.DataFrame, masks: pandas.DataFrame) -> pandas.DataFrame:
    return pandas.DataFrame(masks.replace({True: 'background-color: rgb(255, 0, 0, 64%)', False: 'background-color: rgb(0, 0, 0, 0%)',}))

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, style: str=STYLE, model: str=MODEL) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        # load the model
        __device = 'cuda' if torch.cuda.is_available() else 'cpu'
        __model = psaiops.compose.contrast.lib.get_model(name=model, device=__device)
        __tokenizer = psaiops.compose.contrast.lib.get_tokenizer(name=model, device=__device)
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
