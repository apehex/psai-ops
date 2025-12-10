import functools
import itertools

import gradio
import pandas
import torch
import torch.cuda

import psaiops.common.model
import psaiops.common.tokenizer

# META #########################################################################

MODEL = 'openai/gpt-oss-20b'

STYLE = '''.giga-text input { font-size: 32px; }'''
TITLE = '''Activation Maths'''
INTRO = '''Compose prompts in the latent space.\nUnder construction, only "openai/gpt-oss-20b" is available for now.'''

COUNT = 8

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
    with gradio.Row(equal_height=True, visible=(index == 0)) as __row:
        __operation = gradio.Dropdown(
            label=f'Operation',
            value='' if (index == 0) else operation,
            choices=(index == 0) * [''] + ['+', '-', 'x', '.', 'µ', '='],
            elem_classes='giga-text',
            scale=1,
            show_label=(index == 0),
            allow_custom_value=False,
            multiselect=False,
            interactive=(index != 0),
            visible=(index == 0))
        __alpha = gradio.Slider(
            label='Factor',
            value=1.0,
            minimum=0.0,
            maximum=8.0,
            step=0.1,
            scale=1,
            show_label=(index == 0),
            interactive=True,
            visible=(index == 0))
        __input = gradio.Textbox(
            label=f'Prompt',
            value='',
            placeholder='Some text.',
            lines=2,
            max_lines=2,
            scale=8,
            show_label=(index == 0),
            show_copy_button=True,
            interactive=True,
            visible=(index == 0))
        __delete = gradio.Button(
            value='✖',
            variant='secondary',
            size='lg',
            scale=1,
            interactive=(index != 0),
            visible=(index == 0))
    return {
        f'row_{index}_block': __row,
        f'operation_{index}_block': __operation,
        f'factor_{index}_block': __alpha,
        f'prompt_{index}_block': __input,
        f'button_{index}_block': __delete,}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.Textbox(label='= Total', value='', placeholder='Some text.', lines=2, max_lines=8, scale=1, show_label=True, show_copy_button=True, interactive=False)
    return {'output_block': __output}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __add = gradio.Button(value='Add', variant='primary', size='lg', scale=1, interactive=True)
    __process = gradio.Button(value='Process', variant='primary', size='lg', scale=1, interactive=True)
    return {
        'show_block': __add,
        'process_block': __process,}

# TABLE ########################################################################

def create_table_block() -> dict:
    __table = gradio.DataFrame(label='Summary', type='numpy', headers=None,  row_count=4, col_count=256, scale=1, interactive=False)
    return {'table_block': __table,}

# STATE ########################################################################

def default_state(visible: bool=False) -> dict:
    return {'visible': visible, 'operation': '+', 'factor': 1.0, 'prompt': ''}

def create_state(limit: int=COUNT) -> dict:
    return {
        'cache_block': gradio.State(
            [default_state(True)] + [default_state(False) for _ in range(limit - 1)])}

# LAYOUT #######################################################################

def create_layout(intro: str=INTRO, limit: int=COUNT) -> dict:
    __fields = {}
    __fields.update(create_intro_block(intro=intro))
    with gradio.Tabs():
        with gradio.Tab('Equation') as __main_tab:
            __fields.update({'main_tab': __main_tab})
            for __i in range(limit):
                __fields.update(create_inputs_row(operation='+', index=__i))
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

def get_input_rows(inputs: dict, limit: int=COUNT) -> list:
    return list(itertools.chain.from_iterable([
        [
            inputs.get(f'row_{__i}_block', None),
            inputs.get(f'operation_{__i}_block', None),
            inputs.get(f'factor_{__i}_block', None),
            inputs.get(f'prompt_{__i}_block', None),
            inputs.get(f'button_{__i}_block', None),]
        for __i in range(limit)]))

def render_input_rows(rows: list) -> list:
    return list(itertools.chain.from_iterable([
        [
            gradio.update(visible=__r.get('visible', False)),
            gradio.update(visible=__r.get('visible', False), value=__r.get('operation', '')),
            gradio.update(visible=__r.get('visible', False), value=__r.get('factor', 1.0)),
            gradio.update(visible=__r.get('visible', False), value=__r.get('prompt', '')),
            gradio.update(visible=__r.get('visible', False))]
        for __r in rows]))

def show_input_row(rows: list) -> tuple:
    __count = 0
    __rows = list(rows)
    for __i in range(len(__rows)):
        # count the number of hidden rows (before changing their state)
        __count = __count + int(not __rows[__i]['visible'])
        # all the visible rows stay the same and the first hidden row is toggled
        __rows[__i]['visible'] = __rows[__i]['visible'] or (__count < 2)
    # update state and components
    return __rows, *render_input_rows(__rows)

def hide_input_row(rows: list, index: int) -> tuple:
    __rows = list(rows)
    # always show the first row
    if 0 < index < len(__rows):
        # remove the target row
        __rows.pop(index)
        # keep the number of rows constant
        __rows.append({'visible': False, 'operation': '+', 'factor': 1.0, 'prompt': ''})
    # update state and components
    return __rows, *render_input_rows(__rows)

# EVENTS #######################################################################

def update_layer_range(value: float, model: str) -> dict:
    return gradio.update(maximum=35, value=min(35, int(value))) if '120b' in model else gradio.update(maximum=23, value=min(23, int(value)))

def update_input_cache(cache: list, value: any, index: int, field: str) -> list:
    __cache = list(cache)
    __cache[index][field] = value
    return __cache

def update_operation_cache(cache: list, index: int, value: any) -> list:
    return update_input_cache(cache=cache, index=int(index), value=str(value), field='operation')

def update_factor_cache(cache: list, index: int, value: any) -> list:
    return update_input_cache(cache=cache, index=int(index), value=float(value), field='factor')

def update_prompt_cache(cache: list, index: int, value: any) -> list:
    return update_input_cache(cache=cache, index=int(index), value=str(value), field='prompt')

def update_table_data(tokenizer: object) -> callable:
    # called with unpacked arguments
    def __update_table_data(*prompts: list) -> list:
        # array of token IDs
        __outputs = tokenizer(prompts, return_tensors='pt', padding=True)
        # array of token strings
        __tokens = [tokenizer.convert_ids_to_tokens(__s) for __s in __outputs['input_ids']]
        # shift the special characters
        return [[__t.replace(chr(0x0120), ' ').replace(chr(0x010a), '\\n') for __t in __s] for __s in __tokens]
    # fixed to a given tokenizer
    return __update_table_data

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, style: str=STYLE, limit: int=COUNT, model: str=MODEL) -> gradio.Blocks:
    __inputs = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        # load the model
        __device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # __model = psaiops.common.model.get_model(name=model, device=__device)
        __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=model, device=__device)
        # create the UI
        __inputs.update(create_layout(intro=intro, limit=limit))
        # init the state
        __inputs.update(create_state(limit=limit))
        # apply the configuration
        __format = update_table_data(tokenizer=__tokenizer)
        # change the depth of the model
        __inputs['model_block'].change(
            fn=update_layer_range,
            inputs=[__inputs[__k] for __k in ['layer_block', 'model_block']],
            outputs=__inputs['layer_block'],
            queue=False,
            show_progress='hidden')
        # show hidden row
        __inputs['show_block'].click(
            fn=show_input_row,
            inputs=[__inputs['cache_block']],
            outputs=[__inputs['cache_block']] + get_input_rows(inputs=__inputs, limit=limit),
            queue=False,
            show_progress='hidden')
        # update the table
        __inputs['details_tab'].select(
            fn=__format,
            inputs=[__inputs[f'prompt_{__i}_block'] for __i in range(limit)] + [__inputs['output_block']],
            outputs=__inputs['table_block'],
            queue=False,
            show_progress='hidden')
        # link each row of inputs to the cache
        for __i in range(limit):
            # update the target operation in the cache
            __inputs[f'operation_{__i}_block'].change(
                fn=update_operation_cache,
                inputs=[__inputs['cache_block'], gradio.State(__i), __inputs[f'operation_{__i}_block']],
                outputs=__inputs['cache_block'],
                queue=False,
                show_progress='hidden')
            # update the target factor in the cache
            __inputs[f'factor_{__i}_block'].change(
                fn=update_factor_cache,
                inputs=[__inputs['cache_block'], gradio.State(__i), __inputs[f'factor_{__i}_block']],
                outputs=__inputs['cache_block'],
                queue=False,
                show_progress='hidden')
            # update the target prompt in the cache
            __inputs[f'prompt_{__i}_block'].change(
                fn=update_prompt_cache,
                inputs=[__inputs['cache_block'], gradio.State(__i), __inputs[f'prompt_{__i}_block']],
                outputs=__inputs['cache_block'],
                queue=False,
                show_progress='hidden')
            # hide the target row
            __inputs[f'button_{__i}_block'].click(
                fn=hide_input_row,
                inputs=[__inputs['cache_block'], gradio.State(__i)],
                outputs=[__inputs['cache_block']] + get_input_rows(inputs=__inputs, limit=limit),
                queue=False,
                show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_app()
    __app.launch(share=True, debug=True)
