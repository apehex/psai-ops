import functools
import itertools

import gradio
import pandas
import torch
import torch.cuda

import psaiops.combine.lib

# META #########################################################################

MODEL = 'openai/gpt-oss-20b'

STYLE = '''.giga-text input { font-size: 32px; }'''
TITLE = '''Combine Datasets'''
INTRO = '''Combine and wrap prompts to form new datasets.'''

COUNT = 8

# TEMPLATE #####################################################################

ROLES = ['system', 'developer', 'user', 'assistant', 'tool']
CHANNELS = ['analysis', 'commentary', 'final']

# INTRO ########################################################################

def create_intro_block(intro: str) -> dict:
    __intro = gradio.Markdown(intro, line_breaks=True)
    return {'intro_block': __intro}

# MODEL ########################################################################

def create_model_block() -> dict:
    __model = gradio.Dropdown(label='Model ID', value='openai/gpt-oss-20b', choices=['openai/gpt-oss-20b'], scale=1, allow_custom_value=False, multiselect=False, interactive=True)
    __layer = gradio.Checkbox(label='Apply Template', value=True, scale=1, interactive=True)
    return {
        'model_block': __model,
        'layer_block': __layer,}

# SAMPLING #####################################################################

def create_dataset_block() -> dict:
    __dataset = gradio.Dropdown(label='Dataset IDs', value='', choices=[''], scale=1, allow_custom_value=True, multiselect=True, interactive=True)
    return {'dataset_block': __dataset,}

# INPUTS #######################################################################

def create_inputs_row(index: int=0) -> dict:
    with gradio.Row(equal_height=True, visible=(index == 0)) as __row:
        __role = gradio.Dropdown(
            label=f'Role',
            value='user',
            choices=[__r for __r in ROLES],
            # elem_classes='giga-text',
            scale=1,
            show_label=(index == 0),
            allow_custom_value=False,
            multiselect=False,
            interactive=True,
            visible=(index == 0))
        __channel = gradio.Dropdown(
            label=f'Channel',
            value='final',
            choices=[__c for __c in CHANNELS],
            # elem_classes='giga-text',
            scale=1,
            show_label=(index == 0),
            allow_custom_value=False,
            multiselect=False,
            interactive=True,
            visible=(index == 0))
        __column = gradio.Dropdown(
            label=f'Column',
            value='none',
            choices=['none'],
            # elem_classes='giga-text',
            scale=4,
            show_label=(index == 0),
            allow_custom_value=False,
            multiselect=False,
            interactive=True,
            visible=(index == 0))
        __content = gradio.Textbox(
            label=f'Prompt',
            value='',
            placeholder='Some text.',
            lines=1,
            max_lines=1,
            scale=9,
            show_label=(index == 0),
            show_copy_button=True,
            interactive=True,
            visible=(index == 0))
        __hide = gradio.Button(
            value='X',
            variant='secondary',
            size='lg',
            scale=1,
            interactive=True,
            visible=(index == 0))
    return {
        f'role_{index}_block': __role,
        f'channel_{index}_block': __channel,
        f'column_{index}_block': __column,
        f'content_{index}_block': __content,
        f'button_{index}_block': __hide,}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.Textbox(label='Sample', value='', placeholder='Some text.', lines=2, max_lines=8, scale=1, show_label=True, show_copy_button=True, interactive=False)
    return {'output_block': __output}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __show = gradio.Button(value='Add', variant='primary', size='lg', scale=1, interactive=True)
    __combine = gradio.Button(value='Combine', variant='primary', size='lg', scale=1, interactive=True)
    __upload = gradio.Button(value='Upload', variant='primary', size='lg', scale=1, interactive=True)
    return {
        'show_block': __show,
        'combine_block': __combine,
        'upload_block': __upload,}

# TABLE ########################################################################

def create_table_block() -> dict:
    __table = gradio.DataFrame(label='Table', type='numpy', headers=None,  row_count=4, col_count=256, scale=1, interactive=False)
    return {'table_block': __table,}

# STATE ########################################################################

def create_state(limit: int=COUNT) -> dict:
    return {
        'cache_block': gradio.State(
            [{'visible': True, 'role': 'user', 'channel': 'final', 'column': 'none', 'content': ''}]
            + max(0, limit - 1) * [{'visible': False, 'role': 'user', 'channel': 'final', 'column': 'none', 'content': ''}])}

# LAYOUT #######################################################################

def create_layout(intro: str=INTRO, limit: int=COUNT) -> dict:
    __fields = {}
    __fields.update(create_intro_block(intro=intro))
    with gradio.Tabs():
        with gradio.Tab('Column 0') as __col0_tab:
            __fields.update({'column_0_tab': __col0_tab})
            for __i in range(limit):
                __fields.update(create_inputs_row(index=__i))
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
                    __fields.update(create_dataset_block())
    return __fields

# DYNAMIC ######################################################################

def get_input_rows(inputs: dict, limit: int=COUNT) -> list:
    return list(itertools.chain.from_iterable([
        [
            inputs.get(f'role_{__i}_block', None),
            inputs.get(f'channel_{__i}_block', None),
            inputs.get(f'column_{__i}_block', None),
            inputs.get(f'content_{__i}_block', None),
            inputs.get(f'button_{__i}_block', None),]
        for __i in range(limit)]))

def render_input_rows(rows: list) -> list:
    return list(itertools.chain.from_iterable([
        [
            gradio.update(visible=__r.get('visible', False), value=__r.get('role', 'user')),
            gradio.update(visible=__r.get('visible', False), value=__r.get('channel', 'final')),
            gradio.update(visible=__r.get('visible', False), value=__r.get('column', 'none')),
            gradio.update(visible=__r.get('visible', False), value=__r.get('content', '')),
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
        __rows.append({'visible': False, 'role': 'user', 'channel': 'final', 'column': 'none', 'content': ''})
    # update state and components
    return __rows, *render_input_rows(__rows)

# EVENTS #######################################################################

def update_input_cache(cache: list, index: int, value: any, field: str) -> list:
    __cache = list(cache)
    __cache[index][field] = value
    return __cache

def update_role_cache(cache: list, index: int, value: any) -> list:
    return update_input_cache(cache=cache, index=int(index), value=str(value), field='role')

def update_channel_cache(cache: list, index: int, value: any) -> list:
    return update_input_cache(cache=cache, index=int(index), value=str(value), field='channel')

def update_column_cache(cache: list, index: int, value: any) -> list:
    return update_input_cache(cache=cache, index=int(index), value=str(value), field='column')

def update_content_cache(cache: list, index: int, value: any) -> list:
    return update_input_cache(cache=cache, index=int(index), value=str(value), field='content')

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
        # load the tokenizer
        __tokenizer = psaiops.combine.lib.get_tokenizer(name=model, device='cpu')
        # create the UI
        __inputs.update(create_layout(intro=intro, limit=limit))
        # init the state
        __inputs.update(create_state(limit=limit))
        # apply the configuration
        __format = update_table_data(tokenizer=__tokenizer)
        # show hidden row
        __inputs['show_block'].click(
            fn=show_input_row,
            inputs=[__inputs['cache_block']],
            outputs=[__inputs['cache_block']] + get_input_rows(inputs=__inputs, limit=limit),
            queue=False,
            show_progress='hidden')
        # update the table TODO
        __inputs['details_tab'].select(
            fn=__format,
            inputs=[__inputs[f'content_{__i}_block'] for __i in range(limit)] + [__inputs['output_block']],
            outputs=__inputs['table_block'],
            queue=False,
            show_progress='hidden')
        # link each row of inputs to the cache
        for __i in range(limit):
            # update the target role in the cache
            __inputs[f'role_{__i}_block'].change(
                fn=update_role_cache,
                inputs=[__inputs['cache_block'], gradio.State(__i), __inputs[f'role_{__i}_block']],
                outputs=__inputs['cache_block'],
                queue=False,
                show_progress='hidden')
            # update the target channel in the cache
            __inputs[f'channel_{__i}_block'].change(
                fn=update_channel_cache,
                inputs=[__inputs['cache_block'], gradio.State(__i), __inputs[f'channel_{__i}_block']],
                outputs=__inputs['cache_block'],
                queue=False,
                show_progress='hidden')
            # update the target column in the cache
            __inputs[f'column_{__i}_block'].change(
                fn=update_column_cache,
                inputs=[__inputs['cache_block'], gradio.State(__i), __inputs[f'column_{__i}_block']],
                outputs=__inputs['cache_block'],
                queue=False,
                show_progress='hidden')
            # update the target content in the cache
            __inputs[f'content_{__i}_block'].change(
                fn=update_content_cache,
                inputs=[__inputs['cache_block'], gradio.State(__i), __inputs[f'content_{__i}_block']],
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
