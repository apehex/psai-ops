import functools

import gradio
import torch
import torch.cuda

import psaiops.common.model
import psaiops.common.tokenizer
import psaiops.score.attention.lib

# META #########################################################################

STYLE = '''.white-text span { color: white; }'''
TITLE = '''Attention Scoring'''
INTRO = '''Score each token according to the weights of the attention layers.\nUnder construction, only "openai/gpt-oss-20b" is available for now.'''

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
        'topp_block': __topp}

# TARGET #######################################################################

def create_target_block() -> dict:
    __target = gradio.Radio(label='Score', value='Inputs', choices=['Inputs', 'Everything'], scale=1, interactive=True)
    return {'target_block': __target}

# DISPLAY ######################################################################

# def create_display_block() -> dict:
#     __display = gradio.Radio(label='Display', value='Tokens', choices=['Tokens', 'Indexes'], scale=1, interactive=True)
#     return {'display_block': __display}

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

def update_layer_range(value: float, model: str) -> dict:
    return gradio.update(maximum=35, value=min(35, int(value))) if '120b' in model else gradio.update(maximum=23, value=min(23, int(value)))

def update_position_range(value: float, tokens: float) -> dict:
    return gradio.update(maximum=int(tokens) - 1, value=min(int(tokens) - 1, int(value)))

def update_computation_state(
    token_num: float,
    topk_num: float,
    topp_num: float,
    token_idx: float,
    layer_idx: float,
    head_idx: float,
    prompt_str: str,
    device_str: str,
    model_obj: object,
    tokenizer_obj: object,
) -> tuple:
    # sanitize the inputs
    __token_num = max(1, min(128, int(token_num)))
    __topk_num = max(1, min(8, int(topk_num)))
    __topp_num = max(0.0, min(1.0, float(topp_num)))
    __token_idx = max(-1, min(__token_num, int(token_idx)))
    __layer_idx = max(-1, int(layer_idx))
    __head_idx = max(-1, int(head_idx))
    __prompt_str = prompt_str.strip()
    __device_str = device_str if (device_str in ['cpu', 'cuda']) else 'cpu'
    # exit if some values are missing
    if (not __prompt_str) or (model_obj is None) or (tokenizer_obj is None):
        return ([], [], [], torch.empty(0))
    # handle all exceptions at once
    try:
        # dictionary {'input_ids': _, 'attention_mask': _}
        __input_data = psaiops.common.tokenizer.preprocess_token_ids(
            tokenizer_obj=tokenizer_obj,
            prompt_str=__prompt_str,
            device_str=__device_str)
        # parse the inputs
        __input_dim = int(__input_data['input_ids'].shape[-1])
        # tensor (1, T)
        __output_data = psaiops.common.model.generate_token_ids(
            model_obj=model_obj,
            input_ids=__input_data['input_ids'],
            attention_mask=__input_data['attention_mask'],
            token_num=__token_num,
            topk_num=__topk_num,
            topp_num=__topp_num)
        # tensor (L, S, H, T, T)
        __attention_data = psaiops.score.attention.lib.compute_attention_weights(
            model_obj=model_obj,
            token_obj=__output_data)
        # reduce the layer, sample, head and output token axes => tensor (T,)
        __score_data = psaiops.score.attention.lib.reduce_attention_weights(
            attention_data=__attention_data,
            token_idx=__token_idx,
            layer_idx=__layer_idx,
            head_idx=__head_idx,
            input_dim=__input_dim)
        # translate the scores into integer labels
        __labels = psaiops.score.attention.lib.postprocess_attention_scores(
            attention_data=__score_data,
            input_dim=__input_dim,
            token_idx=__token_idx)
        # detokenize the IDs
        __tokens = psaiops.common.tokenizer.postprocess_token_ids(
            tokenizer_obj=tokenizer_obj,
            token_obj=__output_data)
        # update each component => (input, output, attention, highligh) states
        return (
            list(zip(__tokens, __labels)),
            __tokens[:__input_dim],
            __tokens[__input_dim:],
            __attention_data,)
    except:
        raise Exception('Attention generation aborted with an error.')

def update_text_highlight(
    token_idx: float,
    layer_idx: float,
    head_idx: float,
    input_data: list,
    output_data: list,
    attention_data: torch.Tensor,
) -> list:
    # sanitize the inputs
    __input_data = input_data or []
    __output_data = output_data or []
    __attention_data = torch.empty(0) if (attention_data is None) else attention_data
    __input_dim = len(__input_data)
    __output_dim = len(__output_data)
    __token_idx = max(-1, min(__output_dim, int(token_idx)))
    __layer_idx = max(-1, int(layer_idx))
    __head_idx = max(-1, int(head_idx))
    # exit if the data has not yet been computed
    if (not __input_data) or (not __output_data) or (attention_data is None) or (len(attention_data) == 0):
        return gradio.update()
    # handle all exceptions at once
    try:
        # concat input and output tokens
        __tokens = __input_data + __output_data
        # reduce the layer, sample, head and output token axes => tensor (T,)
        __scores = psaiops.score.attention.lib.reduce_attention_weights(
            attention_data=__attention_data,
            token_idx=__token_idx,
            layer_idx=__layer_idx,
            head_idx=__head_idx,
            input_dim=__input_dim)
        # translate the scores into integer labels
        __labels = psaiops.score.attention.lib.postprocess_attention_scores(
            attention_data=__scores,
            input_dim=__input_dim,
            token_idx=__token_idx)
        # update the component with [(token, label), ...]
        return list(zip(__tokens, __labels))
    except:
        raise Exception('Attention reduction aborted with an error.')

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, style: str=STYLE, model: str=MODEL) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        # load the model
        __device = 'cuda' if torch.cuda.is_available() else 'cpu'
        __model = psaiops.common.model.get_model(name=model, device=__device)
        __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=model, device=__device)
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
