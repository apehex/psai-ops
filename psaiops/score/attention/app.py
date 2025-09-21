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
    __intro = gradio.Markdown(intro, scale=1)
    return {'intro_block': __intro}

# MODEL ########################################################################

def create_model_block() -> dict:
    __model_dd = gradio.Dropdown(label='Model', value='openai/gpt-oss-20b', choices=['openai/gpt-oss-20b'], scale=1, allow_custom_value=False, multiselect=False, interactive=True) # 'openai/gpt-oss-120b'
    __layer_sl = gradio.Slider(label='Layer Depth', value=12, minimum=-1, maximum=23, step=1, scale=1, interactive=True) # info='-1 to average on all layers'
    __head_sl = gradio.Slider(label='Attention Head', value=-1, minimum=-1, maximum=63, step=1, scale=1, interactive=True) # info='-1 to average on all heads'
    __model_dd.change(fn=update_layer_range, inputs=[__layer_sl, __model_dd], outputs=__layer_sl, scale=1, queue=False, show_progress='hidden')
    return {
        'model_block': __model_dd,
        'layer_block': __layer_sl,
        'head_block': __head_sl}

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
    __target = gradio.Radio(label='Score', value='Inputs', choices=['Inputs', 'Outputs', 'Both'], scale=1, interactive=True)
    return {'target_block': __target}

# DISPLAY ######################################################################

def create_display_block() -> dict:
    __display = gradio.Radio(label='Display', value='Tokens', choices=['Tokens', 'Indexes'], scale=1, interactive=True)
    return {'display_block': __display}

# INPUTS #######################################################################

def create_inputs_block() -> dict:
    __input = gradio.Textbox(label='Prompt', value='', placeholder='A string of tokens to score.', lines=4, scale=1, show_copy_button=True, interactive=True)
    return {'input_block': __input}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.HighlightedText(label='Scores', value='', scale=1, interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=True, color_map=create_color_map(), elem_classes='white-text')
    return {'output_block': __output}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __process = gradio.Button('Process', variant='primary', size='lg', scale=1, interactive=True)
    __position = gradio.Slider(label='Position', value=-1, minimum=-1, maximum=128, step=1, scale=1, interactive=True) # info='-1 to average on all tokens'
    return {
        'process_block': __process,
        'position_block': __position}

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
                    __fields.update(create_display_block())
    return __fields

# EVENTS #######################################################################

def update_layer_range(value: float, model: str) -> dict:
    return gradio.update(maximum=35, value=min(35, int(value))) if '120b' in model else gradio.update(maximum=23, value=min(23, int(value)))

def update_position_range(value: float, tokens: list) -> dict:
    return gradio.update(maximum=len(tokens) - 1, value=min(len(tokens) - 1, int(value)))

def update_computation_state(
    token_num: float,
    topk_num: float,
    topp_num: float,
    prompt_str: str,
    device_str: str,
    model_obj: object,
    tokenizer_obj: object,
) -> tuple:
    # sanitize the inputs
    __limit = max(1, min(128, int(token_num)))
    __topk = max(1, min(128, int(token_num)))
    __topp = max(0.0, min(1.0, float(token_num)))
    __prompt = prompt_str.strip()
    __device = device_str if (device_str in ['cpu', 'cuda']) else 'cpu'
    # handle all exceptions at once
    try:
        # dictionary {'input_ids': _, 'attention_mask': _}
        __inputs = psaiops.score.attention.lib.preprocess_token_ids(
            tokenizer_obj=tokenizer_obj,
            prompt_str=__prompt,
            device_str=__device)
        # parse the inputs
        __input_dim = int(__inputs['input_ids'].shape[-1])
        # tensor (1, T)
        __outputs = psaiops.score.attention.lib.generate_token_ids(
            model_obj=model_obj,
            input_args=__inputs,
            token_num=__limit,
            topk_num=__topk,
            topp_num=__topp)
        # tensor (L, S, H, T, T)
        __attentions = psaiops.score.attention.lib.compute_attention_weights(
            model_obj=model_obj,
            token_obj=__outputs)
        # detokenize the IDs
        __tokens = psaiops.score.attention.lib.postprocess_token_ids(
            tokenizer_obj=tokenizer_obj,
            token_obj=__outputs)
        # update each component => (input, output, attention) states
        return (
            gradio.update(value=__tokens[:__input_dim]),
            gradio.update(value=__tokens[__input_dim:]),
            gradio.update(value=__attentions),)
    except:
        raise Exception('Attention generation aborted with an error.')
    finally:
        return (gradio.update(), gradio.update(), gradio.update())

def update_text_highlight(
    token_idx: float,
    layer_idx: float,
    head_idx: float,
    input_data: list,
    output_data: list,
    attention_data: torch.Tensor,
) -> dict:
    # sanitize the inputs
    __input_data = input_data or []
    __output_data = output_data or []
    __attention_data = attention_data or torch.empty(0)
    __input_dim = len(__input_data)
    __token_idx = max(0, min(__input_dim, int(token_idx)))
    __layer_idx = max(0, int(layer_idx))
    __head_idx = max(0, int(head_idx))
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
        return gradio.update(value=list(zip(__tokens, __labels)))
    except:
        raise Exception('Attention reduction aborted with an error.')
    finally:
        return gradio.update()

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
        # fetch the relevant fields
        __button_block, __position_block, __output_block = (__fields['process_block'], __fields['position_block'], __fields['output_block'])
        __output_state, __attention_state = (__fields['output_state'], __fields['attention_state'])
        # wire the input fields
        __button_block.click(
            fn=__compute,
            inputs=[__fields[__k] for __k in ['tokens_block', 'topk_block', 'topp_block', 'input_block']],
            outputs=[__fields[__k] for __k in ['input_state', 'output_state', 'attention_state']],
            queue=False,
            show_progress='full')
        __output_state.change(
            fn=update_position_range,
            inputs=[__position_block, __output],
            outputs=__position_block,
            queue=False,
            show_progress='hidden')
        __attention_state.change(
            fn=update_text_highlight,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'head_block', 'input_state', 'output_state', 'attention_state']],
            outputs=__output_block,
            queue=False,
            show_progress='hidden')
        __position_block.change(
            fn=update_text_highlight,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'head_block', 'input_state', 'output_state', 'attention_state']],
            outputs=__output_block,
            queue=False,
            show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_app()
    __app.launch(share=True, debug=True)
