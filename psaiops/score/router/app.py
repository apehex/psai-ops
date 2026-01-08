import functools

import gradio
import torch
import torch.cuda
import matplotlib.pyplot

import psaiops.common.model
import psaiops.common.tokenizer
import psaiops.score.router.lib

# META #########################################################################

STYLE = '''.white-text span { color: white; }'''
TITLE = '''Router Scoring'''
INTRO = '''Plot the logits of the router for a given prompt.\nUnder construction, only "openai/gpt-oss-20b" is available for now.'''

MODEL = 'openai/gpt-oss-20b'

# COLORS #######################################################################

def create_selection_cmap() -> dict:
    return {
        '0': '#000000',
        '1': '#004444',
        '2': '#444400',
        '3': '#440044',}

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

# INPUTS #######################################################################

def create_inputs_block() -> dict:
    __input = gradio.Textbox(label='Prompt', value='', placeholder='A string of tokens to score.', lines=4, scale=1, show_copy_button=True, interactive=True)
    return {'input_block': __input}

# PLOTS ########################################################################

def create_plot_block(label: str='Router', prefix: str='') -> dict:
    __plot = gradio.Plot(label=label, scale=1)
    return {prefix + 'plot_block': __plot,}

# OUTPUTS ######################################################################

def create_highlight_block(label: str='Output', prefix: str='', cmap: dict=create_selection_cmap()) -> dict:
    __output = gradio.HighlightedText(label=label, value='', scale=1, interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=False, color_map=cmap, elem_classes='white-text')
    return {prefix + 'highlight_block': __output}

# SELECT #######################################################################

def create_selection_block(label: str='Token', prefix: str='') -> dict:
    # __play = gradio.Button('>', variant='primary', size='lg', scale=1, interactive=True)
    __position = gradio.Slider(label=label, value=-1, minimum=-1, maximum=15, step=1, scale=1, interactive=True) # info='-1 to average on all tokens'
    return {prefix + 'position_block': __position,}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __process = gradio.Button('Process', variant='primary', size='lg', scale=1, interactive=True)
    return {'process_block': __process,}

# STATE ########################################################################

def create_state() -> dict:
    return {
        'output_state': gradio.State(None),
        'router_state': gradio.State(None),}

# LAYOUT #######################################################################

def create_layout(intro: str=INTRO) -> dict:
    __fields = {}
    __fields.update(create_intro_block(intro=intro))
    with gradio.Tabs():
        with gradio.Tab('Score Tokens') as __main_tab:
            __fields.update({'main_tab': __main_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_inputs_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_highlight_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_plot_block(label='Left', prefix='left_'))
                __fields.update(create_plot_block(label='Right', prefix='right_'))
            with gradio.Row(equal_height=True):
                __fields.update(create_selection_block(label='Token', prefix='left_'))
                __fields.update(create_selection_block(label='Token', prefix='right_'))
            with gradio.Row(equal_height=True):
                __fields.update(create_actions_block())
        with gradio.Tab('Settings') as __settings_tab:
            __fields.update({'settings_tab': __settings_tab})
            with gradio.Column(scale=1):
                with gradio.Row(equal_height=True):
                    __fields.update(create_model_block())
                with gradio.Row(equal_height=True):
                    __fields.update(create_sampling_block())
    return __fields

# EVENTS #######################################################################

def update_position_range(
    current_val: float,
    token_num: float,
    output_data: torch.Tensor,
) -> dict:
    # exit if values are missing
    if (current_val is None) or (token_num is None):
        return None
    # take the generated tokens into account
    __max = int(token_num) - 1 if (output_data is None) else int(output_data.shape[-1])
    # keep the previous value if possible
    __val = min(int(current_val), __max)
    # return a gradio update dictionary
    return gradio.update(maximum=__max, value=__val)

# GENERATE #####################################################################

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
    __token_num = max(1, min(128, int(token_num)))
    __topk_num = max(1, min(8, int(topk_num)))
    __topp_num = max(0.0, min(1.0, float(topp_num)))
    __prompt_str = prompt_str.strip()
    __device_str = device_str if (device_str in ['cpu', 'cuda']) else 'cpu'
    # exit if some values are missing
    if (not __prompt_str) or (model_obj is None) or (tokenizer_obj is None) or (token_num is None) or (topk_num is None) or (topp_num is None):
        return (torch.empty(0), torch.empty(0))
    # dictionary {'input_ids': _, 'attention_mask': _}
    __input_data = psaiops.common.tokenizer.preprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        prompt_str=__prompt_str,
        device_str=__device_str)
    # tensor (1, T)
    __output_data = psaiops.common.model.generate_token_ids(
        model_obj=model_obj,
        input_ids=__input_data['input_ids'],
        attention_mask=__input_data['attention_mask'],
        token_num=__token_num,
        topk_num=__topk_num,
        topp_num=__topp_num)
    # tensor (L, S, H, T, T)
    __router_data = psaiops.score.router.lib.compute_router_weights(
        model_obj=model_obj,
        token_data=__output_data)
    # update each component => (highlight, plot) states
    return (
        __output_data.cpu(),
        __router_data.cpu(),)

# HIGHLIGHT ####################################################################

def update_token_focus(
    left_idx: float,
    right_idx: float,
    output_data: torch.Tensor,
    tokenizer_obj: object,
) -> list:
    # exit if some values are missing
    if (tokenizer_obj is None) or (left_idx is None) or (right_idx is None) or (output_data is None) or (len(output_data) == 0):
        return None
    # detokenize the IDs
    __token_str = psaiops.common.tokenizer.postprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        token_data=output_data)
    # list of string classes
    __token_cls = psaiops.score.router.lib.postprocess_focus_cls(
        left_idx=int(left_idx),
        right_idx=int(right_idx),
        token_dim=len(__token_str))
    # pairs of token and class
    return list(zip(__token_str, __token_cls))

# PLOTS ########################################################################

def update_router_plot(
    token_idx: float,
    router_data: torch.Tensor,
) -> tuple:
    # exit if some values are missing
    if (token_idx is None) or (router_data is None) or (len(router_data) == 0):
        return None
    # reduce the batch and token axes => tensor (L, E)
    __plot_data = psaiops.score.router.lib.reduce_router_weights(
        router_data=router_data,
        token_idx=int(token_idx),)
    # translate the scores into integer labels
    __plot_data = psaiops.score.router.lib.postprocess_router_weights(
        router_data=__plot_data,)
    # plot the data
    __figure, __axes = matplotlib.pyplot.subplots()
    __axes.imshow(__plot_data.float().numpy(), vmin=0.0, vmax=1.0, cmap='viridis')
    __figure.tight_layout()
    # remove the figure for the pyplot register for garbage collection
    matplotlib.pyplot.close(__figure)
    # update each component => (highlight, plot) states
    return __figure

# APP ##########################################################################

def create_app(compute: callable, highlight: callable, title: str=TITLE, intro: str=INTRO, style: str=STYLE, model: str=MODEL) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        # create the UI
        __fields.update(create_layout(intro=intro))
        # init the state
        __fields.update(create_state())
        # update the data after clicking process
        __fields['process_block'].click(
            fn=compute,
            inputs=[__fields[__k] for __k in ['tokens_block', 'topk_block', 'topp_block', 'input_block']],
            outputs=[__fields[__k] for __k in ['output_state', 'router_state']],
            queue=False,
            show_progress='full'
        ).then(
        # update the range of the position sliders when the output changes
            fn=update_position_range,
            inputs=[__fields[__k] for __k in ['left_position_block', 'tokens_block', 'output_state']],
            outputs=__fields['left_position_block'],
            queue=False,
            show_progress='hidden'
        ).then(
            fn=update_position_range,
            inputs=[__fields[__k] for __k in ['right_position_block', 'tokens_block', 'output_state']],
            outputs=__fields['right_position_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the token highlight when the output data changes
            fn=highlight,
            inputs=[__fields[__k] for __k in ['left_position_block', 'right_position_block', 'output_state']],
            outputs=__fields['highlight_block'],
            queue=False,
            show_progress='full'
        ).then(
        # update the plot when the router data changes
            fn=update_router_plot,
            inputs=[__fields[__k] for __k in ['left_position_block', 'router_state']],
            outputs=__fields['left_plot_block'],
            queue=False,
            show_progress='full'
        ).then(
            fn=update_router_plot,
            inputs=[__fields[__k] for __k in ['right_position_block', 'router_state']],
            outputs=__fields['right_plot_block'],
            queue=False,
            show_progress='full')
        # update the range of the position sliders when the settings change
        __fields['tokens_block'].change(
            fn=update_position_range,
            inputs=[__fields[__k] for __k in ['left_position_block', 'tokens_block', 'output_state']],
            outputs=__fields['left_position_block'],
            queue=False,
            show_progress='hidden'
        ).then(
            fn=update_position_range,
            inputs=[__fields[__k] for __k in ['right_position_block', 'tokens_block', 'output_state']],
            outputs=__fields['right_position_block'],
            queue=False,
            show_progress='hidden')
        # update the left plot when the focus changes
        __fields['left_position_block'].change(
            fn=update_router_plot,
            inputs=[__fields[__k] for __k in ['left_position_block', 'router_state']],
            outputs=__fields['left_plot_block'],
            queue=False,
            show_progress='hidden')
        # update the left plot when the focus changes
        __fields['right_position_block'].change(
            fn=update_router_plot,
            inputs=[__fields[__k] for __k in ['right_position_block', 'router_state']],
            outputs=__fields['right_plot_block'],
            queue=False,
            show_progress='hidden')
        # update the token highlight when the token focus changes
        __fields['left_position_block'].change(
            fn=highlight,
            inputs=[__fields[__k] for __k in ['left_position_block', 'right_position_block', 'output_state']],
            outputs=__fields['highlight_block'],
            queue=False,
            show_progress='hidden')
        __fields['right_position_block'].change(
            fn=highlight,
            inputs=[__fields[__k] for __k in ['left_position_block', 'right_position_block', 'output_state']],
            outputs=__fields['highlight_block'],
            queue=False,
            show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    # load the model
    __device = 'cuda' if torch.cuda.is_available() else 'cpu'
    __model = psaiops.common.model.get_model(name=MODEL, device=__device)
    __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=MODEL, device=__device)
    # adapt the event handlers
    __compute = functools.partial(update_computation_state, model_obj=__model, tokenizer_obj=__tokenizer, device_str=__device)
    __highlight = functools.partial(update_token_focus, tokenizer_obj=__tokenizer)
    # the event handlers are created outside so that they can be wrapped with `spaces.GPU` if necessary
    __app = create_app(compute=__compute, highlight=__highlight)
    __app.launch(share=True, debug=True)
