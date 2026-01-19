import functools

import gradio
import numpy
import torch
import torch.cuda
import matplotlib.pyplot

import psaiops.common.model
import psaiops.common.tokenizer
import psaiops.score.surprisal.lib

# META #########################################################################

STYLE = '''.white-text span { color: white; }'''
TITLE = '''Surprisal Scoring'''
INTRO = '''Plot the following metrics to measure how unexpected each token is:\n- the rank of each token among the output logits shows how likely it is according to the LLM\n- the KL divergence between the final residuals and those at depth L compares the contributions of the prefix and the suffix models'''

MODEL = 'openai/gpt-oss-20b'

# COLORS #######################################################################

def create_selection_cmap() -> dict:
    return {
        '0': '#000000',
        '1': '#004444',
        '2': '#444400',
        '3': '#440044',}

def create_score_cmap() -> dict:
    return {str(__i): '#{:02x}0000'.format(int(2.55 * __i)) for __i in range(101)}

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

# DATAVIZ ######################################################################

def create_visualization_block() -> dict:
    return {}

# INPUTS #######################################################################

def create_inputs_block(label: str='Prompt') -> dict:
    __input = gradio.Textbox(label=label, value='', placeholder='A string of tokens to score.', lines=4, scale=1, interactive=True)
    return {'input_block': __input}

# PLOTS ########################################################################

def create_plot_block(label: str='Residuals', prefix: str='') -> dict:
    __plot = gradio.Plot(label=label, scale=1)
    return {prefix + 'plot_block': __plot,}

# HIGHLIGHT ####################################################################

def create_highlight_block(label: str='Output', prefix: str='', cmap: dict=create_selection_cmap()) -> dict:
    __highlight = gradio.HighlightedText(label=label, value='', scale=1, interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=False, color_map=cmap, elem_classes='white-text')
    return {prefix + 'highlight_block': __highlight}

# SELECT #######################################################################

def create_token_selection_block(label: str='Token', prefix: str='') -> dict:
    __position = gradio.Slider(label=label, value=-1, minimum=-1, maximum=15, step=1, scale=1, interactive=True) # info='-1 to average on all tokens'
    return {prefix + 'position_block': __position,}

def create_layer_selection_block(label: str='Layer', prefix: str='') -> dict:
    __layer = gradio.Slider(label=label, value=-1, minimum=-1, maximum=23, step=1, scale=1, interactive=True) # info='-1 to average on all layers'
    return {prefix + 'layer_block': __layer,}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __process = gradio.Button('Process', variant='primary', size='lg', scale=1, interactive=True)
    return {'process_block': __process,}

# STATE ########################################################################

def create_state() -> dict:
    return {
        'output_state': gradio.State(None),
        'hidden_state': gradio.State(None),}

# LAYOUT #######################################################################

def create_layout(intro: str=INTRO) -> dict:
    __fields = {}
    __fields.update(create_intro_block(intro=intro))
    with gradio.Tabs():
        with gradio.Tab('Surprisal') as __main_tab:
            __fields.update({'main_tab': __main_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_inputs_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_highlight_block(label='Prob By Token', prefix='prob_', cmap=create_score_cmap()))
                __fields.update(create_plot_block(label='Prob By Position', prefix='prob_'))
            with gradio.Row(equal_height=True):
                __fields.update(create_highlight_block(label='Rank By Token', prefix='rank_', cmap=create_score_cmap()))
                __fields.update(create_plot_block(label='Rank By Position', prefix='rank_'))
            with gradio.Row(equal_height=True):
                __fields.update(create_highlight_block(label='KL By Token', prefix='jsd_', cmap=create_score_cmap()))
                __fields.update(create_plot_block(label='KL By Layer', prefix='jsd_'))
            with gradio.Row(equal_height=True):
                __fields.update(create_token_selection_block(label='Token'))
                __fields.update(create_layer_selection_block(label='Layer'))
            with gradio.Row(equal_height=True):
                __fields.update(create_actions_block())
        with gradio.Tab('Settings') as __settings_tab:
            __fields.update({'settings_tab': __settings_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_model_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_sampling_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_visualization_block())
    return __fields

# EVENTS #######################################################################

def update_position_range(
    current_val: float,
    token_num: float,
    output_data: torch.Tensor,
) -> dict:
    # take the generated tokens into account
    __max = int(token_num) - 1 if (output_data is None) else int(output_data.shape[-1])
    # keep the previous value if possible
    __val = min(int(current_val), __max)
    # return a gradio update dictionary
    return gradio.update(maximum=__max, value=__val)

# HIGHLIGHT ####################################################################

def update_token_focus(
    token_idx: float,
    output_data: torch.Tensor,
    tokenizer_obj: object,
) -> list:
    # exit if some values are missing
    if (output_data is None) or (len(output_data) == 0):
        return None
    # detokenize the IDs
    __token_str = psaiops.common.tokenizer.postprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        token_data=output_data)
    # list of string classes
    __token_cls = psaiops.score.surprisal.lib.postprocess_focus_cls(
        token_idx=int(token_idx),
        token_dim=len(__token_str))
    # pairs of token and class
    return list(zip(__token_str, __token_cls))

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
    if (not __prompt_str) or (model_obj is None) or (tokenizer_obj is None):
        return (torch.empty(0), torch.empty(0))
    # dictionary {'input_ids': _, 'attention_mask': _}
    __input_data = psaiops.common.tokenizer.preprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        prompt_str=__prompt_str,
        device_str=__device_str)
    # tensor (1, T) and O * L * (1, I, H)
    __output_data, __hidden_data = psaiops.score.surprisal.lib.generate_token_ids(
        model_obj=model_obj,
        input_ids=__input_data['input_ids'],
        attention_mask=__input_data['attention_mask'],
        token_num=__token_num,
        topk_num=__topk_num,
        topp_num=__topp_num)
    # tensor (1, L, I + O, H)
    __hidden_data = psaiops.score.surprisal.lib.merge_hidden_states(
        hidden_data=__hidden_data)
    # update each component => (highlight, plot) states
    return (
        __output_data.cpu().float(),
        __hidden_data.cpu().float(),)

# PROB SCORE ###################################################################

def update_prob_scores(
    token_idx: float,
    layer_idx: float,
    output_data: object,
    hidden_data: object,
    tokenizer_obj: object,
    model_obj: object,
) -> list:
    return []

def update_prob_plot(
    token_idx: float,
    layer_idx: float,
    hidden_data: object,
    model_obj: object,
) -> object:
    return None

# PROB SCORE ###################################################################

def update_rank_scores(
    token_idx: float,
    layer_idx: float,
    output_data: object,
    hidden_data: object,
    tokenizer_obj: object,
    model_obj: object,
) -> list:
    return []

def update_rank_plot(
    token_idx: float,
    layer_idx: float,
    hidden_data: object,
    model_obj: object,
) -> object:
    return None

# JSD SCORE ####################################################################

def update_jsd_scores(
    token_idx: float,
    layer_idx: float,
    output_data: object,
    hidden_data: object,
    tokenizer_obj: object,
    model_obj: object,
) -> list:
    # exit if some values are missing
    if (output_data is None) or (len(output_data) == 0) or (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # parse the model meta
    __device_str = model_obj.lm_head.weight.device
    __dtype_obj = model_obj.lm_head.weight.dtype
    # detokenize the IDs
    __token_str = psaiops.common.tokenizer.postprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        token_data=output_data)
    # select the relevant hidden states
    __final_states = hidden_data[0, -1, :, :].to(device=__device_str, dtype=__dtype_obj)
    __layer_states = hidden_data[0, int(layer_idx), :, :].to(device=__device_str, dtype=__dtype_obj)
    # compute the logits
    __final_logits = model_obj.lm_head(__final_states).detach().cpu() # already normalized
    __layer_logits = model_obj.lm_head(model_obj.model.norm(__layer_states)).detach().cpu()
    # compute the JSD metric
    __token_jsd = jsd_from_logits(final_logits=__final_logits, prefix_logits=__layer_logits)
    # scale into a [0; 100] label
    __token_cls = postprocess_score_cls(score_data=__token_jsd)
    # color each token according to the distance between the distribution at layer L and the final distribution
    return list(zip(__token_str, __token_cls))

def update_jsd_plot(
    token_idx: float,
    layer_idx: float,
    hidden_data: object,
    model_obj: object,
) -> object:
    # exit if some values are missing
    if (token_idx is None) or (layer_idx is None) or (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # reduce the layer and token axes (B, L, T, E) => (B, E)
    __plot_data = psaiops.score.surprisal.lib.reduce_hidden_states(
        hidden_data=hidden_data,
        layer_idx=int(layer_idx),
        token_idx=int(token_idx),
        axes_idx=(1, 2))
    # rescale the data to [-1; 1] (B, E)
    __plot_data = psaiops.score.surprisal.lib.rescale_hidden_states(
        hidden_data=__plot_data)
    # reshape into a 3D tensor by folding E (B, E) => (B, W, H)
    __plot_data = psaiops.score.surprisal.lib.reshape_hidden_states(
        hidden_data=__plot_data,
        layer_idx=-1) # there is no layer axis
    # map the [-1; 1] activations to RGBA colors
    __plot_data = psaiops.score.surprisal.lib.color_hidden_states(
        hidden_data=__plot_data.numpy())
    # plot the first sample
    __figure = matplotlib.pyplot.figure()
    __axes = __figure.add_subplot(1, 1, 1)
    __axes.imshow(__plot_data[0], vmin=0.0, vmax=1.0, cmap='viridis')
    __figure.tight_layout()
    # remove the figure for the pyplot register for garbage collection
    matplotlib.pyplot.close(__figure)
    # update each component => (highlight, plot) states
    return __figure

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, model: str=MODEL) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(title=title) as __app:
        # load the model
        __device = 'cuda' if torch.cuda.is_available() else 'cpu'
        __model = psaiops.common.model.get_model(name=model, device=__device)
        __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=model, device=__device)
        # adapt the event handlers
        # __highlight = functools.partial(update_token_focus, tokenizer_obj=__tokenizer)
        __compute = functools.partial(update_computation_state, model_obj=__model, tokenizer_obj=__tokenizer, device_str=__device)
        __prob_score = functools.partial(update_prob_scores, tokenizer_obj=__tokenizer, model_obj=__model)
        __prob_plot = functools.partial(update_prob_plot, model_obj=__model)
        __rank_score = functools.partial(update_rank_scores, tokenizer_obj=__tokenizer, model_obj=__model)
        __rank_plot = functools.partial(update_rank_plot, model_obj=__model)
        __jsd_score = functools.partial(update_jsd_scores, tokenizer_obj=__tokenizer, model_obj=__model)
        __jsd_plot = functools.partial(update_jsd_plot, model_obj=__model)
        # create the UI
        __fields.update(create_layout(intro=intro))
        # init the state
        __fields.update(create_state())
        # update the data after clicking process
        __fields['process_block'].click(
            fn=__compute,
            inputs=[__fields[__k] for __k in ['tokens_block', 'topk_block', 'topp_block', 'input_block']],
            outputs=[__fields[__k] for __k in ['output_state', 'hidden_state']],
            queue=False,
            show_progress='full'
        ).then(
        # update the range of the position sliders when the output changes
            fn=update_position_range,
            inputs=[__fields[__k] for __k in ['position_block', 'tokens_block', 'output_state']],
            outputs=__fields['position_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the probability scores when the data changes
            fn=__prob_score,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['prob_highlight_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the probability plot when the data changes
            fn=__prob_plot,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'hidden_state']],
            outputs=__fields['prob_plot_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the rank scores when the data changes
            fn=__rank_score,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['rank_highlight_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the rank plot when the data changes
            fn=__rank_plot,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'hidden_state']],
            outputs=__fields['rank_plot_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the JSD scores when the data changes
            fn=__jsd_score,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['jsd_highlight_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the JSD plot when the data changes
            fn=__jsd_plot,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'hidden_state']],
            outputs=__fields['jsd_plot_block'],
            queue=False,
            show_progress='hidden')
        # update the range of the position slider when the settings change
        __fields['tokens_block'].change(
            fn=update_position_range,
            inputs=[__fields[__k] for __k in ['position_block', 'tokens_block', 'output_state']],
            outputs=__fields['position_block'],
            queue=False,
            show_progress='hidden')
        # update the JSD token scores when the focus changes
        __fields['layer_block'].change(
            fn=__jsd_score,
            inputs=[__fields[__k] for __k in ['layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['highlight_block'],
            queue=False,
            show_progress='hidden')
        # update the JSD plot when the focus changes
        __fields['position_block'].change(
            fn=__jsd_plot,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'hidden_state']],
            outputs=__fields['jsd_plot_block'],
            queue=False,
            show_progress='hidden')
        __fields['layer_block'].change(
            fn=__jsd_plot,
            inputs=[__fields[__k] for __k in ['position_block', 'layer_block', 'hidden_state']],
            outputs=__fields['jsd_plot_block'],
            queue=False,
            show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_app()
    __app.launch(theme=gradio.themes.Soft(), css=STYLE, share=True, debug=True)
