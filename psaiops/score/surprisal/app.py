import functools

import gradio
import numpy
import torch
import torch.cuda
import matplotlib.pyplot

import psaiops.common.model
import psaiops.common.tokenizer
import psaiops.score.residual.lib

# META #########################################################################

STYLE = '''.white-text span { color: white; }'''
TITLE = '''Visualization Of Residuals'''
INTRO = '''Plot the hidden states for a given prompt.\nUnder construction, only "openai/gpt-oss-20b" is available for now.'''

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
    __3d = gradio.Slider(label='3D', value=1, minimum=0, maximum=1, step=1, scale=1, interactive=True)
    __points = gradio.Slider(label='Points', value=128, minimum=32, maximum=2880, step=32, scale=1, interactive=True)
    return {
        'axes_block': __3d,
        'points_block': __points,}

# INPUTS #######################################################################

def create_inputs_block(label: str='Prompt') -> dict:
    __input = gradio.Textbox(label=label, value='', placeholder='A string of tokens to score.', lines=4, scale=1, interactive=True)
    return {'input_block': __input}

# PLOTS ########################################################################

def create_plot_block(label: str='Residuals', prefix: str='') -> dict:
    __plot = gradio.Plot(label=label, scale=1)
    return {prefix + 'plot_block': __plot,}

# OUTPUTS ######################################################################

def create_highlight_block(label: str='Output', prefix: str='', cmap: dict=create_selection_cmap()) -> dict:
    __output = gradio.HighlightedText(label=label, value='', scale=1, interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=False, color_map=cmap, elem_classes='white-text')
    return {prefix + 'highlight_block': __output}

# SELECT #######################################################################

def create_token_selection_block(label: str='Token', prefix: str='') -> dict:
    # __play = gradio.Button('>', variant='primary', size='lg', scale=1, interactive=True)
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
        with gradio.Tab('Residuals') as __main_tab:
            __fields.update({'main_tab': __main_tab})
            with gradio.Row(equal_height=True):
                __fields.update(create_inputs_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_highlight_block())
            with gradio.Row(equal_height=True):
                __fields.update(create_plot_block(label='Left', prefix='left_'))
                __fields.update(create_plot_block(label='Right', prefix='right_'))
            with gradio.Row(equal_height=True):
                __fields.update(create_highlight_block(label='Score', prefix='left_', cmap=create_score_cmap()))
                __fields.update(create_highlight_block(label='Score', prefix='right_', cmap=create_score_cmap()))
            with gradio.Row(equal_height=True):
                __fields.update(create_token_selection_block(label='Token', prefix='left_'))
                __fields.update(create_token_selection_block(label='Token', prefix='right_'))
            with gradio.Row(equal_height=True):
                __fields.update(create_layer_selection_block(label='Layer', prefix='left_'))
                __fields.update(create_layer_selection_block(label='Layer', prefix='right_'))
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
    __output_data, __hidden_data = psaiops.score.residual.lib.generate_token_ids(
        model_obj=model_obj,
        input_ids=__input_data['input_ids'],
        attention_mask=__input_data['attention_mask'],
        token_num=__token_num,
        topk_num=__topk_num,
        topp_num=__topp_num)
    # tensor (1, L, I + O, H)
    __hidden_data = psaiops.score.residual.lib.merge_hidden_states(
        hidden_data=__hidden_data)
    # update each component => (highlight, plot) states
    return (
        __output_data.cpu().float(),
        __hidden_data.cpu().float(),)

# HIGHLIGHT ####################################################################

def update_token_focus(
    left_idx: float,
    right_idx: float,
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
    __token_cls = psaiops.score.residual.lib.postprocess_focus_cls(
        left_idx=int(left_idx),
        right_idx=int(right_idx),
        token_dim=len(__token_str))
    # pairs of token and class
    return list(zip(__token_str, __token_cls))

# SCORES #######################################################################

def update_token_scores(
    layer_idx: float,
    output_data: torch.Tensor,
    hidden_data: torch.Tensor,
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

# PLOT #########################################################################

def update_2d_plot(
    token_idx: float,
    layer_idx: float,
    hidden_data: torch.Tensor,
) -> tuple:
    # reduce the layer and token axes (B, L, T, E) => (B, E)
    __plot_data = psaiops.score.residual.lib.reduce_hidden_states(
        hidden_data=hidden_data,
        layer_idx=int(layer_idx),
        token_idx=int(token_idx),
        axes_idx=(1, 2))
    # rescale the data to [-1; 1] (B, E)
    __plot_data = psaiops.score.residual.lib.rescale_hidden_states(
        hidden_data=__plot_data)
    # reshape into a 3D tensor by folding E (B, E) => (B, W, H)
    __plot_data = psaiops.score.residual.lib.reshape_hidden_states(
        hidden_data=__plot_data,
        layer_idx=-1) # there is no layer axis
    # map the [-1; 1] activations to RGBA colors
    __plot_data = psaiops.score.residual.lib.color_hidden_states(
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

def update_3d_plot(
    token_idx: float,
    layer_idx: float,
    points_num: float,
    hidden_data: torch.Tensor,
) -> tuple:
    # reduce the token axis (B, L, T, E) => (B, L, E)
    __plot_data = psaiops.score.residual.lib.reduce_hidden_states(
        hidden_data=hidden_data,
        token_idx=int(token_idx),
        layer_idx=int(layer_idx),
        axes_idx=2)
    # rescale the data to [-1; 1] (B, L, E)
    __plot_data = psaiops.score.residual.lib.rescale_hidden_states(
        hidden_data=__plot_data)
    # mask the small activations to improve the plot readability
    __mask_data = psaiops.score.residual.lib.mask_hidden_states(
        hidden_data=__plot_data,
        topk_num=int(points_num) if int(layer_idx) == -1 else 2 * int(points_num))
    # reshape into a 3D tensor by folding E (B, L, E) => (B, W, H, L)
    __plot_data = psaiops.score.residual.lib.reshape_hidden_states(
        hidden_data=__plot_data,
        layer_idx=1)
    __mask_data = psaiops.score.residual.lib.reshape_hidden_states(
        hidden_data=__mask_data,
        layer_idx=1)
    # convert to numpy ndarrays
    __plot_data = __plot_data.numpy()
    __mask_data = __mask_data.numpy()
    # map the [-1; 1] activations to RGBA colors
    __rgb_data = psaiops.score.residual.lib.color_hidden_states(
        hidden_data=__plot_data)
    # map the [-1; 1] activations to point areas
    __area_data = psaiops.score.residual.lib.size_hidden_states(
        hidden_data=__plot_data,
        area_min=0.01,
        area_max=16.0,
        gamma_val=1.6)
    # format the first sample for a scatter plot
    __x, __y, __z = numpy.nonzero(__mask_data[0])
    __c = __rgb_data[0, __x, __y, __z]
    __s = __area_data[0, __x, __y, __z]
    # plot the first sample
    __figure = matplotlib.pyplot.figure()
    __axes = __figure.add_subplot(1, 1, 1, projection='3d')
    __axes.scatter(__x, __y, __z, c=__c, s=__s, marker='o', linewidths=0)
    __figure.tight_layout()
    # remove the figure for the pyplot register for garbage collection
    matplotlib.pyplot.close(__figure)
    # update each component => (highlight, plot) states
    return __figure

def update_hidden_plot(
    token_idx: float,
    layer_idx: float,
    axes_num: float,
    points_num: float,
    hidden_data: torch.Tensor,
) -> tuple:
    # exit if some values are missing
    if (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # plot the residuals of a given layer on a 2D heatmap
    if not axes_num: # 0.0 or 0
        return update_2d_plot(
            token_idx=token_idx,
            layer_idx=layer_idx,
            hidden_data=hidden_data)
    # by default, plot the residuals for all the layers in 3D
    return update_3d_plot(
        token_idx=token_idx,
        layer_idx=layer_idx,
        points_num=points_num,
        hidden_data=hidden_data)

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, style: str=STYLE, model: str=MODEL) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        # load the model
        __device = 'cuda' if torch.cuda.is_available() else 'cpu'
        __model = psaiops.common.model.get_model(name=model, device=__device)
        __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=model, device=__device)
        # adapt the event handlers
        __compute = functools.partial(update_computation_state, model_obj=__model, tokenizer_obj=__tokenizer, device_str=__device)
        __highlight = functools.partial(update_token_focus, tokenizer_obj=__tokenizer)
        __score = functools.partial(update_token_scores, tokenizer_obj=__tokenizer, model_obj=__model)
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
            fn=__highlight,
            inputs=[__fields[__k] for __k in ['left_position_block', 'right_position_block', 'output_state']],
            outputs=__fields['highlight_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the left token scores when the output data changes
            fn=__score,
            inputs=[__fields[__k] for __k in ['left_layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['left_highlight_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the right token scores when the output data changes
            fn=__score,
            inputs=[__fields[__k] for __k in ['right_layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['right_highlight_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the plot when the router data changes
            fn=update_hidden_plot,
            inputs=[__fields[__k] for __k in ['left_position_block', 'left_layer_block', 'axes_block', 'points_block', 'hidden_state']],
            outputs=__fields['left_plot_block'],
            queue=False,
            show_progress='hidden'
        ).then(
            fn=update_hidden_plot,
            inputs=[__fields[__k] for __k in ['right_position_block', 'right_layer_block', 'axes_block', 'points_block', 'hidden_state']],
            outputs=__fields['right_plot_block'],
            queue=False,
            show_progress='hidden')
        # update the range of the position slider when the settings change
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
            fn=update_hidden_plot,
            inputs=[__fields[__k] for __k in ['left_position_block', 'left_layer_block', 'axes_block', 'points_block', 'hidden_state']],
            outputs=__fields['left_plot_block'],
            queue=False,
            show_progress='hidden')
        __fields['left_layer_block'].change(
            fn=update_hidden_plot,
            inputs=[__fields[__k] for __k in ['left_position_block', 'left_layer_block', 'axes_block', 'points_block', 'hidden_state']],
            outputs=__fields['left_plot_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the left token scores when the focus changes
            fn=__score,
            inputs=[__fields[__k] for __k in ['left_layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['left_highlight_block'],
            queue=False,
            show_progress='hidden'
        )
        # update the right plot when the focus changes
        __fields['right_position_block'].change(
            fn=update_hidden_plot,
            inputs=[__fields[__k] for __k in ['right_position_block', 'right_layer_block', 'axes_block', 'points_block', 'hidden_state']],
            outputs=__fields['right_plot_block'],
            queue=False,
            show_progress='hidden')
        __fields['right_layer_block'].change(
            fn=update_hidden_plot,
            inputs=[__fields[__k] for __k in ['right_position_block', 'right_layer_block', 'axes_block', 'points_block', 'hidden_state']],
            outputs=__fields['right_plot_block'],
            queue=False,
            show_progress='hidden'
        ).then(
        # update the right token scores when the focus changes
            fn=__score,
            inputs=[__fields[__k] for __k in ['right_layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['right_highlight_block'],
            queue=False,
            show_progress='hidden'
        )
        # update the token highlight when the token focus changes
        __fields['left_position_block'].change(
            fn=__highlight,
            inputs=[__fields[__k] for __k in ['left_position_block', 'right_position_block', 'output_state']],
            outputs=__fields['highlight_block'],
            queue=False,
            show_progress='hidden')
        __fields['right_position_block'].change(
            fn=__highlight,
            inputs=[__fields[__k] for __k in ['left_position_block', 'right_position_block', 'output_state']],
            outputs=__fields['highlight_block'],
            queue=False,
            show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_app()
    __app.launch(share=True, debug=True)
