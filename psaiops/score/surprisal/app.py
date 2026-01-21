import copy
import functools

import gradio
import numpy
import torch
import torch.cuda
import torch.nn.functional
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
    __topk = gradio.Slider(label='Top K', value=4, minimum=1, maximum=16, step=1, scale=1, interactive=True)
    __topp = gradio.Slider(label='Top P', value=0.9, minimum=0.0, maximum=1.0, step=0.05, scale=1, interactive=True)
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

def create_plot_block(label: str='Plot', prefix: str='') -> dict:
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
                __fields.update(create_plot_block(label='KL By Position (Fixed Layer)', prefix='jsd_'))
            with gradio.Row(equal_height=True):
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
        __output_data.cpu(),
        __hidden_data.cpu(),)

# PROB SCORE ###################################################################

def compute_prob_metrics(
    output_data: object,
    hidden_data: object,
    head_obj: object,
) -> object:
    # select the relevant tokens indices
    __indices = output_data[0]
    # select the relevant hidden states
    __logits = hidden_data[0, -1, :, :]
    # compute the logits
    __logits = torch.nn.functional.softmax(head_obj(__logits).detach().float(), dim=-1)
    # fetch the logits of the tokens chosen in the actual output
    return __logits.gather(dim=-1, index=__indices[1:].unsqueeze(-1)).squeeze(-1)

def update_prob_scores(
    output_data: object,
    hidden_data: object,
    tokenizer_obj: object,
    head_obj: object,
) -> list:
    # exit if some values are missing
    if (output_data is None) or (len(output_data) == 0) or (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # detokenize the IDs
    __token_str = psaiops.common.tokenizer.postprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        token_data=output_data)
    # compute the probabilities ofthe chosen tokens, in [0; V-1]
    __token_cls = compute_prob_metrics(output_data=output_data, hidden_data=hidden_data, head_obj=head_obj)
    # postprocess
    __token_cls = 1.0 - __token_cls.clamp(min=0.0, max=1.0)
    # scale into a [0; 100] label
    __token_cls = psaiops.score.surprisal.lib.postprocess_score_cls(score_data=__token_cls, scale_val=100.0)
    # pad with null class for the tokens which have no logit (IE the first token)
    __token_cls = max(0, len(__token_str) - len(__token_cls)) * ['0'] + __token_cls
    # color each token according to its rank in the LLM's predictions
    return list(zip(__token_str, __token_cls))

def update_prob_plot(
    output_data: object,
    hidden_data: object,
    head_obj: object,
) -> object:
    # exit if some values are missing
    if (output_data is None) or (len(output_data) == 0) or (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # compute the rank metric, in [0; V-1]
    __y = compute_prob_metrics(output_data=output_data, hidden_data=hidden_data, head_obj=head_obj)
    # rescale and convert the data
    __y = [0.0] + (100.0 - 100.0 * __y).numpy().tolist()
    # match the metrics with their token position
    __x = range(len(__y))
    # plot the first sample
    __figure = matplotlib.pyplot.figure()
    __axes = __figure.add_subplot(1, 1, 1)
    __axes.plot(__x, __y, linestyle='--', label='1 - p(token(i))')
    # display the legend and remove the extra padding
    __axes.legend()
    __figure.tight_layout()
    # remove the figure for the pyplot register for garbage collection
    matplotlib.pyplot.close(__figure)
    # update each component => (highlight, plot) states
    return __figure

# PROB SCORE ###################################################################

def compute_rank_metrics(
    output_data: object,
    hidden_data: object,
    head_obj: object,
) -> object:
    # select the relevant tokens indices
    __indices = output_data[0]
    # select the relevant hidden states
    __logits = hidden_data[0, -1, :, :]
    # compute the logits
    __logits = head_obj(__logits).detach().float()
    # fetch the logits of the tokens chosen in the actual output
    __chosen = __logits.gather(dim=-1, index=__indices[1:].unsqueeze(-1))
    # count the tokens with higher logits
    return (__logits > __chosen).int().sum(dim=-1)

def update_rank_scores(
    output_data: object,
    hidden_data: object,
    tokenizer_obj: object,
    head_obj: object,
) -> list:
    # exit if some values are missing
    if (output_data is None) or (len(output_data) == 0) or (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # detokenize the IDs
    __token_str = psaiops.common.tokenizer.postprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        token_data=output_data)
    # compute the rank metric, in [0; V-1]
    __token_cls = compute_rank_metrics(output_data=output_data, hidden_data=hidden_data, head_obj=head_obj)
    # postprocess
    __token_cls = __token_cls.clamp(min=0, max=100)
    # scale into a [0; 100] label
    __token_cls = psaiops.score.surprisal.lib.postprocess_score_cls(score_data=__token_cls, scale_val=1)
    # pad with null class for the tokens which have no logit (IE the first token)
    __token_cls = max(0, len(__token_str) - len(__token_cls)) * ['0'] + __token_cls
    # color each token according to its rank in the LLM's predictions
    return list(zip(__token_str, __token_cls))

def update_rank_plot(
    output_data: object,
    hidden_data: object,
    head_obj: object,
) -> object:
    # exit if some values are missing
    if (output_data is None) or (len(output_data) == 0) or (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # compute the rank metric, in [0; V-1]
    __y = compute_rank_metrics(output_data=output_data, hidden_data=hidden_data, head_obj=head_obj)
    # rescale and convert the data
    __y = [0] + __y.clamp(min=0, max=100).numpy().tolist()
    # match the metrics with their token position
    __x = range(len(__y))
    # plot the first sample
    __figure = matplotlib.pyplot.figure()
    __axes = __figure.add_subplot(1, 1, 1)
    __axes.plot(__x, __y, linestyle='--', label='r(token(i))')
    # display the legend and remove the extra padding
    __axes.legend()
    __figure.tight_layout()
    # remove the figure for the pyplot register for garbage collection
    matplotlib.pyplot.close(__figure)
    # update each component => (highlight, plot) states
    return __figure

# JSD SCORE ####################################################################

def compute_jsd_metrics(
    layer_idx: float,
    hidden_data: object,
    head_obj: object,
    norm_obj: object,
) -> object:
    # select the relevant hidden states
    __final_states = hidden_data[0, -1, :, :]
    __layer_states = hidden_data[0, int(layer_idx), :, :]
    # compute the logits
    __final_logits = head_obj(__final_states).detach().float()
    __layer_logits = head_obj(norm_obj(__layer_states)).detach().float()
    # compute the JSD metric, in [0; 1]
    return psaiops.score.surprisal.lib.jsd_from_logits(final_logits=__final_logits, prefix_logits=__layer_logits)

def update_jsd_scores(
    layer_idx: float,
    output_data: object,
    hidden_data: object,
    tokenizer_obj: object,
    head_obj: object,
    norm_obj: object,
) -> list:
    # exit if some values are missing
    if (layer_idx is None) or (output_data is None) or (len(output_data) == 0) or (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # detokenize the IDs
    __token_str = psaiops.common.tokenizer.postprocess_token_ids(
        tokenizer_obj=tokenizer_obj,
        token_data=output_data)
    # compute the JSD metric [0; 1]
    __token_cls = compute_jsd_metrics(layer_idx=layer_idx, hidden_data=hidden_data, head_obj=head_obj, norm_obj=norm_obj)
    # postprocess
    __token_cls = __token_cls.clamp(min=0.0, max=1.0)
    # scale into a [0; 100] label
    __token_cls = psaiops.score.surprisal.lib.postprocess_score_cls(score_data=__token_cls, scale_val=100.0)
    # pad with null class for the tokens which have no logit (IE the first token)
    __token_cls = max(0, len(__token_str) - len(__token_cls)) * ['0'] + __token_cls
    # color each token according to the distance between the distribution at layer L and the final distribution
    return list(zip(__token_str, __token_cls))

def update_jsd_plot(
    layer_idx: float,
    hidden_data: object,
    head_obj: object,
    norm_obj: object,
) -> object:
    __dim = int(hidden_data.shape[1])
    # init the plot
    __figure = matplotlib.pyplot.figure()
    __axes = __figure.add_subplot(1, 1, 1)
    # exit if some values are missing
    if (layer_idx is None) or (hidden_data is None) or (len(hidden_data) == 0):
        return None
    # stack the plots for each layer
    for __l in range(__dim):
        # compute the JSD metric, in [0; 1] => (T,)
        __y = compute_jsd_metrics(layer_idx=__l, hidden_data=hidden_data, head_obj=head_obj, norm_obj=norm_obj)
        # rescale and convert the data
        __y = [0.0] + (100.0 * __y).numpy().tolist()
        # match the metrics with their token position
        __x = range(len(__y))
        # plot the first sample
        __axes.plot(__x, __y, label=f'layer {__l}', linestyle='--' if (__l == int(layer_idx)) else ':')
    # remove the extra padding + show the legend
    __axes.legend()
    __figure.tight_layout()
    # remove the figure for the pyplot register for garbage collection
    matplotlib.pyplot.close(__figure)
    # update each component => (highlight, plot) states
    return __figure

# APP ##########################################################################

def create_app(
    compute: callable,
    prob_score: callable,
    prob_plot: callable,
    rank_score: callable,
    rank_plot: callable,
    jsd_score: callable,
    jsd_plot: callable,
    title: str=TITLE,
    intro: str=INTRO
) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(title=title) as __app:
        # create the UI
        __fields.update(create_layout(intro=intro))
        # init the state
        __fields.update(create_state())
        # update the data after clicking process
        __fields['process_block'].click(
            fn=compute,
            inputs=[__fields[__k] for __k in ['tokens_block', 'topk_block', 'topp_block', 'input_block']],
            outputs=[__fields[__k] for __k in ['output_state', 'hidden_state']],
            queue=False,
            show_progress='full'
        ).then(
        # update the probability scores when the data changes
            fn=prob_score,
            inputs=[__fields[__k] for __k in ['output_state', 'hidden_state']],
            outputs=__fields['prob_highlight_block'],
            queue=False,
            show_progress='full'
        ).then(
        # update the probability plot when the data changes
            fn=prob_plot,
            inputs=[__fields[__k] for __k in ['output_state', 'hidden_state']],
            outputs=__fields['prob_plot_block'],
            queue=False,
            show_progress='full'
        ).then(
        # update the rank scores when the data changes
            fn=rank_score,
            inputs=[__fields[__k] for __k in ['output_state', 'hidden_state']],
            outputs=__fields['rank_highlight_block'],
            queue=False,
            show_progress='full'
        ).then(
        # update the rank plot when the data changes
            fn=rank_plot,
            inputs=[__fields[__k] for __k in ['output_state', 'hidden_state']],
            outputs=__fields['rank_plot_block'],
            queue=False,
            show_progress='full'
        ).then(
        # update the JSD scores when the data changes
            fn=jsd_score,
            inputs=[__fields[__k] for __k in ['layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['jsd_highlight_block'],
            queue=False,
            show_progress='full'
        ).then(
        # update the JSD plot when the data changes
            fn=jsd_plot,
            inputs=[__fields[__k] for __k in ['layer_block', 'hidden_state']],
            outputs=__fields['jsd_plot_block'],
            queue=False,
            show_progress='full')
        # update the JSD token scores when the focus changes
        __fields['layer_block'].change(
            fn=jsd_score,
            inputs=[__fields[__k] for __k in ['layer_block', 'output_state', 'hidden_state']],
            outputs=__fields['jsd_highlight_block'],
            queue=False,
            show_progress='hidden')
        # update the JSD plot when the focus changes
        __fields['layer_block'].change(
            fn=jsd_plot,
            inputs=[__fields[__k] for __k in ['layer_block', 'hidden_state']],
            outputs=__fields['jsd_plot_block'],
            queue=False,
            show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    # load the model
    __device = 'cuda' if torch.cuda.is_available() else 'cpu'
    __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=MODEL, device=__device)
    __model = psaiops.common.model.get_model(name=MODEL, device=__device)
    __norm = copy.deepcopy(__model.model.norm).cpu()
    __head = copy.deepcopy(__model.lm_head).cpu()
    # adapt the event handlers
    # __highlight = functools.partial(update_token_focus, tokenizer_obj=__tokenizer)
    __compute = functools.partial(update_computation_state, model_obj=__model, tokenizer_obj=__tokenizer, device_str=__device)
    __prob_score = functools.partial(update_prob_scores, tokenizer_obj=__tokenizer, head_obj=__head)
    __prob_plot = functools.partial(update_prob_plot, head_obj=__head)
    __rank_score = functools.partial(update_rank_scores, tokenizer_obj=__tokenizer, head_obj=__head)
    __rank_plot = functools.partial(update_rank_plot, head_obj=__head)
    __jsd_score = functools.partial(update_jsd_scores, tokenizer_obj=__tokenizer, head_obj=__head, norm_obj=__norm)
    __jsd_plot = functools.partial(update_jsd_plot, head_obj=__head, norm_obj=__norm)
    # the event handlers are created outside so that they can be wrapped with `spaces.GPU` if necessary
    __app = create_app(compute=__compute, prob_score=__prob_score, prob_plot=__prob_plot, rank_score=__rank_score, rank_plot=__rank_plot, jsd_score=__jsd_score, jsd_plot=__jsd_plot)
    __app.launch(theme=gradio.themes.Soft(), css=STYLE, share=True, debug=True)
