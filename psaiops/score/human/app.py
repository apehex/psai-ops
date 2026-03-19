import functools
import os

import gradio
import torch.cuda

import psaiops.common.model
import psaiops.common.style
import psaiops.common.tokenizer

import psaiops.score.human.lib as _lib
import psaiops.score.human.ui as _ui
import psaiops.score.human.ux as _ux

# META #########################################################################

_PATH = os.path.dirname(__file__)

# APP ##########################################################################

def create_app(
    switch: callable,
    current: callable,
    partition: callable,
    convert: callable,
    compute: callable,
    title: str=_ui.TITLE,
    intro: str=_ui.INTRO,
    tuto: str=_ui.TUTO,
    docs: str=_ui.DOCS,
    export: str='',
    models: list=_ui.MODELS,
    samples: dict=_ui.SAMPLES,
) -> gradio.Blocks:
    # holds all the UI widgets
    __fields = {}
    with gradio.Blocks(title=title) as __app:
        # create the UI
        __fields.update(_ui.create_layout(title=title, intro=intro, tuto=tuto, docs=docs, models=current().get('choices', models)))
        # init the state
        __fields.update(_ui.create_state(export_str=export))
        # split the string into token sub-strings
        __fields['process_block'].click(
            fn=partition,
            inputs=[__fields[__k] for __k in ['input_block', 'export_state']],
            outputs=__fields['tokens_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # translate the string into token indices
            fn=convert,
            inputs=[__fields[__k] for __k in ['input_block', 'export_state']],
            outputs=__fields['indices_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # then compute the associated logits
            fn=compute,
            inputs=[__fields[__k] for __k in ['indices_state', 'export_state']],
            outputs=__fields['logits_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the unicode scores
            fn=_ux.update_unicode_state,
            inputs=[__fields[__k] for __k in ['tokens_state', 'export_state']],
            outputs=__fields['unicode_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the rank scores
            fn=_ux.update_rank_state,
            inputs=[__fields[__k] for __k in ['indices_state', 'logits_state', 'export_state']],
            outputs=__fields['rank_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the entropy scores
            fn=_ux.update_entropy_state,
            inputs=[__fields[__k] for __k in ['logits_state', 'export_state']],
            outputs=__fields['entropy_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the perplexity scores
            fn=_ux.update_perplexity_state,
            inputs=[__fields[__k] for __k in ['indices_state', 'logits_state', 'export_state']],
            outputs=__fields['perplexity_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # compute the surprisal scores
            fn=_ux.update_surprisal_state,
            inputs=[__fields[__k] for __k in ['indices_state', 'logits_state', 'export_state']],
            outputs=__fields['surprisal_state'],
            queue=True,
            show_progress='hidden'
        ).then(
        # then compute the scores
            fn=_ux.update_token_highlights,
            inputs=[__fields[__k] for __k in ['tokens_state', 'unicode_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block', 'export_state']],
            outputs=__fields['highlight_block'],
            queue=True,
            show_progress='full'
        ).then(
        # and plot the metrics
            fn=_ux.update_metric_plots,
            inputs=[__fields[__k] for __k in ['unicode_state', 'rank_state', 'entropy_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block']],
            outputs=__fields['plot_block'],
            queue=True,
            show_progress='full')
        # update the plots when the metric selection changes
        __fields['selection_block'].change(
            fn=_ux.update_token_highlights,
            inputs=[__fields[__k] for __k in ['tokens_state', 'unicode_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block', 'export_state']],
            outputs=__fields['highlight_block'],
            queue=True,
            show_progress='full'
        ).then(
            fn=_ux.update_metric_plots,
            inputs=[__fields[__k] for __k in ['unicode_state', 'rank_state', 'entropy_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block']],
            outputs=__fields['plot_block'],
            queue=True,
            show_progress='full')
        # update the plots when the window changes
        __fields['window_block'].change(
            fn=_ux.update_token_highlights,
            inputs=[__fields[__k] for __k in ['tokens_state', 'unicode_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block', 'export_state']],
            outputs=__fields['highlight_block'],
            queue=True,
            show_progress='full'
        ).then(
            fn=_ux.update_metric_plots,
            inputs=[__fields[__k] for __k in ['unicode_state', 'rank_state', 'entropy_state', 'surprisal_state', 'perplexity_state', 'selection_block', 'window_block']],
            outputs=__fields['plot_block'],
            queue=True,
            show_progress='full')
        # activate the loading button when selecting another model
        __fields['model_block'].change(
            fn=_ux.enable_button,
            inputs=[],
            outputs=__fields['load_block'],
            queue=True,
            show_progress='full')
        # require an explicit button press to load the selected model
        __fields['load_block'].click(
            fn=switch,
            inputs=__fields['model_block'],
            outputs=__fields['model_block'],
            queue=True,
            show_progress='full')
        # make sure the loaded model is selected (in case the user changed the selection without loading the model)
        __fields['settings_tab'].select(
            fn=current,
            inputs=[],
            outputs=__fields['model_block'],
            queue=True,
            show_progress='hidden'
        ).then(
        # disable the loading button
            fn=_ux.disable_button,
            inputs=[],
            outputs=__fields['load_block'],
            queue=True,
            show_progress='hidden')
        # buttons to fill the input with random samples
        for __dataset in samples.keys():
            for __label in samples[__dataset].keys():
                # fill the input textbox when the user clicks on a given sample
                __fields[f'{__dataset}_{__label}_block'].click(
                    fn=_ux.sample_input_text,
                    inputs=[gradio.State(__dataset), gradio.State(__label)],
                    outputs=__fields['input_block'],
                    queue=True,
                    show_progress='hidden')
        # gradio application
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    # load the model
    __device = 'cuda' if torch.cuda.is_available() else 'cpu'
    __tokenizer = psaiops.common.tokenizer.get_tokenizer(name=_ui.MODELS[0], device=__device)
    __model = psaiops.common.model.get_model(name=_ui.MODELS[0], device=__device)
    # adapt the event handlers
    __current = lambda: gradio.update(value=_ui.MODELS[0], choices=_ui.MODELS) # list of the models with the current model at index 0
    __switch = lambda: __current() # disable model switching here
    __partition = functools.partial(_ux.update_tokens_state, tokenizer_obj=__tokenizer)
    __convert = functools.partial(_ux.update_indices_state, tokenizer_obj=__tokenizer)
    __compute = functools.partial(_ux.update_logits_state, model_obj=__model)
    # the event handlers are created outside so that they can be wrapped with `spaces.GPU` if necessary
    __app = create_app(switch=__switch, current=__current, partition=__partition, convert=__convert, compute=__compute)
    __app.launch(theme=gradio.themes.Soft(), css=psaiops.common.style.ALL, share=True, debug=True)
