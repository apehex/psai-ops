import gradio

import psaiops.score.attention.lib

# META #########################################################################

TITLE = '''Attention Scoring'''
INTRO = '''Score each token according to the weights of the attention layers.'''
STYLE = ''''''

MODEL = 'openai/gpt-oss-20b'

# INTRO ########################################################################

def create_intro_block(intro: str) -> dict:
    __intro = gradio.Markdown(intro)
    return {'intro_block': __intro}

# MODEL ########################################################################

def update_slider_range(model: str) -> dict:
    return gradio.update(maximum=35, value=18) if '120b' in model else gradio.update(maximum=23, value=12)

def create_model_block() -> dict:
    __model_dd = gradio.Dropdown(label='Model', value='openai/gpt-oss-20b', choices=['openai/gpt-oss-20b', 'openai/gpt-oss-120b'], allow_custom_value=False, multiselect=False, interactive=True)
    __layer_sl = gradio.Slider(label='Layer Depth', value=12, minimum=-1, maximum=23, step=1, interactive=True) # info='-1 to average on all layers'
    __head_sl = gradio.Slider(label='Attention Head', value=-1, minimum=-1, maximum=63, step=1, interactive=True) # info='-1 to average on all heads'
    __model_dd.change(fn=update_slider_range, inputs=__model_dd, outputs=__layer_sl, queue=False, show_progress='hidden')
    return {
        'model_block': __model_dd,
        'layer_block': __layer_sl,
        'head_block': __head_sl}

# SAMPLING #####################################################################

def create_sampling_block() -> dict:
    __tokens = gradio.Slider(label='Tokens', value=32, minimum=0, maximum=128, step=1, interactive=True)
    __topk = gradio.Slider(label='Top K', value=4, minimum=1, maximum=8, step=1, interactive=True)
    __topp = gradio.Slider(label='Top P', value=0.8, minimum=0.0, maximum=1.0, step=0.1, interactive=True)
    return {
        'tokens_block': __tokens,
        'topk_block': __topk,
        'topp_block': __topp}

# DISPLAY ######################################################################

def create_display_block() -> dict:
    __display = gradio.Radio(label='Display', value='Tokens', choices=['Tokens', 'Indexes'], interactive=True)
    return {'display_block': __display}

# INPUTS #######################################################################

def create_inputs_block() -> dict:
    __input = gradio.Textbox(label='Prompt', value='', placeholder='A string of tokens to score.', lines=4, show_copy_button=True, interactive=True)
    return {'input_block': __input}

# OUTPUTS ######################################################################

def create_outputs_block() -> dict:
    __output = gradio.HighlightedText(label='Scores', value='', interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=True)
    return {'output_block': __output}

# ACTIONS ######################################################################

def create_actions_block() -> dict:
    __process = gradio.Button('Process', variant='primary', size='lg', interactive=True)
    __position = gradio.Slider(label='Position', value=-1, minimum=-1, maximum=128, step=1, interactive=True) # info='-1 to average on all tokens'
    return {
        'process_block': __process,
        'position_block': __position}

# STATE ########################################################################

def create_state(model: str=MODEL) -> dict:
    __device = gradio.State('cuda' if torch.cuda.is_available() else 'cpu')
    __model = gradio.State(psaiops.score.attention.lib.get_model(name=model, device=str(__device)))
    __tokenizer = gradio.State(psaiops.score.attention.lib.get_tokenizer(name=model, device=str(__device)))
    __data = gradio.State(None)
    return {
        'device_state': __device,
        'model_state': __model,
        'tokenizer_state': __tokenizer,
        'data_state': __data,}

# LAYOUT #######################################################################

def create_layout(intro: str=INTRO) -> dict:
    __fields = {}
    fields.update(create_intro_block(intro=intro))
    with gradio.Tabs():
        with gradio.Tab('Score Tokens'):
            with gradio.Row():
                with gradio.Column(scale=1):
                    fields.update(create_inputs_block())
                with gradio.Column(scale=1):
                    fields.update(create_outputs_block())
            with gradio.Row():
                fields.update(create_actions_block())
        with gradio.Tab('Settings'):
            with gradio.Column(scale=1):
                with gradio.Row():
                    fields.update(create_model_block())
                with gradio.Row():
                    fields.update(create_sampling_block())
                with gradio.Row():
                    fields.update(create_display_block())
    return __fields

# EVENTS #######################################################################

# APP ##########################################################################

def create_app(title: str=TITLE, intro: str=INTRO, style: str=STYLE) -> gradio.Blocks:
    __fields = {}
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        fields.update(create_layout(intro=intro))
        fields.update(create_state(model=str(fields['model_block'])))
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_app()
    __app.launch(share=True, debug=True)
