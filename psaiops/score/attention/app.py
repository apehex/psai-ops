import gradio

# META #########################################################################

TITLE = '''Attention Scoring'''
INTRO = '''Score each token according to the weights of the attention layers.'''
STYLE = ''''''

# INTRO ########################################################################

def create_intro_block(intro: str) -> None:
    gradio.Markdown(intro)

# MODEL ########################################################################

def update_slider_range(model: str) -> dict:
    return gradio.update(maximum=35, value=18) if '120b' in model else gradio.update(maximum=23, value=12)

def create_model_block() -> None:
    __model_dd = gradio.Dropdown(label='Model', value='openai/gpt-oss-20b', choices=['openai/gpt-oss-20b', 'openai/gpt-oss-120b'], allow_custom_value=False, multiselect=False, interactive=True)
    __layer_sl = gradio.Slider(label='Layer Depth', value=12, minimum=-1, maximum=23, step=1, interactive=True) # info='-1 to average on all layers'
    __head_sl = gradio.Slider(label='Attention Head', value=-1, minimum=-1, maximum=63, step=1, interactive=True) # info='-1 to average on all heads'
    __model_dd.change(fn=update_slider_range, inputs=__model_dd, outputs=__layer_sl, queue=False, show_progress='hidden')

# SAMPLING #####################################################################

def create_sampling_block() -> None:
    gradio.Slider(label='Tokens', value=32, minimum=0, maximum=128, step=1, interactive=True)
    gradio.Slider(label='Top K', value=4, minimum=1, maximum=8, step=1, interactive=True)
    gradio.Slider(label='Top P', value=0.8, minimum=0.0, maximum=1.0, step=0.1, interactive=True)

# MODE #########################################################################

# def create_metric_block() -> None:
#     gradio.Radio(label='Mode', value='Attention', choices=['Attention', 'Shapley', 'Similarity'], interactive=True)

# DISPLAY ######################################################################

def create_display_block() -> None:
    gradio.Radio(label='Display', value='Tokens', choices=['Tokens', 'Indexes'], interactive=True)

# INPUTS #######################################################################

def create_inputs_block() -> None:
    gradio.Textbox(label='Prompt', value='', placeholder='A string of tokens to score.', lines=4, interactive=True)

# OUTPUTS ######################################################################

def create_outputs_block() -> None:
    gradio.HighlightedText(label='Scores', value='', interactive=False, show_legend=False, show_inline_category=False, combine_adjacent=True)

# ACTIONS ######################################################################

def create_actions_block() -> None:
    gradio.Button('Process', variant='primary', size='lg', interactive=True)
    gradio.Slider(label='Position', value=-1, minimum=-1, maximum=128, step=1, interactive=True) # info='-1 to average on all tokens'

# LAYOUT #######################################################################

def create_layout(title: str=TITLE, intro: str=INTRO, style: str=STYLE) -> gradio.Blocks:
    with gradio.Blocks(theme=gradio.themes.Soft(), title=title, css=style) as __app:
        create_intro_block(intro=intro)
        with gradio.Tabs():
            with gradio.Tab('Score Tokens'):
                with gradio.Row():
                    with gradio.Column(scale=1):
                        create_inputs_block()
                    with gradio.Column():
                        create_outputs_block()
                with gradio.Row():
                    create_actions_block()
            with gradio.Tab('Settings'):
                with gradio.Column(scale=1):
                    with gradio.Row():
                        create_model_block()
                    with gradio.Row():
                        create_sampling_block()
                    with gradio.Row():
                        # create_metric_block()
                        create_display_block()
        return __app

# MAIN #########################################################################

if __name__ == '__main__':
    __app = create_layout()
    __app.launch(share=True, debug=True)
