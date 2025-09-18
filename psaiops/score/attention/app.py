import gradio

# META #########################################################################

TITLE = '''Attention Scoring'''
INTRO = '''Score each token according to the weights of the attention layers.'''
STYLE = ''''''

# INTRO ########################################################################

def create_intro_block(intro: str) -> None:
    gradio.Markdown(intro)

# MODEL ########################################################################

def create_model_block() -> None:
    gradio.Dropdown(label='Model', value='openai/gpt-oss-20b', choices=['openai/gpt-oss-20b', 'openai/gpt-oss-120b'], allow_custom_value=False, multiselect=False, interactive=True)
    gradio.Slider(label='Layer Depth', value=8, minimum=0, maximum=24, step=1, interactive=True)
    # gradio.Dropdown(label='Layer', value='', choices=[''], allow_custom_value=False, multiselect=False, interactive=True)
    # gradio.Dropdown(label='Tokenizer', value='o200k_harmony', choices=['o200k_harmony'], allow_custom_value=False, multiselect=False, interactive=True)

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
