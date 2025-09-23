import gradio

import psaiops.elements.data

# AUTO-COMPLETE ################################################################

def update_dropdown(label: str, data: gradio.KeyUpData):
    # model_dropdown.key_up(fn=update_dropdown, inputs=[model_dropdown, gradio.State("model")], outputs=model_dropdown, queue=False, show_progress="hidden")
    datasets = psaiops.elements.data.query_huggingface(target=data.input_value, label=label, limit=16)
    return gradio.update(choices=datasets, visible=True)

# with gradio.Blocks() as demo:
#     model_dropdown = gradio.Dropdown(label="Models Auto-Complete", choices=[""], allow_custom_value=True)
#     dataset_dropdown = gradio.Dropdown(label="Datasets Auto-Complete", choices=[""], allow_custom_value=True)
#     spaces_dropdown = gradio.Dropdown(label="Spaces Auto-Complete", choices=[""], allow_custom_value=True)
#     model_dropdown.key_up(fn=update_dropdown, inputs=[gradio.State("model")], outputs=model_dropdown, queue=False, show_progress="hidden")
#     dataset_dropdown.key_up(fn=update_dropdown, inputs=[gradio.State("dataset")], outputs=dataset_dropdown, queue=False, show_progress="hidden")
#     spaces_dropdown.key_up(fn=update_dropdown, inputs=[gradio.State("space")], outputs=spaces_dropdown, queue=False, show_progress="hidden")
# demo.launch(share=True, debug=True)
