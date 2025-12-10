import requests

# CONSTANTS ####################################################################

HF_URL = 'https://huggingface.co/api/quicksearch?q={target}&type={label}&limit={limit}'

# HUGGING FACE #################################################################

def query_huggingface(target: str, label: str='model', limit: int=16, endpoint: str=HF_URL) -> list:
    __results = []
    # make sure the label has no trailing "s"
    __label = label.rstrip('s').strip(' ')
    # the HTTP request or the parsing may fail
    try:
        # query HF
        __response = requests.get(endpoint.format(target=target, label=__label, limit=limit))
        # filter by type ('models' / 'datasets' / 'spaces')
        __results = [__d.get('id', '') for __d in __response.json().get(f'{__label}s', [])]
    # ignore all the errors
    except:
        __results = []
    # list of strings
    return __results

# EXAMPLES #####################################################################

def update_dropdown(target: str, label: str) -> dict:
    # query huggingface
    __data = psaiops.elements.data.query_huggingface(target=target, label=label, limit=16)
    # list choices in the dropdown
    return gradio.update(choices=__data, visible=True)
