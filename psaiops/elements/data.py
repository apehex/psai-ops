import requests

# CONSTANTS ####################################################################

HF_URL = 'https://huggingface.co/api/quicksearch?q={target}&type={label}&limit={limit}'

# HUGGING FACE #################################################################

def query_huggingface(target: str, label: str='model', limit: int=16, endpoint: str=HF_URL) -> list:
    __results = []
    # handle all the errors
    try:
        # query HF
        __response = requests.get(endpoint.format(target=target, label=label, limit=limit))
        # filter by type ('models' / 'datasets' / 'spaces')
        __results = [__d.get('id', '') for __d in __response.json().get(f'{label}s', [])]
    except:
        __results = []
    # list of strings
    return __results
