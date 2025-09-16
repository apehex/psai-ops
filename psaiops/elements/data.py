import requests

# CONSTANTS ####################################################################

HF_URL = 'https://huggingface.co/api/quicksearch?q={target}&type={label}&limit={limit}'

# HUGGING FACE #################################################################

def query_huggingface(target: str, label: str='model', limit: int=16, endpoint: str=HF_URL) -> list:
    __results = []
    # query HF
    __response = requests.get(endpoint.format(target=target, label=label, limit=limit))
    # no error
    if __response.status_code == 200:
        __results = [__d['id'] for __d in __response.json().get(f'{label}s', [])]
    # list of strings
    return __results
