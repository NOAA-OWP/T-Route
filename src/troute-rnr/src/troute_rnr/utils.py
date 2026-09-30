import httpx


def get(url: str, headers=None, params=None):
    """A function to GET data from a json endpoint"""
    r = httpx.get(url, headers=headers, params=params, timeout=None)
    return r
