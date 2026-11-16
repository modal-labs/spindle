import os
from urllib.parse import urlsplit


def modal_proxy_headers(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not (parsed.hostname or "").endswith(".modal.run")
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Proxy authentication requires an HTTPS Modal endpoint")
    return {
        "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
        "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
    }
