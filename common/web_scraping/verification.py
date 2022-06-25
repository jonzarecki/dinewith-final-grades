from typing import Dict, Optional

import requests


def assert_request_is_working(
    url: str,
    is_get: bool,
    headers: Optional[Dict[str, str]] = None,
    cookies: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, str]] = None,
) -> None:
    """Assert the website works with the data given, and return response 200."""
    if is_get:
        assert data is None, "can't pass data in get request"
        r = requests.get(url, headers=headers, cookies=cookies)
    else:
        r = requests.get(url, headers=headers, cookies=cookies, data=data)
    assert r.status_code == 200
