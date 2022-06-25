import pytest

from common.web_scraping.verification import assert_request_is_working


@pytest.mark.parametrize("is_get", (True, False))
def test_verification_on_google(is_get: bool) -> None:
    assert_request_is_working("https://www.google.com/", is_get=is_get)
