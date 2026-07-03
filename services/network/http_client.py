"""
HTTP Client

Reusable HTTP client for API communication.
"""

from typing import Any, Optional

import requests
from requests import Response


class HttpClient:
    """
    Reusable HTTP client.
    """

    def get(
        self,
        url: str,
        timeout: int = 30,
        **kwargs: Any,
    ) -> Response:
        """
        Execute HTTP GET request.
        """
        return requests.get(
            url=url,
            timeout=timeout,
            **kwargs,
        )

    def post(
        self,
        url: str,
        json: Optional[dict] = None,
        timeout: int = 180,  # changed from 60
        **kwargs: Any,
    ) -> Response:
        """
        Execute HTTP POST request.
        """
        return requests.post(
            url=url,
            json=json,
            timeout=timeout,
            **kwargs,
        )