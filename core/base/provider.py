"""
Base Provider

Defines the common interface for all AI providers.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """
    Abstract base class for all providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the provider name.
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the provider.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether the provider is available.
        """
        pass

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generate output from the provider.
        """
        pass