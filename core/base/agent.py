"""
Base Agent

Defines the common behavior for all AI agents.
"""

from abc import ABC, abstractmethod
from typing import Any

from core.logging.logger import logger


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    """

    def __init__(self, name: str):
        self.name = name

    def start(self) -> None:
        """
        Start the agent.
        """
        logger.info("%s started.", self.name)

    def stop(self) -> None:
        """
        Stop the agent.
        """
        logger.info("%s stopped.", self.name)

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the agent task.
        """
        pass