"""Expose the forecasting model and its configuration."""

from .config import WZTARFConfig
from .wztarf import WZTARF

__all__ = ["WZTARF", "WZTARFConfig"]
