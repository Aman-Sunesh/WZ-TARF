"""Expose the forecasting model."""

from .wztarf import WZTARF, WZTARFConfig

__all__ = [
    "WZTARF",
    "WZTARFConfig",
]
