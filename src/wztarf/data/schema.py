"""Canonical dataset-level temporal constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceSpec:
    """Fixed temporal specification used by the current dataset."""

    fps: int = 5
    history_steps: int = 10
    future_steps: int = 25
    num_modes: int = 6

    @property
    def dt(self) -> float:
        """Seconds between adjacent samples."""
        return 1.0 / self.fps


DEFAULT_SEQUENCE_SPEC = SequenceSpec()
