"""Horizon- and behavior/route-aware future contrastive pretraining.

The final implementation must exclude temporally overlapping windows from the
negative set so nearly identical futures are not incorrectly pushed apart.
"""

def future_contrastive_loss(*args, **kwargs):
    """Compute 1 s, 3 s, and 5 s future-aware contrastive alignment."""
    raise NotImplementedError
