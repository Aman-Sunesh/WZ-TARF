"""Trainable parameter-count metric."""


def parameter_count(model) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
