"""Throughput calculation from measured mean latency."""


def throughput_samples_per_second(mean_latency_ms: float, batch_size: int = 1) -> float:
    """Convert batch latency to samples per second."""
    if mean_latency_ms <= 0:
        raise ValueError("mean_latency_ms must be positive")
    return batch_size * 1000.0 / mean_latency_ms
