"""Convert measured model latency into inference throughput."""


def throughput_samples_per_second(
    mean_latency_ms: float,
    batch_size: int = 1,
) -> float:
    """Convert mean batch latency to samples processed per second."""
    if mean_latency_ms <= 0:
        raise ValueError(
            "mean_latency_ms must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    return (
        batch_size
        *
        1000.0
        /
        mean_latency_ms
    )
