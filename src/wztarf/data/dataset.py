"""Load final serialized WorkZone samples from one or more dataset roots."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .schema import DEFAULT_SEQUENCE_SPEC, SequenceSpec, validate_sample


def discover_pt_files(
    roots: Sequence[str | Path],
    split: str | None = None,
) -> list[Path]:
    """Discover serialized `.pt` samples under one or more roots.

    When `split` is provided, the function first looks for directories such
    as `<root>/train`, `<root>/val`, or `<root>/test`.

    If that exact directory does not exist, it recursively searches only paths
    containing the requested split as one of their directory components.

    Args:
        roots:
            Dataset roots, for example the final Lap 1 and Lap 2 roots.

        split:
            Optional split name such as `"train"`, `"val"`, or `"test"`.

    Returns:
        Sorted list of unique `.pt` paths.

    Raises:
        FileNotFoundError:
            If a root does not exist or no matching `.pt` files are found.
    """
    discovered: list[Path] = []

    for root_value in roots:
        root = Path(root_value).expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        if not root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {root}")

        if split is None:
            discovered.extend(root.rglob("*.pt"))
            continue

        split_name = split.lower()
        direct_split_dir = root / split

        if direct_split_dir.is_dir():
            discovered.extend(direct_split_dir.rglob("*.pt"))
            continue

        # Also support roots whose directory layout has the split nested
        # somewhere below the root.
        for path in root.rglob("*.pt"):
            lower_parts = {part.lower() for part in path.parts}

            if split_name in lower_parts:
                discovered.append(path)

    # De-duplicate while preserving deterministic ordering.
    unique = sorted(set(discovered))

    if not unique:
        split_text = f" for split '{split}'" if split is not None else ""

        raise FileNotFoundError(
            f"No .pt samples found{split_text} under roots: "
            f"{[str(Path(root)) for root in roots]}"
        )

    return unique

def validate_edge_type_capacity(
    dataset: Dataset,
    *,
    num_edge_types: int,
    max_samples: int | None = None,
) -> None:
    """Verify sampled serialized edge IDs fit the model embedding table.

    A full dataset scan is intentionally optional because opening every
    individual `.pt` file at training startup can take many minutes.
    """
    maximum = -1

    dataset_size = len(dataset)

    if dataset_size == 0:
        return

    if max_samples is None or max_samples >= dataset_size:
        indices = range(dataset_size)
    else:
        if max_samples <= 0:
            return

        # Deterministic approximately-uniform coverage over the dataset.
        step = dataset_size / float(max_samples)
        indices = [
            min(
                int(sample_index * step),
                dataset_size - 1,
            )
            for sample_index in range(max_samples)
        ]

    for index in indices:
        sample = dataset[
            index
        ]

        mask = sample[
            "lane_edge_mask"
        ].bool()

        if not bool(
            mask.any()
        ):
            continue

        local_max = int(
            sample[
                "lane_edge_type"
            ][
                mask
            ].max().item()
        )

        maximum = max(
            maximum,
            local_max,
        )

    if maximum >= num_edge_types:
        raise ValueError(
            f"Dataset uses lane_edge_type={maximum}, but "
            f"model.num_edge_types={num_edge_types}. "
            f"Set num_edge_types to at least {maximum + 1}."
        )

class WorkZoneDataset(Dataset):
    """Dataset for the final WorkZone samples.

    The loader itself performs no hidden normalization. Samples are loaded from
    disk, validated against the canonical schema, and then optionally passed
    through an explicit transform.

    This keeps serialized dataset content separate from model-specific feature
    construction.
    """

    def __init__(
        self,
        *,
        files: Sequence[str | Path] | None = None,
        roots: Sequence[str | Path] | None = None,
        split: str | None = None,
        spec: SequenceSpec = DEFAULT_SEQUENCE_SPEC,
        validate: bool = True,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        include_source_path: bool = False,
    ) -> None:
        """Create the dataset.

        Exactly one of `files` or `roots` must be provided.

        Args:
            files:
                Explicit sample paths.

            roots:
                One or more dataset roots.

            split:
                Split used when discovering from roots.

            spec:
                Temporal specification used for schema validation.

            validate:
                Validate every sample when loaded.

            transform:
                Optional explicit sample transformation.

            include_source_path:
                Add the serialized file path to each returned sample.
        """
        if (files is None) == (roots is None):
            raise ValueError(
                "Provide exactly one of `files` or `roots`."
            )

        if files is not None:
            sample_files = [
                Path(path).expanduser().resolve()
                for path in files
            ]

            missing = [
                path
                for path in sample_files
                if not path.is_file()
            ]

            if missing:
                raise FileNotFoundError(
                    f"Missing sample files: {missing[:5]}"
                )

            self.files = sorted(sample_files)

        else:
            assert roots is not None

            self.files = discover_pt_files(
                roots=roots,
                split=split,
            )

        if not self.files:
            raise ValueError("Dataset contains zero samples.")

        self.spec = spec
        self.validate = validate
        self.transform = transform
        self.include_source_path = include_source_path

    def __len__(self) -> int:
        """Return the number of serialized samples."""
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and return one WorkZone sample."""
        path = self.files[index]

        sample = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(sample, dict):
            raise TypeError(
                f"Expected dictionary sample in {path}, "
                f"got {type(sample).__name__}."
            )

        if self.validate:
            validate_sample(
                sample,
                spec=self.spec,
                source=str(path),
            )

        # Shallow copy prevents adding runtime fields to the object returned
        # directly by torch.load.
        sample = dict(sample)

        if self.include_source_path:
            sample["source_path"] = str(path)

        if self.transform is not None:
            sample = self.transform(sample)

            if not isinstance(sample, dict):
                raise TypeError(
                    "Dataset transform must return a dictionary."
                )

        return sample

    def sample_path(self, index: int) -> Path:
        """Return the serialized path corresponding to a dataset index."""
        return self.files[index]
