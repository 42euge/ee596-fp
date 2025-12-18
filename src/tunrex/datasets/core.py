"""Core TunRex class for dataset loading and preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tunrex.datasets.config import TunRexConfig
from tunrex.datasets.loaders import (
    load_from_huggingface,
    load_from_kaggle,
    load_from_tfds,
    load_openrubrics,
)
from tunrex.datasets.transforms import create_grpo_transform, create_raw_transform

if TYPE_CHECKING:
    import grain


def _get_grain():
    """Lazy import grain to allow package import without grain installed."""
    try:
        import grain
        return grain
    except ImportError as e:
        raise ImportError(
            "grain is required for dataset preparation. "
            "Install it with: pip install grain"
        ) from e


class TunRex:
    """Main class for loading and preparing datasets for GRPO training.

    Example:
        >>> config = TunRexConfig(source="huggingface", dataset_name="gsm8k")
        >>> trex = TunRex(config)
        >>> train_ds, val_ds, test_ds = trex.prepare_datasets()
        >>> print(f"Train batches: {len(train_ds)}")
    """

    def __init__(self, config: TunRexConfig):
        """Initialize TunRex with configuration.

        Args:
            config: TunRexConfig instance with dataset settings
        """
        self.config = config
        self._train_data: list[dict[str, Any]] | None = None
        self._test_data: list[dict[str, Any]] | None = None
        self._datasets_prepared = False
        self._info: dict[str, Any] = {}

    def _load_raw_data(self, split: str) -> list[dict[str, Any]] | Any:
        """Load raw data from configured source.

        Args:
            split: Dataset split ("train" or "test")

        Returns:
            Raw data as list of dicts or iterable
        """
        config = self.config
        data_dir = config.train_data_dir if split == "train" else config.test_data_dir

        # Handle OpenRubrics specially
        if config.dataset_name == "OpenRubrics/OpenRubrics":
            return load_openrubrics(
                split=config.hf_subset or split,
                max_examples=config.max_examples,
            )

        if config.source == "kaggle":
            return load_from_kaggle(data_dir, split)
        elif config.source == "huggingface":
            return load_from_huggingface(
                config.dataset_name,
                split=split,
                subset=config.hf_subset,
            )
        elif config.source == "tfds":
            return load_from_tfds(data_dir, split)
        else:
            raise ValueError(f"Unknown source: {config.source}")

    def _create_dataset(self, raw_data: Any) -> "grain.MapDataset":
        """Create a grain.MapDataset from raw data with transforms applied.

        Args:
            raw_data: Raw data to wrap

        Returns:
            grain.MapDataset with shuffling and transforms applied
        """
        grain = _get_grain()
        config = self.config

        # Create appropriate transform
        if config.apply_template:
            transform = create_grpo_transform(
                template=config.template,
                system_prompt=config.system_prompt,
                answer_extractor=config.answer_extractor,
                include_rubric=config.dataset_name == "OpenRubrics/OpenRubrics",
            )
        else:
            transform = create_raw_transform()

        # Build the dataset pipeline
        dataset = (
            grain.MapDataset.source(raw_data)
            .shuffle(seed=config.shuffle_seed)
            .map(transform)
        )

        return dataset

    def load(self) -> "TunRex":
        """Load raw data from the configured source.

        Returns:
            Self for method chaining
        """
        print(f"Loading {self.config.dataset_name} from {self.config.source}...")

        self._train_data = self._load_raw_data("train")
        self._test_data = self._load_raw_data("test")

        print(f"Loaded {len(self._train_data)} train examples")
        print(f"Loaded {len(self._test_data)} test examples")

        return self

    def prepare_datasets(
        self,
    ) -> tuple["grain.MapDataset", "grain.MapDataset | None", "grain.MapDataset"]:
        """Prepare train, validation, and test datasets.

        Loads data if not already loaded, applies transforms, batches,
        and splits according to configuration.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            val_dataset may be None if train_fraction == 1.0
        """
        config = self.config

        # Load data if needed
        if self._train_data is None:
            self.load()

        # Create base datasets with transforms
        train_base = self._create_dataset(self._train_data)
        test_base = self._create_dataset(self._test_data)

        # Apply max_examples limit
        if config.max_examples:
            train_base = train_base[: config.max_examples]

        # Batch the datasets
        train_batched = train_base.batch(config.batch_size)
        test_batched = test_base.batch(config.batch_size)

        # Apply batch limits
        if config.max_train_batches:
            train_batched = train_batched[: config.max_train_batches]
        if config.max_test_batches:
            test_batched = test_batched[: config.max_test_batches]

        total_train_batches = len(train_batched)

        # Split into train/val
        if config.train_fraction < 1.0:
            train_split = int(total_train_batches * config.train_fraction)
            val_split = int(total_train_batches * (config.train_fraction + config.val_fraction))

            train_dataset = train_batched[:train_split]
            val_dataset = train_batched[train_split:val_split]

            # Repeat for epochs
            if config.num_epochs > 1:
                train_dataset = train_dataset.repeat(config.num_epochs)
                val_dataset = val_dataset.repeat(config.num_epochs)
        else:
            # No validation split
            train_dataset = train_batched
            val_dataset = None

            if config.num_epochs > 1:
                train_dataset = train_dataset.repeat(config.num_epochs)

        test_dataset = test_batched

        # Store info
        self._info = {
            "train_batches": len(train_dataset),
            "val_batches": len(val_dataset) if val_dataset else 0,
            "test_batches": len(test_dataset),
            "batch_size": config.batch_size,
            "train_fraction": config.train_fraction,
            "val_fraction": config.val_fraction,
            "num_epochs": config.num_epochs,
        }

        self._datasets_prepared = True
        self._print_summary()

        return train_dataset, val_dataset, test_dataset

    def _print_summary(self) -> None:
        """Print dataset summary."""
        info = self._info
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Source:            {self.config.source}")
        print(f"Dataset:           {self.config.dataset_name}")
        print(f"Batch size:        {info['batch_size']}")
        print(f"Train batches:     {info['train_batches']:,}")
        print(f"Validation batches:{info['val_batches']:,}")
        print(f"Test batches:      {info['test_batches']:,}")
        print("=" * 60 + "\n")

    @property
    def info(self) -> dict[str, Any]:
        """Get dataset info after preparation.

        Returns:
            Dictionary with dataset statistics
        """
        if not self._datasets_prepared:
            raise RuntimeError("Call prepare_datasets() first")
        return self._info.copy()

    def get_sample(self, split: str = "train", n: int = 1) -> list[dict]:
        """Get sample examples from the dataset.

        Args:
            split: Which split to sample from ("train" or "test")
            n: Number of samples

        Returns:
            List of sample dictionaries
        """
        if self._train_data is None:
            self.load()

        data = self._train_data if split == "train" else self._test_data

        # Create a mini dataset and get samples
        config = self.config
        if config.apply_template:
            transform = create_grpo_transform(
                template=config.template,
                system_prompt=config.system_prompt,
                answer_extractor=config.answer_extractor,
            )
        else:
            transform = create_raw_transform()

        return [transform(data[i]) for i in range(min(n, len(data)))]

    def preview(self, dataset=None, n: int = 1) -> None:
        """Preview samples from a dataset.

        Simple visualization of dataset contents using pprint.

        Args:
            dataset: A grain.MapDataset to preview. If None, previews raw train data.
            n: Number of samples to show (default: 1)

        Example:
            >>> trex = TunRex(config)
            >>> train_ds, val_ds, test_ds = trex.prepare_datasets()
            >>> trex.preview(train_ds)
        """
        from pprint import pprint

        if dataset is None:
            # Preview raw samples
            samples = self.get_sample("train", n)
            print(f"Previewing {len(samples)} raw sample(s):")
            for sample in samples:
                pprint(sample)
        else:
            # Preview from dataset
            print(f"Previewing {n} batch(es) from dataset:")
            for ele in dataset[:n]:
                pprint(ele)
