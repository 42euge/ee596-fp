"""
Dataset loaders for different file formats
"""

import json
import jsonlines
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


def convert_numpy_to_python(obj: Any) -> Any:
    """
    Recursively convert NumPy types to native Python types

    Args:
        obj: Object that may contain NumPy types

    Returns:
        Object with NumPy types converted to Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    return obj


class DatasetLoader(ABC):
    """Base class for dataset loaders"""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    @abstractmethod
    def load(self, start_row: int = 0, num_rows: int = 100) -> Tuple[List[Dict], int]:
        """
        Load dataset with pagination

        Args:
            start_row: Starting row index
            num_rows: Number of rows to load

        Returns:
            Tuple of (data_rows, total_rows)
        """
        pass

    @abstractmethod
    def get_columns(self) -> List[str]:
        """Get column names"""
        pass

    @abstractmethod
    def get_total_rows(self) -> int:
        """Get total number of rows in dataset"""
        pass


class ParquetLoader(DatasetLoader):
    """Loader for Parquet files"""

    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self._df = None
        self._total_rows = None

    def _ensure_loaded(self):
        """Lazy load the parquet file"""
        if self._df is None:
            self._df = pd.read_parquet(self.file_path)
            self._total_rows = len(self._df)

    def load(self, start_row: int = 0, num_rows: int = 100) -> Tuple[List[Dict], int]:
        self._ensure_loaded()
        end_row = min(start_row + num_rows, self._total_rows)
        subset = self._df.iloc[start_row:end_row]
        records = subset.to_dict('records')
        # Convert NumPy arrays to Python lists
        records = [convert_numpy_to_python(record) for record in records]
        return records, self._total_rows

    def get_columns(self) -> List[str]:
        self._ensure_loaded()
        return self._df.columns.tolist()

    def get_total_rows(self) -> int:
        self._ensure_loaded()
        return self._total_rows


class JSONLLoader(DatasetLoader):
    """Loader for JSONL (newline-delimited JSON) files"""

    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self._total_rows = None
        self._columns = None

    def _count_rows(self) -> int:
        """Count total rows in JSONL file"""
        if self._total_rows is None:
            with open(self.file_path, 'r') as f:
                self._total_rows = sum(1 for _ in f)
        return self._total_rows

    def load(self, start_row: int = 0, num_rows: int = 100) -> Tuple[List[Dict], int]:
        total = self._count_rows()
        rows = []

        with jsonlines.open(self.file_path) as reader:
            for i, obj in enumerate(reader):
                if i < start_row:
                    continue
                if i >= start_row + num_rows:
                    break
                rows.append(obj)

                # Extract columns from first row
                if self._columns is None and i == 0:
                    self._columns = list(obj.keys())

        return rows, total

    def get_columns(self) -> List[str]:
        if self._columns is None:
            # Load first row to get columns
            with jsonlines.open(self.file_path) as reader:
                first = next(reader, None)
                if first:
                    self._columns = list(first.keys())
                else:
                    self._columns = []
        return self._columns

    def get_total_rows(self) -> int:
        return self._count_rows()


class JSONLoader(DatasetLoader):
    """Loader for JSON files"""

    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self._data = None
        self._is_array = False

    def _ensure_loaded(self):
        """Lazy load the JSON file"""
        if self._data is None:
            with open(self.file_path, 'r') as f:
                loaded = json.load(f)

            # Check if it's an array or object
            if isinstance(loaded, list):
                self._data = loaded
                self._is_array = True
            else:
                # Single object - wrap in list
                self._data = [loaded]
                self._is_array = False

    def load(self, start_row: int = 0, num_rows: int = 100) -> Tuple[List[Dict], int]:
        self._ensure_loaded()
        total = len(self._data)
        end_row = min(start_row + num_rows, total)
        subset = self._data[start_row:end_row]
        return subset, total

    def get_columns(self) -> List[str]:
        self._ensure_loaded()
        if self._data:
            return list(self._data[0].keys())
        return []

    def get_total_rows(self) -> int:
        self._ensure_loaded()
        return len(self._data)


class CSVLoader(DatasetLoader):
    """Loader for CSV files"""

    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self._df = None
        self._total_rows = None

    def _ensure_loaded(self):
        """Lazy load the CSV file"""
        if self._df is None:
            self._df = pd.read_csv(self.file_path)
            self._total_rows = len(self._df)

    def load(self, start_row: int = 0, num_rows: int = 100) -> Tuple[List[Dict], int]:
        self._ensure_loaded()
        end_row = min(start_row + num_rows, self._total_rows)
        subset = self._df.iloc[start_row:end_row]
        records = subset.to_dict('records')
        return records, self._total_rows

    def get_columns(self) -> List[str]:
        self._ensure_loaded()
        return self._df.columns.tolist()

    def get_total_rows(self) -> int:
        self._ensure_loaded()
        return self._total_rows


def detect_format(file_path: Path) -> Optional[str]:
    """
    Detect file format from extension

    Args:
        file_path: Path to file

    Returns:
        Format string: 'parquet', 'jsonl', 'json', 'csv', or None
    """
    suffix = file_path.suffix.lower()

    format_map = {
        '.parquet': 'parquet',
        '.jsonl': 'jsonl',
        '.json': 'json',
        '.csv': 'csv',
    }

    return format_map.get(suffix)


def load_dataset(file_path: Path, start_row: int = 0, num_rows: int = 100) -> Tuple[List[Dict], int, List[str]]:
    """
    Load dataset automatically detecting format

    Args:
        file_path: Path to dataset file
        start_row: Starting row index
        num_rows: Number of rows to load

    Returns:
        Tuple of (data_rows, total_rows, columns)

    Raises:
        ValueError: If format is not supported
    """
    file_format = detect_format(file_path)

    if file_format is None:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    loader_map = {
        'parquet': ParquetLoader,
        'jsonl': JSONLLoader,
        'json': JSONLoader,
        'csv': CSVLoader,
    }

    loader_class = loader_map[file_format]
    loader = loader_class(file_path)

    rows, total = loader.load(start_row, num_rows)
    columns = loader.get_columns()

    return rows, total, columns
