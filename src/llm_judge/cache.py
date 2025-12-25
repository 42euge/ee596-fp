"""Rubric caching utilities.

Provides file-based caching for generated rubrics to avoid
regenerating rubrics for the same questions.
"""

import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .rubric_generator import Rubric


class RubricCache:
    """File-based cache for generated rubrics.

    Stores rubrics as JSON files indexed by question hash.
    """

    def __init__(self, cache_dir: str = "./.rubric_cache"):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cached rubrics
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key.

        Args:
            key: Cache key (question hash)

        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional["Rubric"]:
        """Retrieve a rubric from cache.

        Args:
            key: Cache key (question hash)

        Returns:
            Rubric if found, None otherwise
        """
        from .rubric_generator import Rubric

        path = self._get_cache_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Remove cache metadata before creating Rubric
            data.pop("_cached_at", None)
            return Rubric.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Invalid cache entry, remove it
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, rubric: "Rubric") -> None:
        """Store a rubric in cache.

        Args:
            key: Cache key (question hash)
            rubric: Rubric to cache
        """
        path = self._get_cache_path(key)
        data = rubric.to_dict()
        data["_cached_at"] = datetime.now().isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def delete(self, key: str) -> bool:
        """Delete a cached rubric.

        Args:
            key: Cache key (question hash)

        Returns:
            True if deleted, False if not found
        """
        path = self._get_cache_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cached rubrics.

        Returns:
            Number of items deleted
        """
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        items = list(self.cache_dir.glob("*.json"))
        total_size = sum(p.stat().st_size for p in items)
        return {
            "num_items": len(items),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir.absolute()),
        }

    def list_keys(self) -> list:
        """List all cached keys.

        Returns:
            List of cache keys
        """
        return [p.stem for p in self.cache_dir.glob("*.json")]

    def __contains__(self, key: str) -> bool:
        """Check if a key is in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists in cache
        """
        return self._get_cache_path(key).exists()

    def __len__(self) -> int:
        """Return number of cached items."""
        return len(list(self.cache_dir.glob("*.json")))
