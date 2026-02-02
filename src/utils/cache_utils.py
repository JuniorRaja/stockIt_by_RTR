"""Lightweight cache utilities."""

from collections import OrderedDict
from typing import Dict, Iterable, Iterator, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache:
    """Minimal LRU cache with bounded entries."""

    def __init__(self, max_entries: Optional[int] = None):
        self.max_entries = max_entries
        self._data: Dict[K, V] = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: K, value: V) -> None:
        if self.max_entries is not None and self.max_entries <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        self._evict()

    def delete(self, key: K) -> None:
        if key in self._data:
            del self._data[key]

    def clear(self) -> None:
        self._data.clear()

    def items(self) -> Iterable[Tuple[K, V]]:
        return self._data.items()

    def __contains__(self, key: K) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def _evict(self) -> None:
        if self.max_entries is None or self.max_entries <= 0:
            return
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)
