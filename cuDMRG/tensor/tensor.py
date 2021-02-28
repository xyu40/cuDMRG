try:
    import cupy as xp
except ImportError:
    import numpy as xp
from copy import deepcopy
from typing import List, Optional
from .index import Index, getEinsumRule
from ..utils import get_logger

logger = get_logger(__name__)


class Tensor:
    def __init__(
        self,
        indices: List[Index],
        data: Optional[xp.ndarray] = None,
    ):
        if data is not None and len(indices) != len(data.shape):
            error_msg = "indices shape does not match data shape"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self._rank = len(indices)
        self._indices = deepcopy(indices)
        if data is None:
            self.setZero()
        else:
            self._data = deepcopy(data)

    def setZero(self) -> "Tensor":
        self._data = xp.zeros([idx.size for idx in self._indices])
        return self

    def setOne(self) -> "Tensor":
        self._data = xp.ones((idx.size for idx in self._indices))
        return self

    def setRandom(self) -> "Tensor":
        self._data = xp.random.random([idx.size for idx in self._indices])
        return self

    def __mul__(self, rhs: "Tensor") -> "Tensor":
        axes = getEinsumRule(self._indices, rhs._indices)
        res_indices = (
            [idx for i, idx in enumerate(self._indices) if i not in axes[0]] +
            [idx for j, idx in enumerate(rhs._indices) if j not in axes[1]])
        res_data = xp.tensordot(self._data, rhs._data, axes=axes)
        return Tensor(res_indices, res_data)

    def __str__(self):
        indices_str = ", ".join([str(idx) for idx in self._indices])
        data_str = str(self._data)
        return indices_str + "\n" + data_str
