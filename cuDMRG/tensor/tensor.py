try:
    import cupy as xp
except ImportError:
    import numpy as xp
from numbers import Number
from copy import deepcopy
from typing import List, Optional, Any
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

    def __add__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number):
            res_tensor = deepcopy(self)
            res_tensor._data += rhs
            return res_tensor
        elif isinstance(rhs, Tensor):
            res_data = self._data + rhs._data
            res_indices = deepcopy(self._indices)
            return Tensor(res_indices, res_data)
        else:
            msg = f"Unsupported multiplication with {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __iadd__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number):
            self._data += rhs
        elif isinstance(rhs, Tensor):
            self._data = self._data + rhs._data
        else:
            msg = f"Unsupported multiplication with {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)
        return self

    def __sub__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number):
            res_tensor = deepcopy(self)
            res_tensor._data -= rhs
            return res_tensor
        elif isinstance(rhs, Tensor):
            res_data = self._data - rhs._data
            res_indices = deepcopy(self._indices)
            return Tensor(res_indices, res_data)
        else:
            msg = f"Unsupported multiplication with {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __isub__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number):
            self._data -= rhs
        elif isinstance(rhs, Tensor):
            self._data = self._data - rhs._data
        else:
            msg = f"Unsupported multiplication with {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)
        return self

    def __mul__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number):
            res_tensor = deepcopy(self)
            res_tensor._data *= rhs
            return res_tensor
        elif isinstance(rhs, Tensor):
            axes = getEinsumRule(self._indices, rhs._indices)
            res_indices = ([
                idx for i, idx in enumerate(self._indices) if i not in axes[0]
            ] + [
                idx for j, idx in enumerate(rhs._indices) if j not in axes[1]
            ])
            res_data = xp.tensordot(self._data, rhs._data, axes=axes)
            return Tensor(res_indices, res_data)
        else:
            msg = f"Unsupported multiplication with {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __rmul__(self, lhs: Any) -> "Tensor":
        if isinstance(lhs, Number):
            res_tensor = deepcopy(self)
            res_tensor._data *= lhs
            return res_tensor
        elif isinstance(lhs, Tensor):
            axes = getEinsumRule(lhs._indices, self._indices)
            res_indices = ([
                idx for j, idx in enumerate(lhs._indices) if j not in axes[1]
            ] + [
                idx for i, idx in enumerate(self._indices) if i not in axes[0]
            ])
            res_data = xp.tensordot(lhs._data, self._data, axes=axes)
            return Tensor(res_indices, res_data)
        else:
            msg = f"Unsupported multiplication with {type(lhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __imul__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number):
            self._data *= rhs
        elif isinstance(rhs, Tensor):
            axes = getEinsumRule(self._indices, rhs._indices)
            res_indices = ([
                idx for i, idx in enumerate(self._indices) if i not in axes[0]
            ] + [
                idx for j, idx in enumerate(rhs._indices) if j not in axes[1]
            ])
            res_data = xp.tensordot(self._data, rhs._data, axes=axes)
            self._indices = res_indices
            self._data = res_data
        else:
            msg = f"Unsupported multiplication with {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)
        return self

    def __truediv__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number):
            res_tensor = deepcopy(self)
            res_tensor._data /= rhs
            return res_tensor
        else:
            msg = f"Unsupported division with {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __idiv__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number):
            self._data /= rhs
            return self
        else:
            msg = f"Unsupported division with {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __str__(self) -> str:
        indices_str = ", ".join([str(idx) for idx in self._indices])
        data_str = str(self._data)
        return indices_str + "\n" + data_str
