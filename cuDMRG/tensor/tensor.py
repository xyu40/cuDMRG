try:
    import cupy as xp
    from cupy import cutensor
    USE_CUPY = True
except ImportError:
    import numpy as xp
    USE_CUPY = False
from numbers import Number
from copy import copy
from functools import reduce
from typing import List, Tuple, Optional, Any
from .index import Index, IndexType, getEinsumRule
from ..utils import get_logger

logger = get_logger(__name__)


class Tensor:
    def __init__(self,
                 indices: List[Index],
                 data: Optional[xp.ndarray] = None,
                 use_cutensor: bool = False) -> None:
        if data is not None and len(indices) != len(data.shape):
            error_msg = "indices shape does not match data shape"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self._rank = len(indices)
        self._indices = [copy(idx) for idx in indices]
        if data is None:
            self.setZero()
        else:
            self._data = data
        self.use_cutensor = USE_CUPY and use_cutensor

    def copy(self) -> "Tensor":
        res = Tensor([])
        res._rank = self.rank
        res._indices = [copy(idx) for idx in self._indices]
        res._data = self._data
        return res

    def deepcopy(self) -> "Tensor":
        res = Tensor([])
        res._rank = self.rank
        res._indices = [copy(idx) for idx in self._indices]
        res._data = self._data.copy()
        return res

    def norm(self) -> float:
        return xp.linalg.norm(self._data)

    def normalize(self) -> "Tensor":
        self._data /= self.norm()
        return self

    def setZero(self) -> "Tensor":
        self._data = xp.zeros([idx.size for idx in self._indices])
        return self

    def setOne(self) -> "Tensor":
        self._data = xp.ones([idx.size for idx in self._indices])
        return self

    def setRandom(self) -> "Tensor":
        self._data = xp.random.random([idx.size for idx in self._indices])
        return self

    def raiseIndexLevel(self,
                        indexType: IndexType = IndexType.ANYTYPE) -> "Tensor":
        for idx in self._indices:
            idx.raiseLevel(indexType)
        return self

    def lowerIndexLevel(self,
                        indexType: IndexType = IndexType.ANYTYPE) -> "Tensor":
        for idx in self._indices:
            idx.lowerLevel(indexType)
        return self

    def resetIndexLevel(self,
                        indexType: IndexType = IndexType.ANYTYPE) -> "Tensor":
        for idx in self._indices:
            idx.resetLevel(indexType)
        return self

    def mapIndexLevel(self,
                      level_from: int,
                      level_to: int,
                      indexType: IndexType = IndexType.ANYTYPE) -> "Tensor":
        for idx in self._indices:
            idx.mapLevel(level_from, level_to, indexType)
        return self

    def transpose(self, axes, inplace=True) -> "Tensor":
        if len(set(axes)) != len(axes):
            msg = "Invalid transpose input"
            logger.error(msg)
            raise RuntimeError(msg)

        transpose_needed = False
        for i, j in enumerate(axes):
            if i != j:
                transpose_needed = True
                break

        if inplace:
            if transpose_needed:
                self._indices = [self._indices[i] for i in axes]
                self._data = xp.transpose(self._data, axes=axes)
            return self
        else:
            res = self.copy()
            if transpose_needed:
                res._indices = [res._indices[i] for i in axes]
                res._data = xp.transpose(res._data, axes=axes)
            return res

    def diagonal(self) -> "Tensor":
        if self._rank % 2 != 0:
            msg = "Cannot get diagonal from Tensor with odd rank"
            logger.error(msg)
            raise RuntimeError(msg)

        lhs_indices = []
        rhs_indices = []
        for i in range(self._rank):
            for j in range(i + 1, self._rank):
                if self._indices[i].almostIndentical(self._indices[j]):
                    lhs_indices.append(i)
                    rhs_indices.append(j)
        self.transpose(lhs_indices + rhs_indices)

        res_size = [self._indices[i].size for i in lhs_indices]
        diag_size = reduce(lambda x, y: x * y, res_size)
        res_indices = [
            copy(self._indices[i]).resetLevel() for i in lhs_indices
        ]
        res_data = xp.diag(self._data.reshape(diag_size, -1)).reshape(res_size)
        return Tensor(res_indices, res_data)

    def decompose(
            self,
            lhs: List[int],
            rhs: List[int],
            mergeV: bool = True,
            cutoff: float = 1e-12,
            maxdim: int = 2147483648
    ) -> Tuple["Tensor", "Tensor", xp.array, int]:
        lhs_size = reduce(lambda x, y: x * y,
                          [self._indices[i].size for i in lhs])
        rhs_size = reduce(lambda x, y: x * y,
                          [self._indices[i].size for i in rhs])
        self.transpose(lhs + rhs)
        u, s, v = xp.linalg.svd(self._data.reshape([lhs_size, rhs_size]),
                                full_matrices=False,
                                compute_uv=True)

        s_norm = xp.linalg.norm(s)
        s_cutoff = (1 - cutoff) * s_norm * s_norm
        s_squared_cumsum = xp.cumsum(xp.power(s, 2))

        # dim = 0
        # for i in range(s.size):
        #     dim += 1
        #     if s_squared_cumsum[i] >= s_cutoff or (dim + 1) > maxdim:
        #         break
        dim = int(xp.searchsorted(s_squared_cumsum[:maxdim], s_cutoff)) + 1
        dim = min(dim, s.size, maxdim)

        u = u[:, :dim]
        s = xp.clip(s[:dim] * s_norm / xp.sqrt(s_squared_cumsum[dim - 1]),
                    a_min=1e-32,
                    a_max=None)
        v = v[:dim, :]

        if mergeV:
            v = xp.diag(s) @ v
        else:
            u = u @ xp.diag(s)

        a = Index(dim)
        lhs_indices = self._indices[:len(lhs)] + [a]
        rhs_indices = [a] + self._indices[len(lhs):]
        lhs_tensor = Tensor(lhs_indices,
                            u.reshape([idx.size for idx in lhs_indices]))
        rhs_tensor = Tensor(rhs_indices,
                            v.reshape([idx.size for idx in rhs_indices]))
        return lhs_tensor, rhs_tensor, s, dim

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def indices(self) -> List[Index]:
        return self._indices

    def __add__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number) or isinstance(rhs, xp.ndarray):
            res_tensor = self.deepcopy()
            res_tensor._data += rhs
            return res_tensor
        elif isinstance(rhs, Tensor):
            res = self.deepcopy()
            res._data += rhs._data
            return res
        else:
            msg = f"Unsupported __add__ with rhs of type {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __iadd__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number) or isinstance(rhs, xp.ndarray):
            self._data += rhs
        elif isinstance(rhs, Tensor):
            self._data = self._data + rhs._data
        else:
            msg = f"Unsupported __iadd__ with rhs of type {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)
        return self

    def __sub__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number) or isinstance(rhs, xp.ndarray):
            res_tensor = self.deepcopy()
            res_tensor._data -= rhs
            return res_tensor
        elif isinstance(rhs, Tensor):
            res = self.deepcopy()
            res._data -= rhs._data
            return res
        else:
            msg = f"Unsupported __sub__ with rhs of type {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __isub__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number) or isinstance(rhs, xp.ndarray):
            self._data -= rhs
        elif isinstance(rhs, Tensor):
            self._data = self._data - rhs._data
        else:
            msg = f"Unsupported __isub__ with rhs of type {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)
        return self

    def __mul__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number) or isinstance(rhs, xp.ndarray):
            res_tensor = self.deepcopy()
            res_tensor._data *= rhs
            return res_tensor
        elif isinstance(rhs, Tensor):
            axes = getEinsumRule(self._indices, rhs._indices)
            res_indices = ([
                idx for i, idx in enumerate(self._indices) if i not in axes[0]
            ] + [
                idx for j, idx in enumerate(rhs._indices) if j not in axes[1]
            ])
            if not self.use_cutensor:
                res_data = xp.tensordot(self._data, rhs._data, axes=axes)
                return Tensor(res_indices, res_data)
            else:
                a = xp.ascontiguousarray(self._data)
                b = xp.ascontiguousarray(rhs._data)
                c = xp.zeros([idx.size for idx in res_indices])
                desc_a = cutensor.create_tensor_descriptor(a)
                desc_b = cutensor.create_tensor_descriptor(b)
                desc_c = cutensor.create_tensor_descriptor(c)
                mode_a = [chr(97 + i) for i in range(self._rank)]
                mode_b = [
                    chr(97 + i)
                    for i in range(self._rank, self._rank + rhs._rank)
                ]
                for i, j in zip(axes[0], axes[1]):
                    mode_b[j] = mode_a[i]
                mode_c = (
                    [mode_a[i]
                     for i in range(self._rank) if i not in axes[0]] +
                    [mode_b[j] for j in range(rhs._rank) if j not in axes[1]])
                mode_a = cutensor.create_mode(*mode_a)
                mode_b = cutensor.create_mode(*mode_b)
                mode_c = cutensor.create_mode(*mode_c)
                cutensor.contraction(1.0, a, desc_a, mode_a, b, desc_b, mode_b,
                                     0.0, c, desc_c, mode_c)
                return Tensor(res_indices, c)
        else:
            msg = f"Unsupported __mul__ with rhs of type {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __imul__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number) or isinstance(rhs, xp.ndarray):
            self._data *= rhs
        elif isinstance(rhs, Tensor):
            axes = getEinsumRule(self._indices, rhs._indices)
            res_indices = ([
                idx for i, idx in enumerate(self._indices) if i not in axes[0]
            ] + [
                idx for j, idx in enumerate(rhs._indices) if j not in axes[1]
            ])
            if not self.use_cutensor:
                self._data = xp.tensordot(self._data, rhs._data, axes=axes)
            else:
                a = xp.ascontiguousarray(self._data)
                b = xp.ascontiguousarray(rhs._data)
                c = xp.zeros([idx.size for idx in res_indices])
                desc_a = cutensor.create_tensor_descriptor(a)
                desc_b = cutensor.create_tensor_descriptor(b)
                desc_c = cutensor.create_tensor_descriptor(c)
                mode_a = [chr(97 + i) for i in range(self._rank)]
                mode_b = [
                    chr(97 + i)
                    for i in range(self._rank, self._rank + rhs._rank)
                ]
                for i, j in zip(axes[0], axes[1]):
                    mode_b[j] = mode_a[i]
                mode_c = (
                    [mode_a[i]
                     for i in range(self._rank) if i not in axes[0]] +
                    [mode_b[j] for j in range(rhs._rank) if j not in axes[1]])
                mode_a = cutensor.create_mode(*mode_a)
                mode_b = cutensor.create_mode(*mode_b)
                mode_c = cutensor.create_mode(*mode_c)
                cutensor.contraction(1.0, a, desc_a, mode_a, b, desc_b, mode_b,
                                     0.0, c, desc_c, mode_c)
                self._data = c
            self._indices = res_indices
            self._rank = len(self._indices)
        else:
            msg = f"Unsupported __imul__ with rhs of type {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)
        return self

    def __rmul__(self, lhs: Any) -> "Tensor":
        if isinstance(lhs, Number) or isinstance(lhs, xp.ndarray):
            res_tensor = self.deepcopy()
            res_tensor._data *= lhs
            return res_tensor
        else:
            msg = f"Unsupported __rmul__ with lhs of type {type(lhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __truediv__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number) or isinstance(rhs, xp.ndarray):
            res_tensor = self.deepcopy()
            res_tensor._data /= rhs
            return res_tensor
        else:
            msg = f"Unsupported __truediv__ with rhs of type {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __idiv__(self, rhs: Any) -> "Tensor":
        if isinstance(rhs, Number) or isinstance(rhs, xp.ndarray):
            self._data /= rhs
            return self
        else:
            msg = f"Unsupported __idiv__ with rhs of type {type(rhs)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def __str__(self) -> str:
        rank_str = f"rank = {self._rank}"
        indices_str = "\n".join([str(idx) for idx in self._indices])
        data_str = str(self._data.shape)
        return "\n".join([rank_str, indices_str, data_str])
