try:
    import cupy as xp
except ImportError:
    import numpy as xp
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from typing import List
from ..tensor import Tensor
from ..utils import get_logger

logger = get_logger(__name__)


class LinearOp(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        pass


class LinearMult(LinearOp):
    def __init__(self, lhs: List[Tensor], rhs: List[Tensor]):
        super().__init__()

        self._lhs = lhs
        self._rhs = rhs
        self.trivial = (len(lhs) + len(rhs) == 0)

    def __call__(self, t: Tensor) -> Tensor:
        if self.trivial:
            return deepcopy(t)
        else:
            res = reduce(lambda x, y: x * y, self._lhs + [t])
            res = reduce(lambda x, y: x * y, [res] + self._rhs)
            res._indices = deepcopy(t._indices)
            return res


def lanczos(op: LinearMult,
            x: Tensor,
            krylov_size: int,
            num_restarts: int,
            smallest: bool = True):
    v_next = deepcopy(x)
    beta = xp.zeros(krylov_size + 1)
    alpha = xp.zeros(krylov_size)
    for _ in range(num_restarts):
        beta[0] = 0.0
        v_prev = deepcopy(x).setZero()
        v_next.normalize()
        V = xp.zeros([x.size, krylov_size])
        for i in range(0, krylov_size):
            w = op(v_next)
            alpha[i] = xp.dot(w._data, v_next._data)
            w -= (v_next * alpha[i] + v_prev * beta[i])
            beta[i + 1] = w.norm()
            v_prev = deepcopy(v_next)
            v_next = w / beta[i + 1]
            V[:, i] = v_prev._data

        tridiag = xp.diag(alpha)
        for i in range(0, krylov_size - 1):
            tridiag[i, i + 1] = beta[i + 1]
            tridiag[i + 1, i] = beta[i + 1]
        d, v = xp.linalg.eigh(tridiag)

        if smallest:
            ev = d[0]
            v_next._data = V @ v[:, 0]
        else:
            ev = d[-1]
            v_next._data = V @ v[:, -1]

    return ev, v_next
