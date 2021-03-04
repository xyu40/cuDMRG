try:
    import cupy as xp
except ImportError:
    import numpy as xp
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple
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
    def __init__(self, lhs: List[Tensor], rhs: List[Tensor]) -> None:
        super().__init__()

        self._lhs_empty = False
        if len(lhs) > 0:
            self._lhs = deepcopy(lhs[0])
            for t in lhs[1:]:
                self._lhs *= t
        else:
            self._lhs_empty = True

        self._rhs_empty = False
        if len(rhs) > 0:
            self._rhs = deepcopy(rhs[0])
            for t in rhs[1:]:
                self._rhs *= t
        else:
            self._rhs_empty = True

        self.trivial = (len(lhs) + len(rhs) == 0)

    def __call__(self, t: Tensor) -> Tensor:
        if self.trivial:
            return deepcopy(t)
        else:
            if not self._lhs_empty:
                res = deepcopy(self._lhs)
                res *= t
            else:
                res = deepcopy(t)

            if not self._rhs_empty:
                res *= self._rhs
            res._indices = deepcopy(t._indices)
            return res


def lanczos(op: LinearOp,
            x: Tensor,
            krylov_size: int,
            num_restarts: int,
            smallest: bool = True) -> Tuple[float, Tensor]:
    v_next = deepcopy(x)
    v_prev = deepcopy(x)
    beta = xp.zeros(krylov_size + 1)
    alpha = xp.zeros(krylov_size)
    for _ in range(num_restarts):
        beta[0] = 0.0
        v_prev.setZero()
        v_next.normalize()
        V = xp.zeros([x.size, krylov_size])
        for i in range(0, krylov_size):
            w = op(v_next)
            alpha[i] = xp.dot(w._data.reshape(x.size),
                              v_next._data.reshape(x.size))
            v_next *= alpha[i]
            v_prev *= beta[i]
            w -= v_next
            w -= v_prev
            beta[i + 1] = w.norm()
            v_prev = v_next / alpha[i]
            v_next = w / beta[i + 1]
            V[:, i] = v_prev._data.reshape(x.size)

        tridiag = xp.diag(alpha)
        for i in range(0, krylov_size - 1):
            tridiag[i, i + 1] = beta[i + 1]
            tridiag[i + 1, i] = beta[i + 1]
        d, v = xp.linalg.eigh(tridiag)

        if smallest:
            ev = d[0]
            v_next._data = (V @ v[:, 0]).reshape(x._data.shape)
        else:
            ev = d[-1]
            v_next._data = (V @ v[:, -1]).reshape(x._data.shape)

    return ev, v_next.normalize()
