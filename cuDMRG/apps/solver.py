try:
    import cupy as xp
except ImportError:
    import numpy as xp
from abc import ABC, abstractmethod
from typing import Tuple
from ..tensor import Tensor
from ..utils import get_logger

logger = get_logger(__name__)


class LinearOp(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, t: Tensor):
        pass


class LinearMult(LinearOp):
    def __init__(self, lhs: Tensor, rhs: Tensor, x: Tensor) -> None:
        super().__init__()
        self._lhs = lhs
        self._rhs = rhs
        self._x = x.copy()

    def __call__(self, t: xp.ndarray) -> xp.ndarray:
        self._x._data = t
        res = self._lhs * self._x
        res *= self._rhs
        return res._data


def lanczos(op: LinearOp,
            x: Tensor,
            krylov_size: int,
            num_restarts: int,
            smallest: bool = True) -> Tuple[float, Tensor]:
    v_next = x._data.copy()
    beta = xp.zeros(krylov_size + 1)
    alpha = xp.zeros(krylov_size)
    for _ in range(num_restarts):
        beta[0] = 0.0
        v_prev = xp.zeros(x._data.shape)
        v_next /= xp.linalg.norm(v_next)
        V = xp.zeros([x.size, krylov_size])
        for i in range(0, krylov_size):
            w = op(v_next)
            alpha[i] = xp.dot(w.reshape(x.size), v_next.reshape(x.size))
            w -= (alpha[i] * v_next + beta[i] * v_prev)
            beta[i + 1] = xp.linalg.norm(w)
            v_prev = v_next.copy()
            v_next = w / beta[i + 1]
            V[:, i] = v_prev.reshape(x.size)

        tridiag = xp.diag(alpha)
        for i in range(0, krylov_size - 1):
            tridiag[i + 1, i] = beta[i + 1]
        d, v = xp.linalg.eigh(tridiag, UPLO="L")

        if smallest:
            ev = d[0]
            v_next = (V @ v[:, 0]).reshape(x._data.shape)
        else:
            ev = d[-1]
            v_next = (V @ v[:, -1]).reshape(x._data.shape)
    x._data = v_next
    return ev, x
