try:
    import cupy as xp
except ImportError:
    import numpy as xp
from .sites import Sites
from .mpo import MPO
from ..utils import get_logger

logger = get_logger(__name__)


class Heisenberg(MPO):
    """
    H = sum_i J/2(S+_i S-_{i+1} + S-_i S+_{i+1}) + J Sz_i Sz_{+1} + h S_z_i
    On each site, except at the boundary, the matrix looks like
    I     0     0     0     0
    hSz   I     JSz   JS-/2 JS+/2
    Sz    0     0     0     0
    S+    0     0     0     0
    S-    0     0     0     0
    """
    def __init__(self, sites: Sites, J: float = 1.0, h: float = 0.0) -> None:
        if sites.physicalDim != 2:
            msg = "Wrong physical dimension for Heisenberg MPO!"
            logger.error(msg)
            raise RuntimeError(msg)
        super().__init__(sites, 5)
        self._J = float(J)
        self._h = float(h)

    def build(self) -> "Heisenberg":
        # the xp.transpose is needed to make the data layout match
        # the indices definition in base class
        # self._tensors: List[Tensor] = [
        #     Tensor([
        #         virtualIndices[i],
        #         physicalIndicesPrime[i],
        #         physicalIndices[i],
        #         virtualIndices[i + 1],
        #     ]).setZero() for i in range(self._length)
        # ]
        self._tensors[0]._data = xp.ascontiguousarray(
            xp.transpose(
                xp.array([
                    [self._h / 2, 1.0, self._J / 2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, self._J / 2],
                    [0.0, 0.0, 0.0, self._J / 2, 0.0],
                    [-self._h / 2, 1.0, -self._J / 2, 0.0, 0.0],
                ]).reshape(2, 2, 1, 5),
                axes=[2, 0, 1, 3],
            ))

        for i in range(1, self._length - 1):
            self._tensors[i]._data = xp.ascontiguousarray(
                xp.transpose(
                    xp.array([
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [self._h / 2, 1.0, self._J / 2, 0.0, 0.0],
                            [0.5, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, self._J / 2],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, self._J / 2, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [-self._h / 2, 1.0, -self._J / 2, 0.0, 0.0],
                            [-0.5, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]).reshape(2, 2, 5, 5),
                    axes=[2, 0, 1, 3],
                ))

        self._tensors[-1]._data = xp.ascontiguousarray(
            xp.transpose(
                xp.array([
                    [1.0, self._h / 2, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, -self._h / 2, -0.5, 0.0, 0.0],
                ]).reshape(2, 2, 5, 1),
                axes=[2, 0, 1, 3],
            ))

        return self
