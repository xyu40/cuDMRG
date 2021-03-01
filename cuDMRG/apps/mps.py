from .sites import Sites
from ..tensor import Tensor
from ..utils import get_logger

logger = get_logger(__name__)


class MPS:
    def __init__(self, sites: Sites, VirtualDim: int = 1) -> None:
        self._center = 0
        self._length = sites.length
        self._physicalDim = sites.physicalDim
        self._physicalIndices = sites.physicalIndices

        self._virtualIndices = sites.virtualIndices
        for i, vidx in enumerate(self._virtualIndices):
            if i == 0 or i == self._length:
                vidx.setSize(1)
            else:
                vidx.setSize(VirtualDim)

        self._tensors = [
            Tensor([
                self._virtualIndices[i],
                self._physicalIndices[i],
                self._virtualIndices[i + 1],
            ]).setZero() for i in range(self._length)
        ]

    def setZero(self) -> "MPS":
        for t in self._tensors:
            t.setZero()
        return self

    def setRandom(self) -> "MPS":
        for t in self._tensors:
            t.setRandom()
        return self

    def setOne(self) -> "MPS":
        for t in self._tensors:
            t.setOne()
        return self

    def canonicalize(self) -> "MPS":
        for i in range(self._length - 1, -1, -1):
            lhs, rhs, _ = self._tensors[i].deompose(lhs=[0],
                                                    rhs=[1, 2],
                                                    mergeV=False)
            self._tensors[i] = rhs
            if i > 0:
                self._tensors[i - 1] *= lhs
        self._center = 0
        return self
