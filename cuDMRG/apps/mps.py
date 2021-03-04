from copy import copy
from typing import List
from .sites import Sites
from ..tensor import Tensor
from ..utils import get_logger

logger = get_logger(__name__)


class MPS:
    def __init__(self, sites: Sites, VirtualDim: int = 1) -> None:
        self._center = 0
        self._length = sites.length
        self._physicalDim = sites.physicalDim

        physicalIndices = sites.physicalIndices
        virtualIndices = sites.virtualIndices
        for i, vidx in enumerate(virtualIndices):
            if i == 0 or i == self._length:
                vidx.setSize(1)
            else:
                vidx.setSize(VirtualDim)

        self._tensors: List[Tensor] = [
            Tensor([
                virtualIndices[i],
                physicalIndices[i],
                virtualIndices[i + 1],
            ]).setZero() for i in range(self._length)
        ]

    @property
    def length(self) -> int:
        return self._length

    @property
    def physicalDim(self) -> int:
        return self._physicalDim

    @property
    def leftIndex(self):
        return copy(self._tensors[0].indices[0])

    @property
    def rightIndex(self):
        return copy(self._tensors[-1].indices[-1])

    @property
    def tensors(self) -> List[Tensor]:
        return self._tensors

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
        for i in range(self._length - 1, 0, -1):
            lhs, rhs, _, _ = self._tensors[i].decompose(lhs=[0],
                                                        rhs=[1, 2],
                                                        mergeV=False)
            self._tensors[i] = rhs
            self._tensors[i - 1] *= lhs

        self._tensors[0].normalize()
        self._center = 0
        return self
