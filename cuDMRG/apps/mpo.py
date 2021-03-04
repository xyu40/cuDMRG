from abc import ABC, abstractmethod
from copy import copy
from typing import List
from .sites import Sites
from .mps import MPS
from ..tensor import Tensor
from ..utils import get_logger

logger = get_logger(__name__)


class MPO(ABC):
    def __init__(self, sites: Sites, VirtualDim: int = 1) -> None:
        self._length = sites.length
        self._physicalDim = sites.physicalDim

        physicalIndices = sites.physicalIndices
        physicalIndicesPrime = [
            copy(idx).raiseLevel() for idx in physicalIndices
        ]
        virtualIndices = sites.virtualIndices
        for i, vidx in enumerate(virtualIndices):
            if i == 0 or i == self._length:
                vidx.setSize(1)
            else:
                vidx.setSize(VirtualDim)

        self._tensors: List[Tensor] = [
            Tensor([
                virtualIndices[i],
                physicalIndicesPrime[i],
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
    def tensors(self) -> List[Tensor]:
        return self._tensors

    @property
    def leftIndex(self):
        return copy(self._tensors[0].indices[0])

    @property
    def rightIndex(self):
        return copy(self._tensors[-1].indices[-1])

    @abstractmethod
    def build(self) -> "MPO":
        pass


def psiHphi(psi: MPS, H: MPO, phi: MPS) -> float:
    if (psi.length != H.length or psi.length != phi.length
            or psi.physicalDim != H.physicalDim
            or psi.physicalDim != phi.physicalDim):
        msg = "Input dimensions do not match"
        logger.error(msg)
        raise RuntimeError(msg)

    left_indices = [psi.leftIndex.raiseLevel(), H.leftIndex, phi.leftIndex]
    res = Tensor(left_indices).setOne()

    if psi is phi:
        for i in range(H.length):
            res *= psi.tensors[i].copy().raiseIndexLevel()
            res *= H.tensors[i]
            res *= phi.tensors[i]

    else:
        for i in range(H.length):
            res *= psi.tensors[i].raiseIndexLevel()
            res *= H.tensors[i]
            res *= phi.tensors[i]
            psi.tensors[i].lowerIndexLevel()

    return res._data.reshape(res._data.size)[0]
