from copy import copy
from typing import List
from ..tensor import Index, IndexType
from ..utils import get_logger

logger = get_logger(__name__)


class Sites:
    def __init__(self, length: int, physicalDim: int) -> None:
        self._length = length
        self._physicalDim = physicalDim

        self._physicalIndices = [
            Index(physicalDim, IndexType.PHYSICAL) for i in range(length)
        ]

    @property
    def length(self) -> int:
        return self._length

    @property
    def physicalDim(self) -> int:
        return self._physicalDim

    @property
    def virtualIndices(self) -> List[Index]:
        # must be freshly generated everytime
        return [Index(1, IndexType.VIRTUAL) for i in range(self._length + 1)]

    @property
    def physicalIndices(self) -> List[Index]:
        return [copy(idx) for idx in self._physicalIndices]
