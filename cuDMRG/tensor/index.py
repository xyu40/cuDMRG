import uuid
from enum import Enum
from typing import List, Tuple
from ..utils import get_logger

logger = get_logger(__name__)


class IndexType(Enum):
    VIRTUAL = 1
    PHYSICAL = 2
    ANYTYPE = 3


class Index:
    def __init__(
        self,
        size: int,
        index_type: IndexType = IndexType.VIRTUAL,
        level: int = 0,
    ) -> None:
        self._id = uuid.uuid4()
        self._size: int = size
        self._index_type: IndexType = index_type
        self._level: int = level

    @property
    def size(self) -> int:
        return self._size

    @property
    def indexType(self) -> IndexType:
        return self._index_type

    @property
    def level(self) -> int:
        return self._level

    def setSize(self, size: int) -> "Index":
        self._size = size
        return self

    def raiseLevel(self, index_type=IndexType.ANYTYPE) -> "Index":
        if index_type is IndexType.ANYTYPE or self._index_type == index_type:
            self._level += 1
        return self

    def lowerLevel(self, index_type=IndexType.ANYTYPE) -> "Index":
        if index_type is IndexType.ANYTYPE or self._index_type == index_type:
            self._level -= 1
        return self

    def setLevel(self, level: int, index_type=IndexType.ANYTYPE) -> "Index":
        if index_type is IndexType.ANYTYPE or self._index_type == index_type:
            self._level = level
        return self

    def resetLevel(self, index_type=IndexType.ANYTYPE) -> "Index":
        if index_type is IndexType.ANYTYPE or self._index_type == index_type:
            self._level = 0
        return self

    def mapLevel(self,
                 level_from: int,
                 level_to: int,
                 index_type=IndexType.ANYTYPE) -> "Index":
        if index_type is IndexType.ANYTYPE or self._index_type == index_type:
            if self._level == level_from:
                self._level = level_to
        return self

    def almostIndentical(self, rhs: "Index") -> bool:
        return self._id == rhs._id and self._level != rhs._level

    def __key(self) -> Tuple[int, int]:
        return (self._id.int, self._level)

    def __hash__(self) -> int:
        return hash(self.__key())

    def __eq__(self, rhs: object) -> bool:
        if isinstance(rhs, self.__class__):
            return self._id == rhs._id and self._level == rhs._level
        else:
            return False

    def __str__(self):
        return (f"Index({self._size}, {self._index_type}, "
                f"{self._level}, {self._id})")


def getEinsumRule(lhs: List[Index],
                  rhs: List[Index]) -> Tuple[List[int], List[int]]:
    lhs_list = []
    rhs_list = []
    lhs_map = {index: i for i, index in enumerate(lhs)}
    for j, index in enumerate(rhs):
        if index in lhs_map:
            lhs_list.append(lhs_map[index])
            rhs_list.append(j)
    return lhs_list, rhs_list
