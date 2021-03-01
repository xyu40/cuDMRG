from copy import deepcopy
from cuDMRG import Index, Tensor

if __name__ == "__main__":
    a = Index(6)
    b = Index(2)
    c = Index(8)

    A = Tensor([a, b]).setRandom()
    B = Tensor([b, c]).setRandom()
    C = A * B

    print(C)

    lhs, rhs = C.deompose([0], [1])
    print(lhs._data.T @ lhs._data)
    print(rhs._data @ rhs._data.T)
