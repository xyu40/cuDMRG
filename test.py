from copy import deepcopy
from cuDMRG import Index, Tensor
try:
    import cupy as xp
except ImportError:
    import numpy as xp

if __name__ == "__main__":
    a = Index(40)
    b = Index(20)
    c = Index(80)

    A = Tensor([a, b]).setRandom()
    B = Tensor([b, c]).setRandom()
    C = (A * B).normalize()
    print(C)

    lhs, rhs, dim = C.deompose(lhs=[0], rhs=[1], cutoff=1e-9, maxdim=18)
    print(dim, xp.linalg.norm(lhs._data.T @ lhs._data - xp.eye(dim)))
    print((lhs * rhs - C).norm())
