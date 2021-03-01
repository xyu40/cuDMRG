from cuDMRG.apps.sites import Sites
from cuDMRG import Index, Tensor, MPS

if __name__ == "__main__":
    a = Index(6)
    b = Index(4)
    c = Index(3)
    d = Index(5)
    print(a)
    print(b)
    print(c)
    print(d)

    A = Tensor([a, b, c]).setRandom()
    B = Tensor([c, d]).setRandom()
    C = (A * B).normalize()
    print(C)

    lhs, rhs, dim = C.deompose(lhs=[0, 1],
                               rhs=[2],
                               mergeV=False,
                               cutoff=1e-9,
                               maxdim=8)
    print(dim)
    print((lhs * rhs - C).norm())

    sites = Sites(10, 2)
    m = MPS(sites, 2).setRandom()
    m.canonicalize()

    print(m._tensors[0].norm())
    print(m._tensors[-1].norm())
