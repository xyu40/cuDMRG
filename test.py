from cuDMRG import Index, Tensor, MPS, Sites, Heisenberg, psiHphi

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
    C = A * B
    print(C.norm())
    C.normalize()
    print(C.norm())

    lhs, rhs, s, dim = C.deompose(lhs=[0, 1],
                                  rhs=[2],
                                  mergeV=False,
                                  cutoff=1e-9,
                                  maxdim=8)
    print(dim)
    print((lhs * rhs - C).norm())

    sites = Sites(10, 2)
    psi = MPS(sites, 1).setRandom().canonicalize()
    phi = MPS(sites, 1).setRandom().canonicalize()
    H = Heisenberg(sites, J=1, h=0).build()
    print(psiHphi(psi, H, phi))
