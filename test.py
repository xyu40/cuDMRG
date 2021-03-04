from cuDMRG import (Index, Tensor, MPS, Sites, Heisenberg, psiHphi, LinearMult,
                    lanczos, DMRG)
import numpy as np


def test_basic_tensor():
    a = Index(6)
    b = Index(4)
    c = Index(3)
    d = Index(5)

    A = Tensor([a, b, c]).setRandom()
    B = Tensor([c, d]).setRandom()
    C = A * B
    C.normalize()

    if not -1e-9 <= (C.norm() - 1.0) <= 1e-9:
        print("Basic test failed")
    else:
        lhs, rhs, _, _ = C.decompose(lhs=[0, 1],
                                     rhs=[2],
                                     mergeV=True,
                                     cutoff=1e-9,
                                     maxdim=8)

        if not -1e-9 <= (lhs * rhs - C).norm() <= 1e-9:
            print("Basic tensor test failed")
        else:
            print("Basic tensor test passed")


def test_basic_mps_mpo():
    sites = Sites(10, 2)
    psi = MPS(sites, 1).setOne().canonicalize()
    H = Heisenberg(sites, J=0, h=1).build()
    if not -1e-9 <= psiHphi(psi, H, psi) <= 1e-9:
        print("Basic mps mpo test failed")
    else:
        print("Basic mps mpo test passed")


def test_solver():
    a = Index(100)
    b = Index(100)
    A = Tensor([a, b]).setRandom()
    B = A.transpose([1, 0], inplace=False)
    B.indices[-1].raiseLevel()

    M = A * B

    op = LinearMult([A, B], [])
    x = Tensor([a]).setRandom().raiseIndexLevel()
    ev, x = lanczos(op, x, 4, 10, smallest=False)

    w, _ = np.linalg.eigh(M._data)

    if not -1e-3 <= (ev - w[-1]) <= 1e-3:
        print("Basic solver test failed")
    else:
        print("Basic solver test passed")


def test_basic_dmrg():
    sites = Sites(100, 2)
    H = Heisenberg(sites, J=1, h=0).build()
    model = DMRG(sites, H)
    model.run()


if __name__ == "__main__":
    test_basic_tensor()
    test_basic_mps_mpo()
    test_solver()
    test_basic_dmrg()
