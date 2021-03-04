try:
    import cupy as xp
except ImportError:
    import numpy as xp
from copy import deepcopy
from typing import Dict, Any, Optional
from .sites import Sites
from .mps import MPS
from .mpo import MPO
from .solver import LinearMult, lanczos
from ..tensor import Tensor
from ..utils import get_logger

logger = get_logger(__name__)

CONFIG: Dict[str, Any] = {
    "num_sweeps": 10,
    "svd_error": 1e-10,
    "max_bond_dimension": 200,
    "lanczos_search_size": 3,
    "lanczos_num_restart": 1,
    "lanczos_smallest": True,
    "log_every_step": False
}


class DMRG:
    def __init__(self,
                 sites: Sites,
                 H: MPO,
                 psi: MPS = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        if sites.length <= 2:
            msg = "Length of the problem needs to be > 2"
            logger.error(msg)
            raise RuntimeError(msg)

        self._sites = sites
        self._H = H

        if psi is None:
            self._psi = MPS(sites, 1).setRandom().canonicalize()
        else:
            self._psi = psi.canonicalize()

        self._config = deepcopy(CONFIG)
        if config is not None:
            for key in config:
                self._config[key] = config[key]

        self._lEnvs: Dict[int, Tensor] = {}
        self._rEnvs: Dict[int, Tensor] = {}
        self._buildEnv()

    def run(self) -> None:
        L = self._sites.length
        for sweep in range(self._config["num_sweeps"]):
            for i in range(1, L):
                ev = self._update(i, move_right=True)
            for i in range(L - 1, 0, -1):
                ev = self._update(i, move_right=False)
            logger.info(f"sweep = {sweep}, E = {ev}")

    def _buildEnv(self):
        L = self._sites.length

        self._lEnvs[0] = Tensor([
            self._psi.leftIndex.raiseLevel(), self._H.leftIndex,
            self._psi.leftIndex
        ]).setOne()

        self._rEnvs[L - 1] = Tensor([
            self._psi.rightIndex.raiseLevel(), self._H.rightIndex,
            self._psi.rightIndex
        ]).setOne()

        for i in range(L - 2, -1, -1):
            self._rEnvs[i] = (self._psi.tensors[i + 1].raiseIndexLevel() *
                              self._rEnvs[i + 1])
            self._rEnvs[i] *= self._H.tensors[i + 1]
            self._rEnvs[i] *= self._psi.tensors[i + 1].lowerIndexLevel()

    def _update(self, i: int, move_right: bool = True) -> float:
        op = LinearMult(lhs=[self._lEnvs[i - 1], self._H.tensors[i - 1]],
                        rhs=[self._H.tensors[i], self._rEnvs[i]])
        x = self._psi.tensors[i - 1] * self._psi.tensors[i]

        ev, x = lanczos(op, x, self._config["lanczos_search_size"],
                        self._config["lanczos_num_restart"],
                        self._config["lanczos_smallest"])

        self._psi.tensors[i - 1], self._psi.tensors[i], s, dim = x.decompose(
            [0, 1], [2, 3], move_right, self._config["svd_error"],
            self._config["max_bond_dimension"])

        if self._config["log_every_step"]:
            vnEE = -xp.sum(s * xp.log(s))
            logger.info(f"Optimized bond ({i-1}, {i}), "
                        f"dim = {dim}, vnEE = {vnEE}, E = {ev}")

        if move_right:
            self._lEnvs[i] = (self._lEnvs[i - 1] *
                              self._psi.tensors[i - 1].raiseIndexLevel())
            self._lEnvs[i] *= self._H.tensors[i - 1]
            self._lEnvs[i] *= self._psi.tensors[i - 1].lowerIndexLevel()
        else:
            self._rEnvs[i - 1] = (self._psi.tensors[i].raiseIndexLevel() *
                                  self._rEnvs[i])
            self._rEnvs[i - 1] *= self._H.tensors[i]
            self._rEnvs[i - 1] *= self._psi.tensors[i].lowerIndexLevel()

        return ev