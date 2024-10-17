import jax.numpy as jnp
import numpy as np

from lattice import LatticeD2Q9
from model import BGK
from func import *


class Couette2D(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_bc(self):
        pass

    def data(self):
        rho = np.array(kwargs['rho'])
        u = np.array(kwargs['u'])
        it = np.array(kwargs['it'])
        u_prev = np.array(kwargs['u_prev'])

        error = np.sum(np.abs(np.linalg.norm(u, axis=-1) - np.linalg.norm(u_prev, axis=-1)))
        print("error= {:06.5f}".format(error))

        plot = Plot()
        plot.rho(it, rho)


if __name__ == "__main__":
    lattice = LatticeD2Q9()
    nx = 500
    ny = 100

    tau = 1.0

    kwargs = {
        'lattice': lattice,
        'tau': tau,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'save_info_rate': 100,
        'print_info_rate': 100
    }
    sim = Couette2D(**kwargs)
    sim.run(1000)
