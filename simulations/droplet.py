import numpy as np

from lattice import *
from boundary_conditions import *
from model import *


class Droplet(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_l = 1
        self.rho_v = 0.1

    def init_macro_vars(self):
        u = jnp.zeros((self.nx, self.ny, self.d))
        rho = self.rho_v * jnp.ones((self.nx, self.ny))
        x = jnp.arange(self.nx)
        y = jnp.arange(self.ny)
        i, j = jnp.meshgrid(x, y)
        r0 = (self.nx + self.ny)/8
        centre_x, centre_y = self.nx/2, self.ny/2
        distance = jnp.sqrt((i - centre_x) ** 2 + (j - centre_y) ** 2)
        mask = distance <= r0
        rho = rho.at[mask].set(self.rho_l)
        return [rho, u]

    def set_bc(self):
        pass

    def data(self, **kwargs):
        rho = kwargs["rho"]
        u = kwargs["u"]
        u_prev = kwargs["u_prev"]
        it = kwargs["it"]

        error = np.sum(np.abs(np.linalg.norm(u, axis=-1) - np.linalg.norm(u_prev, axis=-1)))
        print("error= {:06.5f}".format(error))


if __name__ == "__main__":
    lattice = LatticeD2Q9()

    nx = 200
    ny = 200

    Re = 200.0
