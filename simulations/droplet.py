from lattice import *
from boundary_conditions import *
from model import *


class Droplet(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_l = 1
        self.rho_v = 0.1

    def init_population_distribution_droplet(self):
        u = jnp.zeros((2, self.nx, self.ny))
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
