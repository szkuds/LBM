import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from termcolor import colored
from jax import jit, lax, vmap


class LBM(object):
    def __init__(self, **kwargs):
        self.tau = kwargs.get("tau")
        self.nx = kwargs.get("nx")
        self.ny = kwargs.get("ny")
        self.nz = kwargs.get("nz")

        self.lattice = kwargs.get("lattice")
        self.print_info_rate = kwargs.get("printInfoRate", 100)
        self.save_info_rate = kwargs.get("saveInfoRate", 0)

        # Extract relevant parameters from lattice
        self.c = self.lattice.c
        self.q = self.lattice.q
        self.w = self.lattice.w
        self.d = self.lattice.d
        self.cs = self.lattice.cs
        self.cs2 = self.lattice.cs2
        self.inv_cs2: float = self.lattice.inv_cs2
        self.i = self.lattice.i

        self.print_simulation_parameters()

        # Store grid information
        self.grid_info = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "D": self.lattice.d,
            "Q": self.lattice.q,
            "lattice": self.lattice
        }

        # Compute the bounding box indices for boundary conditions
        self.bc_i = self.bc_i()
        # Create boundary data for the simulation
        self.create_boundary_data()
        self.force = self.calculate_force()

        # Directory to store the data
        self.save_dir = kwargs.get("save_dir", "./sim_data/")

    def run(self, nt):
        f = self.init_f
        for it in range(nt):
            save_flag = self.save_info_rate > 0 and it % self.save_info_rate == 0
            print_flag = self.print_info_rate > 0 and it % self.print_info_rate == 0

            f, f_post_col = self.update(f)

            if print_flag:
                print("Iteration" + f"{it} of {nt}" + "completed")

            if save_flag:
                # Save data
                rho_prev, u_prev = self.macro_vars(f)
                rho, u = self.macro_vars(f_post_col)
                self.save(it, f, f_post_col, rho, rho_prev, u, u_prev)
        return f

    @partial(jax.jit, static_argnums=0)
    def update(self, f_prev):
        f_post_col = self.collision(f_prev)
        f_post_col = self.apply_bc(f_post_col, f_prev)
        f_post_col = self.streaming(f_post_col)
        return f_post_col, f_prev

    @property
    def init_f(self):
        rho0, u0 = self.init_macro_vars()
        return self.equilibrium(rho0, u0)

    def init_macro_vars(self):
        return jnp.ones((self.nx, self.ny)), jnp.zeros((self.nx, self.ny, self.d))

    @partial(jax.jit, static_argnums=0)
    def equilibrium(self, rho, u):
        cu = self.inv_cs2 * (jnp.einsum('xya, ai-> xyi', u, self.c))
        usq = 0.5 * self.inv_cs2 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        rho_w = jnp.einsum('xy, i->xyi', rho, self.w)
        return rho_w * (1.0 + cu * (1.0 + 0.5 * cu) - usq)

    #  TODO: Need to ensure that in the documentation it is noted that the index should always be the last element
    @partial(jit, static_argnums=0, inline=True)
    def macro_vars(self, f):
        rho = jnp.sum(f, axis=-1)
        u = jnp.dot(f, self.c.T) / rho[..., np.newaxis]
        return rho, u

    def calculate_force(self):
        pass

    def apply_force(self, f_post_col, f_eq, rho, u):
        pass

    def create_boundary_data(self):
        self.BCs = []
        self.set_bc()

    #  boundary condition indices
    def bc_i(self):
        if self.d == 2:
            return {
                "bottom": np.array([[i, 0] for i in range(self.nx)], dtype=int),
                "top": np.array([[i, self.ny - 1] for i in range(self.nx)], dtype=int),
                "left": np.array([[0, i] for i in range(self.ny)], dtype=int),
                "right": np.array([[self.nx - 1, i] for i in range(self.ny)], dtype=int)}

        elif self.d == 3:
            return {
                "bottom": np.array([[i, j, 0] for i in range(self.nx) for j in range(self.ny)], dtype=int),
                "top": np.array([[i, j, self.nz - 1] for i in range(self.nx) for j in range(self.ny)], dtype=int),
                "left": np.array([[0, j, k] for j in range(self.ny) for k in range(self.nz)], dtype=int),
                "right": np.array([[self.nx - 1, j, k] for j in range(self.ny) for k in range(self.nz)], dtype=int),
                "front": np.array([[i, 0, k] for i in range(self.nx) for k in range(self.nz)], dtype=int),
                "back": np.array([[i, self.ny - 1, k] for i in range(self.nx) for k in range(self.nz)], dtype=int)}

    def set_bc(self):
        """
        Needs to be implemented in subclass
        """
        pass

    @partial(jit, static_argnums=(0,))
    def streaming(self, f):
        def stream_i(f_i, c):
            if self.d == 2:
                return jnp.roll(f_i, (c[0], c[1]), axis=(0, 1))
            if self.d == 3:
                return jnp.roll(f_i, (c[0], c[1], c[2]), axis=(0, 1, 2))

        return jax.vmap(stream_i, in_axes=(-1, 0), out_axes=-1)(f, self.c.T)

    def collision(self, f):
        pass

    def data(self, **kwargs):
        pass

    def save(self, it, f, f_post_col, rho, rho_prev, u, u_prev):
        kwargs = {
            "it": it,
            "f": f,
            "f_post_col": f_post_col,
            "rho": rho,
            "rho_prev": rho_prev,
            "u": u,
            "u_prev": u_prev
        }
        self.data(**kwargs)

    def print_simulation_parameters(self):
        descriptive_names = {
            'tau': 'Relaxation Time',
            'nx': 'Grid Points in X',
            'ny': 'Grid Points in Y',
            'nz': 'Grid Points in Z',
            'dim': 'Dimensionality',
            'lattice': 'Lattice Type',
            'print_info_rate': 'Print Info Rate',
            'save_info_rate': 'Save Info Rate',
            'c': 'Lattice Velocities',
            'q': 'Number of Discrete Velocities',
            'w': 'Lattice Weights'
        }
        simulation_name = self.__class__.__name__

        print(colored(f'**** Simulation Parameters for {simulation_name} ****', 'green'))

        header = f"{colored('Parameter', 'blue'):>30} | {colored('Value', 'yellow')}"
        print(header)
        print('-' * 50)

        for attr, value in self.__dict__.items():
            if not attr.startswith('_'):  # Skip private attributes
                descriptive_name = descriptive_names.get(attr, attr.capitalize())
                if isinstance(value, (int, float, str, bool)):
                    row = f"{colored(descriptive_name, 'blue'):>30} | {colored(value, 'yellow')}"
                elif isinstance(value, (list, tuple, np.ndarray)):
                    row = f"{colored(descriptive_name, 'blue'):>30} | {colored('Array/List', 'yellow')}"
                elif hasattr(value, '__name__'):
                    row = f"{colored(descriptive_name, 'blue'):>30} | {colored(value.__name__, 'yellow')}"
                else:
                    row = f"{colored(descriptive_name, 'blue'):>30} | {colored(type(value).__name__, 'yellow')}"
                print(row)

    def apply_bc(self, f_post_col, f_prev):
        return f_post_col

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        if value is None:
            raise ValueError("tau must be provided")
        if not isinstance(value, (int, float)):
            raise TypeError("tau must be an integer or a float")
        if value <= 0:
            raise ValueError("tau must be a positive integer")
        self._tau = value

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        if value is None:
            raise ValueError("nx must be provided")
        if not isinstance(value, int):
            raise TypeError("nx must be an integer")
        if value <= 0:
            raise ValueError("nx must be a positive integer")
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        if value is None:
            raise ValueError("ny must be provided")
        if not isinstance(value, int):
            raise TypeError("ny must be an integer")
        if value <= 0:
            raise ValueError("ny must be a positive integer")
        self._ny = value

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value is None:
            raise ValueError("nz must be provided")
        if not isinstance(value, int):
            raise TypeError("nz must be an integer")
        self._nz = value

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, value):
        if value is None:
            raise ValueError("Lattice type must be provided.")
        if self.nz == 0 and value.name not in ['D2Q9']:
            raise ValueError("For 2D simulations, lattice type must be LatticeD2Q9.")
        if self.nz != 0 and value.name not in ['D3Q19', 'D3Q27']:
            raise ValueError("For 3D simulations, lattice type must be LatticeD3Q19, or LatticeD3Q27.")
        self._lattice = value

    @property
    def print_info_rate(self):
        return self._printInfoRate

    @print_info_rate.setter
    def print_info_rate(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("printInfoRate must be a non-negative integer")
        self._printInfoRate = value

    @property
    def save_info_rate(self):
        return self._save_info_rate

    @save_info_rate.setter
    def save_info_rate(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("saveInfoRate must be a non-negative integer")
        self._save_info_rate = value

