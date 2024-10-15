import matplotlib.pyplot as plt
from main import *


class Plot(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        plt.ioff()  # Turn off interactive mode
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(jnp.zeros((1, 1)), cmap='viridis')
        plt.colorbar(self.im)
        self.ax.invert_yaxis()

    def update_plot(self, var):
        self.im.set_data(var.T)
        self.im.set_clim(var.min(), var.max())
        plt.savefig(self.save_dir)

    def rho(self, it, rho):
        self.ax.set_title(f"it: {it} sum_rho: {jnp.sum(rho):.2f}")
        self.update_plot(rho)
        plt.clf()
