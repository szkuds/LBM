import matplotlib.pyplot as plt
from main import LBM


class plotting(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
