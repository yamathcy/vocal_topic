import numpy as np
import pyro
import pyro.distributions as dist

from collections import Counter

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components

    def sample