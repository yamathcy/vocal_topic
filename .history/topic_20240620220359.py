import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import ELBO
from collections import Counter

def model(counts):
    