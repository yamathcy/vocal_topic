import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import ELBO
from collections import Counter

def model(counts):
    # 事前分布からトピック比率をサンプリング
    theta = pyro.sample("theta", dist.Dirichlet(0.5 * torch.ones(len(counts))))