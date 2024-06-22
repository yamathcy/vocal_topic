import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import ELBO
from collections import Counter

def model(counts: torch.Tensor):
    # 事前分布(一様分布)からトピック比率θをサンプリング
    theta = pyro.sample("theta", dist.Dirichlet(0.5 * torch.ones(len(counts))))
    total_count = int(sum(counts))
    
    # トピック比率θに基づいて、各トピックから単語をサンプリング
    pyro.sample("counts", dist.Multinomial(total_count, theta), obs=counts)

