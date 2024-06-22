import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
from collections import Counter

def model(counts: torch.Tensor):
    # 事前分布(一様分布)からトピック比率θをサンプリング
    theta = pyro.sample("theta", dist.Dirichlet(0.5 * torch.ones(len(counts))))
    total_count = int(sum(counts))

    # トピック比率θに基づいて、各トピックから単語をサンプリング
    pyro.sample("counts", dist.Multinomial(total_count, theta), obs=counts)

data = torch.tensor([5, 4, 2, 5, 6, 5, 3, 3, 1, 5, 5, 3, 5, 3, 5, \
                     3, 5, 5, 3, 5, 5, 3, 1, 5, 3, 3, 6, 5, 5, 6])
counts = torch.unique(data, return_counts=True)[1].float()

nuts_kernel = NUTS(model)
num_samples, warmup_steps = (1000, 200) if not smoke_test else (10, 10)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(counts)
hmc_samples = {k: v.detach().cpu().numpy()
               for k, v in mcmc.get_samples().items()}
