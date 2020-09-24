import numpy as np
import scipy.stats as ss
from math import sqrt, factorial


RANDOM_SEED = 1812


class DistHandler:
    """
    Class handles five distributions:
        Normal(0, 1),
        Cauchy(0, 1),
        Laplace(0, 1/sqrt(2)),
        Poisson(10),
        Uniform(-sqrt(3), sqrt(3))
    """
    distributions = {
            'Normal': ss.norm(loc=0, scale=1),
            'Cauchy': ss.cauchy(loc=0, scale=1),
            'Laplace': ss.laplace(loc=0, scale=1. / sqrt(2.)),
            'Poisson': ss.poisson(mu=10),
            'Uniform': ss.uniform(loc=-sqrt(3.), scale=2 * sqrt(3.))
        }

    @staticmethod
    def get_sample(dist_name, n, with_random_seed=False):
        dist = DistHandler.distributions[dist_name]
        if with_random_seed:
            return dist.rvs(n, random_state=RANDOM_SEED)
        return dist.rvs(n)

    @staticmethod
    def get_pdf_data(sample, dist_name, bounds=None, n_splits=1000):
        dist = DistHandler.distributions[dist_name]

        sample_min = sample.min()
        sample_max = sample.max()

        if dist_name == 'Poisson':
            x = np.arange(sample_max + 1) if not bounds else np.arange(*bounds)
            y = dist.pmf(x)

        else:
            x = np.linspace(sample_min, sample_max, n_splits) if not bounds else np.linspace(*bounds, n_splits)
            y = dist.pdf(x)

        return x, y

    @staticmethod
    def get_cdf_data(sample, dist_name, linspace_params, n_splits=1000):
        dist = DistHandler.distributions[dist_name]

        if dist_name == 'Poisson':
            x = np.linspace(*linspace_params, n_splits)
            y = dist.cdf(x)

        else:
            x = np.linspace(*linspace_params, n_splits)
            y = dist.cdf(x)

        return x, y
