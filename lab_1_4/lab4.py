from DistHandler import DistHandler
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as ss


def get_ecdf_data(sample, linspace_params, n_splits=1000):
    def ecdf(x):
        return len(sample[sample < x]) / len(sample)
    x = np.linspace(*linspace_params, n_splits)
    y = np.array(list(map(ecdf, x)))
    return x, y


def record_cdf_vs_ecdf():
    sample_sizes = [20, 60, 100]
    dist_names = DistHandler.distributions.keys()

    for dist_name in dist_names:
        fig, axs = plt.subplots(1, len(sample_sizes), figsize=(12, 5))
        for idx, sample_size in enumerate(sample_sizes):
            sample = DistHandler.get_sample(dist_name, sample_size, with_random_seed=True)
            cont_linspace_params = (-4, 4)
            disc_linspace_params = (6, 14)
            if dist_name == 'Poisson':
                sns.lineplot(*DistHandler.get_cdf_data(sample, dist_name, disc_linspace_params), ax=axs[idx])
                sns.lineplot(*get_ecdf_data(sample, disc_linspace_params), ax=axs[idx])
            else:
                sns.lineplot(*DistHandler.get_cdf_data(sample, dist_name, cont_linspace_params), ax=axs[idx])
                sns.lineplot(*get_ecdf_data(sample, cont_linspace_params), ax=axs[idx])

            axs[idx].set_title(f'{dist_name}, n = {sample_size}')
            axs[idx].set(xlabel='x', ylabel='F(x)')
            axs[idx].legend(labels=['cdf', 'ecdf'])
        fig.savefig(f'plots/lab4/cdf_vs_ecdf/{dist_name}.png')
        plt.show()


def get_title_by_factor(factor):
    eps = 1e-14
    if abs(factor - 0.5) < eps:
        return '$\\frac{h^S_n}{2}$'
    if abs(factor - 1.) < eps:
        return '$h^S_n$'
    if abs(factor - 2.) < eps:
        return '$2h^S_n$'


def kernel(x):
    return 1./np.sqrt(2. * np.pi) * np.exp(-x**2 / 2.)


def get_kde(sample, h, x):
    return 1./(len(sample) * h) * np.sum([kernel((x - x_i) / h) for x_i in sample])


def record_pdf_kde():
    sample_sizes = [20, 60, 100]
    dist_names = DistHandler.distributions.keys()
    factors = [0.5, 1., 2.]

    for dist_name in dist_names:
        for sample_size in sample_sizes:
            sample = DistHandler.get_sample(dist_name, sample_size)
            fig, axes = plt.subplots(1, len(factors), figsize=(14, 5))
            x_cont = np.linspace(6, 14, 1000) if dist_name is 'Poisson' else np.linspace(-4, 4, 1000)
            bounds = (left, right) = (6, 14) if dist_name is 'Poisson' else (-4, 4)
            x, y_pdf = DistHandler.get_pdf_data(sample, dist_name, bounds=bounds)
            for factor, ax in zip(factors, axes):
                trimmed = sample[sample <= right]
                trimmed = trimmed[left <= trimmed]

                h_s = np.power(4. / 3., 1. / 5.) * np.std(trimmed) * np.power(len(trimmed), -1. / 5.)
                sns.lineplot(x, y_pdf, ax=ax)

                if dist_name is 'Poisson':
                    y_kde_cont = np.array([get_kde(trimmed, h_s * factor, elem) for elem in x_cont])
                    sns.lineplot(x_cont, y_kde_cont, ax=ax)
                else:
                    y_kde = np.array([get_kde(trimmed, h_s * factor, elem) for elem in x])
                    sns.lineplot(x, y_kde, ax=ax)
                ax.set_title(get_title_by_factor(factor))
                ax.set(xlabel='x', ylabel='f(x)')
                ax.legend(labels=['pdf', 'kde of pdf'])
            fig.suptitle(dist_name + ' ' + str(sample_size))
            # plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            # fig.savefig('kde/' + dist_name + str(sample_size) + '.png')
            fig.savefig(f'plots/lab4/kde/{dist_name}{str(sample_size)}.png')
            plt.show()


def main():
    # record_cdf_vs_ecdf()
    record_pdf_kde()
    pass


if __name__ == '__main__':
    main()
