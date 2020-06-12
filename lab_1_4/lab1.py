from matplotlib import pyplot as plt

import seaborn as sns
from DistHandler import DistHandler


def main():
    sample_sizes = [10, 50, 1000]
    dist_names = DistHandler.distributions.keys()

    for dist_name in dist_names:
        fig, axs = plt.subplots(1, len(sample_sizes), sharey='row', figsize=(14, 6)) if \
                        dist_name is not 'Cauchy' else plt.subplots(1, 3, figsize=(14, 6))
        for idx, sample_size in enumerate(sample_sizes):
            sample = DistHandler.get_sample(dist_name, sample_size)
            sns.distplot(sample, kde=False, norm_hist=True, ax=axs[idx])
            sns.lineplot(*DistHandler.get_pdf_data(sample, dist_name), ax=axs[idx])

            axs[idx].set_title(f'{dist_name}, sample size = {sample_size}')
            axs[idx].set(xlabel='x', ylabel='Density')
        plt.show()
        fig.savefig(f'plots/lab1/{dist_name}.png')


if __name__ == '__main__':
    main()










