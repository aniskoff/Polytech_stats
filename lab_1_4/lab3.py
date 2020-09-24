from DistHandler import DistHandler
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def record_experiment_outliers(dist_names, sample_sizes, num_experiments=1000):
    with open('tex_files/lab3/experiment_table.tex', 'w', encoding='utf8') as file:
        file.write('\\begin{tabular}{|c|c|c|}\n')
        file.write('\\hline\n')
        file.write('Выборка & Доля выбросов & Дисперсия доли выбросов \\\\ \n')
        file.write('\\hline \\hline \n')
        for dist_name in dist_names:
            for sample_size in sample_sizes:
                outliers_rates = []
                for _ in range(num_experiments):
                    sample = DistHandler.get_sample(dist_name, sample_size, with_random_seed=False)
                    outliers_rate = float(len(matplotlib.cbook.boxplot_stats(sample)[0]["fliers"])) / len(sample)
                    outliers_rates.append(outliers_rate)

                mean_outliers_rate = np.mean(outliers_rates)
                var_outliers_rate = np.var(outliers_rates)
                file.write(dist_name + ' ' + str(sample_size) +
                           ' & {:.3f}'.format(np.around(mean_outliers_rate, 3)) +
                           ' & {:.3f}'.format(np.around(var_outliers_rate, 3)) + ' \\\\ \n')
                file.write('\\hline \n')
        file.write('\\end{tabular}\n')
        file.write('\\caption{Экспериментальная доля выбросов}\n')


def main():
    sample_sizes = [20, 100]
    dist_names = DistHandler.distributions.keys()
    # for dist_name in dist_names:
    #     fig, axs = plt.subplots(1, len(sample_sizes))
    #     for idx, sample_size in enumerate(sample_sizes):
    #         sample = DistHandler.get_sample(dist_name, sample_size, with_random_seed=True)
    #         sns.boxplot(sample, ax=axs[idx])
    #         axs[idx].set_title(f'{dist_name}, n={sample_size}')
    #         print(f'{dist_name}, n={sample_size}, {matplotlib.cbook.boxplot_stats(sample)[0]["fliers"]}')
    #     fig.savefig(f'plots/lab3/{dist_name}.png')

    record_experiment_outliers(dist_names, sample_sizes)


if __name__ == '__main__':
    main()
