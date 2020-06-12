from DistHandler import DistHandler
import numpy as np
import scipy.stats as ss
from itertools import product
from copy import deepcopy


def get_data_row(data, stat_type):
    return "{:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f}".format(data['mean'][stat_type], data['median'][stat_type],
                                                               data['z_R'][stat_type], data['z_Q'][stat_type],
                                                               data['z_tr'][stat_type])


def get_rus_caption(dist_name):
    if dist_name is 'Normal':
        return 'Нормальное распределение'
    if dist_name is 'Cauchy':
        return 'Распределение Коши'
    if dist_name is 'Laplace':
        return 'Распределение Лапласа'
    if dist_name is 'Poisson':
        return 'Распределение Пуассона'
    if dist_name is 'Uniform':
        return 'Равномерное распределение'


def record_measurements(table, sample_sizes):
    dist_names = table.keys()
    for dist_name in dist_names:
        with open('tex_files/lab2/' + dist_name + '.tex', 'w', encoding='utf8') as file:
            file.write('\\begin{tabular}{|c|||c|c|c|c|c|}\n')
            file.write('\\hline\n')
            file.write(dist_name + ' & $\\overline{x}$ & $med \\, x$ & $z_R$ & $z_Q$ & $z_{tr}$ \\\\ \n')
            file.write('\\hline \\hline \\hline \n')
            data = table[dist_name]
            for sample_size in sample_sizes:
                file.write('$E(z)$ ' + str(sample_size) + ' & ' +
                           get_data_row(data[sample_size], 'mean') + '\\\\ \n')
                file.write('\\hline\n')
                file.write('$D(z)$ ' + str(sample_size) + ' & ' + get_data_row(data[sample_size], 'var') + '\\\\ \n')
                if not sample_size == sample_sizes[-1]:
                    file.write('\\hline')
                file.write('\\hline\n')
            file.write('\\end{tabular}\n')
            file.write('\\caption{' + get_rus_caption(dist_name) + '}\n')


def main():
    sample_sizes = [10, 100, 1000]
    n_iter = 1000
    dist_names = DistHandler.distributions.keys()

    stats = {'means': [], 'medians': [], 'z_Rs': [], 'z_Qs': [], 'z_trs': []}
    stats_with_sizes = {sample_size: deepcopy(stats) for sample_size in sample_sizes}
    table = {dist_name: deepcopy(stats_with_sizes) for dist_name in dist_names}

    for dist_name, sample_size in product(dist_names, sample_sizes):
        for _ in range(n_iter):
            sample = DistHandler.get_sample(dist_name, sample_size)
            mean = np.mean(sample)
            median = np.median(sample)
            z_R = (sample.min() + sample.max()) / 2.
            q_25, q_75 = np.quantile(sample, [0.25, 0.75])
            z_Q = (q_25 + q_75) / 2.
            z_tr = ss.trim_mean(sample, 0.25)

            table[dist_name][sample_size]['means'].append(mean)
            table[dist_name][sample_size]['medians'].append(median)
            table[dist_name][sample_size]['z_Rs'].append(z_R)
            table[dist_name][sample_size]['z_Qs'].append(z_Q)
            table[dist_name][sample_size]['z_trs'].append(z_tr)

    for dist_name, sample_size, stat_name in product(dist_names, sample_sizes, stats.keys()):
        table[dist_name][sample_size][stat_name[:-1]] = {'mean': np.mean(table[dist_name][sample_size][stat_name]),
                                                         'var': np.var(table[dist_name][sample_size][stat_name])}
        del table[dist_name][sample_size][stat_name]

    record_measurements(table, sample_sizes)

    # for dist_name in dist_names:
    #     print(f'{dist_name}: ')
    #     for sample_size in sample_sizes:
    #         print(f'\tn={sample_size}: ')
    #         for stat_name in stats.keys():
    #             print(f'\t\t{stat_name}:\tMean: {np.mean(table[dist_name][sample_size][stat_name])}, Var: '
    #                   f'{np.var(table[dist_name][sample_size][stat_name])}')
    #         print()
    #     print('\n' * 2)


if __name__ == '__main__':
    main()
