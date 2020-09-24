import numpy as np
from scipy.stats import spearmanr, pearsonr
from matplotlib.patches import Ellipse, transforms
import matplotlib.pyplot as plt


def get_multi_normal_sample(correlation, size):
    return np.random.multivariate_normal([0., 0.], [[1., correlation], [correlation, 1.]], size=size)


def get_mixture_sample(size):
    return 0.9 * np.random.multivariate_normal([0., 0.], [[1., 0.9], [0.9, 1.]], size=size) + \
           0.1 * np.random.multivariate_normal([0., 0.], [[10., -9.], [-9., 10.]], size=size)


def quadrantr(x, y):
    med_x, med_y = np.median(x), np.median(y)

    return np.sum([np.sign(x_i - med_x) * np.sign(y_i - med_y) for x_i, y_i in zip(x, y)]) / len(x)


def correlation_to_str(correlation):
    eps = 1e-13

    if abs(correlation - 0.) < eps:
        return '$\\rho = 0$'
    if abs(correlation - 0.5) < eps:
        return '$\\rho = 0.5$'
    if abs(correlation - 0.9) < eps:
        return '$\\rho = 0.9$'


def get_data_row(data):
    return "{:.5f} & {:.5f} & {:.5f}".format(data['r_p'], data['r_s'], data['r_q'])


def print_table_normal(data, size, correlations):
    with open('tables/Normal' + str(size) + '.tex', 'w', encoding='utf8') as file:
        file.write('\\begin{tabular}{|c||c|c|c|}\n')
        file.write('\\hline\n')
        file.write('Normal, $n = ' + str(size) + '$ & $r$ & $r_S$ & $r_Q$  \\\\ \n')
        file.write('\\hline \\hline \n')
        for correlation in correlations:
            file.write('$E(z)$, ' + correlation_to_str(correlation) + ' & ' + get_data_row(data[correlation]['mean']) +
                       '\\\\ \n')
            file.write('\\hline\n')
            file.write('$D(z)$, ' + correlation_to_str(correlation) + ' & ' + get_data_row(data[correlation]['var']) +
                       '\\\\ \n')
            if not correlation == correlations[-1]:
                file.write('\\hline')
            file.write('\\hline\n')
        file.write('\\end{tabular}\n')
        file.write('\\caption{Двумерное нормальное распределение, $n = ' + str(size) + '$}\n')


def calculate_coeffs_normal(sizes, correlations):
    for size in sizes:
        data = {}
        for correlation in correlations:
            r_p, r_s, r_q = [], [], []
            for _ in range(1000):
                sample = get_multi_normal_sample(correlation, size)
                r_p.append(pearsonr(sample[:, 0], sample[:, 1])[0])
                r_q.append(quadrantr(sample[:, 0], sample[:, 1]))
                r_s.append(spearmanr(sample[:, 0], sample[0:, 1])[0])
            data[correlation] = {'mean': {'r_p': np.around(np.mean(r_p), 5),
                                          'r_s': np.around(np.mean(r_s), 5),
                                          'r_q': np.around(np.mean(r_q), 5)},
                                 'var': {'r_p': np.around(np.var(r_p), 5),
                                         'r_s': np.around(np.var(r_s), 5),
                                         'r_q': np.around(np.var(r_q), 5)}}

        print_table_normal(data, size, correlations)


def print_table_mixture(data, sizes):
    with open('tables/Mixture.tex', 'w', encoding='utf8') as file:
        file.write('\\begin{tabular}{|c||c|c|c|}\n')
        file.write('\\hline\n')
        file.write('Mixture & $r$ & $r_S$ & $r_Q$  \\\\ \n')
        file.write('\\hline \\hline \n')
        for size in sizes:
            file.write('$E(z)$, $n = ' + str(size) + '$ & ' + get_data_row(data[size]['mean']) + '\\\\ \n')
            file.write('\\hline\n')
            file.write('$D(z)$, $n = ' + str(size) + '$ & ' + get_data_row(data[size]['var']) + '\\\\ \n')
            if not size == sizes[-1]:
                file.write('\\hline')
            file.write('\\hline\n')
        file.write('\\end{tabular}\n')
        file.write('\\caption{Смесь распредлений}\n')


def calculate_coeffs_mixture(sizes):
    data = {}
    for size in sizes:
        r_p, r_s, r_q = [], [], []
        for _ in range(1000):
            sample = get_mixture_sample(size)
            r_p.append(pearsonr(sample[:, 0], sample[:, 1])[0])
            r_q.append(quadrantr(sample[:, 0], sample[:, 1]))
            r_s.append(spearmanr(sample[:, 0], sample[0:, 1])[0])

        data[size] = {'mean': {'r_p': np.around(np.mean(r_p), 5),
                               'r_s': np.around(np.mean(r_s), 5),
                               'r_q': np.around(np.mean(r_q), 5)},
                      'var': {'r_p': np.around(np.var(r_p), 5),
                              'r_s': np.around(np.var(r_s), 5),
                              'r_q': np.around(np.var(r_q), 5)}}
    print_table_mixture(data, sizes)


def confidence_ellipse(x, y, ax, n_std=2.5, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def create_ellipse(correlations, sizes):
    for size in sizes:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        fig.suptitle("Normal, " + str(size))
        for ax, correlation in zip(axes, correlations):
            sample = get_multi_normal_sample(correlation, size)
            ax.scatter(sample[:, 0], sample[:, 1], s=0.6, c='k')

            ax.axvline(c='blue', lw=1)
            ax.axhline(c='blue', lw=1)
            ax.set_xlim(np.min(sample[:, 0]) - 3, np.max(sample[:, 0]) + 3)
            ax.set_ylim(np.min(sample[:, 1]) - 3, np.max(sample[:, 1]) + 3)
            confidence_ellipse(sample[:, 0], sample[:, 1], ax, edgecolor='green')
            ax.set_title(correlation_to_str(correlation))
        fig.savefig('ellipses/Normal' + str(size) + '.png')


def lab_5():
    sizes = [20, 60, 100]
    correlations = [0., 0.5, 0.9]
  #  calculate_coeffs_normal(sizes, correlations)
  #  calculate_coeffs_mixture(sizes)
    create_ellipse(correlations, sizes)
