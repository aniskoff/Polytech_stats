import numpy as np
import scipy.stats


def calculate_classic_mean(sample, gamma):
    n = len(sample)
    mean, s = np.mean(sample), np.std(sample)
    rng = s * scipy.stats.t.ppf((1. + gamma) / 2., n - 1) / np.power(n - 1, 0.5)
    return mean - rng, mean + rng


def calculate_classic_std(sample, gamma):
    n = len(sample)
    s = np.std(sample)

    return s * np.power(n, 0.5) / np.power(scipy.stats.chi2.ppf((1. + gamma) / 2., n - 1), 0.5), \
           s * np.power(n, 0.5) / np.power(scipy.stats.chi2.ppf((1. - gamma) / 2., n - 1), 0.5)


def calculate_asympt_mean(sample, gamma):
    n = len(sample)
    mean, s = np.mean(sample), np.std(sample)
    rng = s * scipy.stats.norm.ppf((1. + gamma) / 2.) / np.power(n, 0.5)
    return mean - rng, mean + rng


def calculate_asympt_std(sample, gamma):
    n = len(sample)
    s = np.std(sample)
    e = scipy.stats.kurtosis(sample, fisher=True)
    rng = 0.5 * scipy.stats.norm.ppf((1. + gamma) / 2.) * np.power((e + 2.) / n, 0.5)
    return s * (1. - rng), s * (1. + rng)


def calculate_intervals(sizes, gamma):
    for size in sizes:
        sample = np.random.normal(size=size)
        cl_mean, cl_std = calculate_classic_mean(sample, gamma), calculate_classic_std(sample, gamma)
        a_mean, a_std = calculate_asympt_mean(sample, gamma), calculate_asympt_std(sample, gamma)

        with open('intervals/int_' + str(size) + '.txt', 'w') as file:
            file.write('Classic: \n')
            file.write('   mean: {:.2f} < m < {:.2f}\n'.format(np.around(cl_mean[0], 2), np.around(cl_mean[1], 2)))
            file.write('   std: {:.2f} < sigma < {:.2f}\n'.format(np.around(cl_std[0], 2), np.around(cl_std[1], 2)))
            file.write('Asympt: \n')
            file.write('   mean: {:.2f} < m < {:.2f}\n'.format(np.around(a_mean[0], 2), np.around(a_mean[1], 2)))
            file.write('   std: {:.2f} < sigma < {:.2f}\n'.format(np.around(a_std[0], 2), np.around(a_std[1], 2)))


def lab_8():
    sizes = [20, 100]
    gamma = 0.95
    calculate_intervals(sizes, gamma)