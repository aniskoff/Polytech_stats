import scipy.stats
import numpy as np


def print_results(mu, sigma, limits, n_v, p_v, sample_size, filename):
    with open('chi/coeffs_' + filename, 'w') as file:
        file.write('mu={:.2f}  sigma={:.2f}\n'.format(np.around(mu, 2), np.around(sigma, 2)))
        file.write('limits          n_i  p_i  np_i  n_i - np_i  frac\n')
        for i in range(len(n_v)):
            if i == 0:
                file.write('(-inf; {:.2f}]  '.format(np.around(limits[0], 2)))
            elif i == len(n_v) - 1:
                file.write('({:.2f}; +inf)  '.format(np.around(limits[-1], 2)))
            else:
                file.write('({:.2f}; {:.2f}]  '.format(np.around(limits[i - 1], 2), np.around(limits[i], 2)))
            file.write(str(n_v[i]) + '  ')
            file.write('{:.4f}  '.format(np.around(p_v[i], 4)))
            file.write('{:.2f}  '.format(np.around(p_v[i] * sample_size, 2)))
            file.write('{:.2f}  '.format(np.around(n_v[i] - sample_size * p_v[i], 2)))
            val = np.power(n_v[i] - sample_size * p_v[i], 2) / (sample_size * p_v[i])
            file.write('{:.2f}  \n'.format(np.around(val, 2)))


def calculate_coeffs(sample, filename):
    k = int(1.72 * np.power(len(sample), 1. / 3.))
    left, right = -1.1, 1.1
    mu, sigma = np.mean(sample), np.std(sample)
    limits = np.linspace(left, right, k - 1)
    n_v, p_v = [], []
    for i in range(k):
        if i == 0:
            n_cur = len([elem for elem in sample if elem <= limits[0]])
            p_cur = scipy.stats.norm.cdf(limits[0], loc=mu, scale=sigma)
        elif i == k - 1:
            n_cur = len([elem for elem in sample if elem > limits[-1]])
            p_cur = 1 - scipy.stats.norm.cdf(limits[-1], loc=mu, scale=sigma)
        else:
            n_cur = len([elem for elem in sample if limits[i - 1] < elem <= limits[i]])
            p_cur = scipy.stats.norm.cdf(limits[i], loc=mu, scale=sigma) - scipy.stats.norm.cdf(limits[i - 1], loc=mu, scale=sigma)
        n_v.append(n_cur)
        p_v.append(p_cur)
    print_results(mu, sigma, limits, n_v, p_v, len(sample), filename)


def lab_7():
    sample = np.random.normal(loc=0., scale=1., size=100)
    #sample = np.random.uniform(low=-2., high=2., size=30)
    #sample = np.random.standard_cauchy(size=30)
    calculate_coeffs(sample, 'normal')

