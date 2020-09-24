import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def get_least_squares_solution(x, y):
    x_m, y_m, x_2_m = np.mean(x), np.mean(y), np.mean(x * x)
    b = (np.mean(x * y) - x_m * y_m) / (x_2_m - np.power(x_m, 2))
    return np.array([y_m - x_m * b, b])


def get_least_abs_solution(x, y):
    return minimize(lambda v: np.sum(np.abs(y - v[0] - v[1] * x)), np.array([0., 0.]), method='Nelder-Mead')


def print_solution(x, y, true, lsm, lad, name):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x, y, c='black', label='Выборка')
    ax.plot(x, true[0] + true[1] * x, c='r', label='Эталон.\nзавис-ть')
    ax.plot(x, lsm[0] + lsm[1] * x, c='g', label='МНК')
    ax.plot(x, lad[0] + lad[1] * x, c='b', label='МНМ')
    plt.legend(bbox_to_anchor=(1.00, 0.5), loc="center left", borderaxespad=0)
    fig.savefig('regression/' + name + '.png')


def write_coeffs(lsm_np, lad_np, lsm_p, lad_p):
    with open('regression/result.txt', 'w', encoding='utf8') as file:
        file.write("Без возмущений:\n")
        file.write("    МНК: a = {:.2f} b = {:.2f}\n".format(np.around(lsm_np[0], 2), np.around(lsm_np[1], 2)))
        file.write("    МНМ: a = {:.2f} b = {:.2f}\n\n".format(np.around(lad_np[0], 2), np.around(lad_np[1], 2)))
        file.write("С возмущениями:\n")
        file.write("    МНК: a = {:.2f} b = {:.2f}\n".format(np.around(lsm_p[0], 2), np.around(lsm_p[1], 2)))
        file.write("    МНМ: a = {:.2f} b = {:.2f}\n".format(np.around(lad_p[0], 2), np.around(lad_p[1], 2)))


def lab_6():
    x = np.linspace(-1.8, 2.0, 20)
    eps = np.random.normal(0., 1., 20)
    y_np = 2. + 2. * x + eps
    y_p = 2. + 2. * x + eps

    lsm_np = get_least_squares_solution(x, y_np)
    lad_np = get_least_abs_solution(x, y_np).x
    print_solution(x, y_np, np.array([2., 2.]), lsm_np, lad_np, 'NotPerturbated')

    y_p[0] += 10.
    y_p[-1] -= 10.

    lsm_p = get_least_squares_solution(x, y_p)
    lad_p = get_least_abs_solution(x, y_p).x
    print_solution(x, y_p, np.array([2., 2.]), lsm_p, lad_p, 'Perturbated')

    write_coeffs(lsm_np, lad_np, lsm_p, lad_p)