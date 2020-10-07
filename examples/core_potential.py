import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, elementary_charge


def f(in_val):
    # needs fitting
    return in_val


def phi(in_val):
    # this screening function needs fitting
    return in_val


def a(z1, z2):
    return 0.46848 / (z1**0.23 + z2**0.23)


def pair_potential(in_grid, z1, z2):
    values = []
    for g in in_grid:
        values.append(1 / (4 * np.pi * epsilon_0) * z1 * z2 * elementary_charge**2 / g * phi(g/a(z1, z2)) * f(g))
    return np.array(values)


grid = np.linspace(start=0.005, stop=1, num=20, endpoint=True)
vals = pair_potential(in_grid=grid, z1=1, z2=1)

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', serif='Palatino')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3

plt.plot(grid, vals)

plt.show()
