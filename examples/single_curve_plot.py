import os
import numpy as np
import matplotlib.pyplot as plt


grid = np.linspace(start=0.3, stop=1, num=5, endpoint=False)
grid = np.append(grid, np.linspace(start=1, stop=3.5, num=30, endpoint=False))
grid = np.append(grid, np.linspace(start=3.5, stop=8, num=15, endpoint=True))

energy = []

for g in grid:
    with open(os.path.join(str(np.round(g, 2)), 'run.light'), 'r') as f:
        for line in f:
            if '| Total energy of the DFT / Hartree-Fock s.c.f. calculation      :' in line:
                energy.append(float(line.split()[11]))

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
plt.plot(grid, energy, '.-', color='navy')
plt.xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
plt.ylabel(r'Free Energy [eV]', fontsize=22, color='k')
plt.title(r'Pt -- Pt', fontsize=16)
plt.xlim(1.5, 4.5)
plt.ylim(np.amin(energy)-1, np.amin(energy)+10)
plt.tight_layout()
plt.savefig('dimer.png', dpi=300)
plt.close()
