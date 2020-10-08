import os
import json
import matplotlib.pyplot as plt


dft_energy = []
gap_energy = []
gap_time = []

gap_per_atom = []
gap_per_atom_err = []

n_atoms = []

with open('AlNi_complete.json', 'r') as f:
    structures = json.load(f)

energy_gap = {'ce': {'calc': [], 'pred': []}}

for i in range(len(structures)):
    tmp = ''
    for char in structures[i]['xyz']:
        if char == '\n':
            break
        else:
            tmp += char
    if int(tmp) < 2:
        print('Structure {} is single-atom, skipping'.format(i))
        continue
    with open(os.path.join('lammps_gap', str(i), 'log.lammps')) as f:
        found = False
        for line in f:
            if 'Loop time of' in line:
                gap_time.append(float(line.split()[3]))
                n_atoms.append(float(line.split()[11]))
                found = True
        if not found:
            print('RUN {} needs checking'.format(i))
    with open(os.path.join('lammps_gap', str(i), 'energies')) as f:
        ens = json.load(f)
        gap_energy.append(float(ens['lammps']))
        dft_energy.append(float(ens['dft']))

for i in range(len(gap_energy)):
    energy_gap['ce']['pred'].append(gap_energy[i] / n_atoms[i])
    energy_gap['ce']['calc'].append(dft_energy[i] / n_atoms[i])
    gap_per_atom.append(gap_energy[i] / n_atoms[i])
    gap_per_atom_err.append(abs(abs(gap_energy[i] / n_atoms[i]) - abs(dft_energy[i] / n_atoms[i])))

# print(len(energy_gap['ce']['pred']))

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

plt.scatter(energy_gap['ce']['calc'], energy_gap['ce']['pred'], marker='.', color='navy', label=None, s=2)

plt.xlabel(r'Computed Energy [eV/atom]', fontsize=14, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=14, color='k')

plt.scatter(5000, 5000, marker='.', color='k', label=r'AlNi GAP', facecolor='w', s=25)

plt.plot([-40000, 0], [-40000, 0], '-', color='k', linewidth=0.25)

plt.text(-30000.0, -10000.6, r'Max GAP error: {} eV/atom'.format(round(max(gap_per_atom_err), 3)), fontsize=8)
plt.text(-31000.0, -12000.8, r'Mean GAP error: {} meV/atom'.format(
    round(sum(gap_per_atom_err)/len(gap_per_atom_err)*1000, 1)), fontsize=8)

plt.legend(loc='upper left')

plt.xlim(-40000, 0)
plt.ylim(-40000, 0)

plt.tight_layout()
plt.savefig('GAP_vs_DFT.png', dpi=300)
plt.close()
