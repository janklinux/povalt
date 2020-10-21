import os
import json
import matplotlib.pyplot as plt
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor


dft_energy = []
gap_energy = []
gap_time = []

gap_per_atom = []
gap_per_atom_err = []

n_atoms = []

with open('CuPtIr_complete.json', 'r') as f:
    structures = json.load(f)

energy_gap = {'ce': {'calc': [], 'pred': []}}

for i in range(len(structures)):
    tmp = ''
    lines = []
    for char in structures[i]['xyz']:
        if '\n' in char:
            lines.append(tmp)
            tmp = ''
        else:
            tmp += char

    if int(lines[0]) < 3:  # need three atoms at least
        continue

    atoms = AseAtomsAdaptor().get_atoms(Structure.from_dict(structures[i]['structure']))

    tmp = []
    for a in atoms.get_chemical_symbols():
        if a not in tmp:
            tmp.append(a)

    if len(tmp) != 3:  # need all three different species for GAP
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
            continue

    with open(os.path.join('lammps_gap', str(i), 'energies')) as f:
        ens = json.load(f)
        gap_energy.append(float(ens['lammps']))
        dft_energy.append(float(ens['dft']))

for i in range(len(gap_energy)):
    energy_gap['ce']['pred'].append(gap_energy[i] / n_atoms[i])
    energy_gap['ce']['calc'].append(dft_energy[i] / n_atoms[i])
    gap_per_atom.append(gap_energy[i] / n_atoms[i])
    gap_per_atom_err.append(abs(abs(gap_energy[i] / n_atoms[i]) - abs(dft_energy[i] / n_atoms[i])))


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

plt.scatter(50000, 50000, marker='.', color='k', label=r'Cu--Ir--Pt GAP', facecolor='w', s=25)

plt.plot([-490000, -500], [-490000, -500], '-', color='k', linewidth=0.25)

plt.text(-305000.0, -50000.6, r'Max GAP error: {} eV/atom'.format(round(max(gap_per_atom_err), 3)), fontsize=14)
plt.text(-310000.0, -64000.8, r'Mean GAP error: {} meV/atom'.format(
    round(sum(gap_per_atom_err)/len(gap_per_atom_err)*1000, 1)), fontsize=14)

plt.legend(loc='upper left')

plt.xlim(-500000, 0)
plt.ylim(-500000, 0)

plt.show()
plt.tight_layout()
plt.savefig('GAP_vs_DFT.png', dpi=300)
plt.close()
