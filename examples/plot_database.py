import os
import json
import matplotlib.pyplot as plt


dft_energy = []
eam_energy = []
gap_energy = []
eam_time = []
gap_time = []

eam_per_atom = []
gap_per_atom = []
eam_per_atom_err = []
gap_per_atom_err = []

n_atoms = []

with open('systems.json', 'r') as f:
    system = json.load(f)

energy_eam = {'fcc': {'calc': [], 'pred': []},
              'bcc': {'calc': [], 'pred': []},
              'sc': {'calc': [], 'pred': []},
              'hcp': {'calc': [], 'pred': []},
              'slab': {'calc': [], 'pred': []},
              'cluster': {'calc': [], 'pred': []},
              'addition': {'calc': [], 'pred': []}}

energy_gap = {'fcc': {'calc': [], 'pred': []},
              'bcc': {'calc': [], 'pred': []},
              'sc': {'calc': [], 'pred': []},
              'hcp': {'calc': [], 'pred': []},
              'slab': {'calc': [], 'pred': []},
              'cluster': {'calc': [], 'pred': []},
              'addition': {'calc': [], 'pred': []}}

energy_3bd = {'fcc': {'calc': [], 'pred': []},
              'bcc': {'calc': [], 'pred': []},
              'sc': {'calc': [], 'pred': []},
              'hcp': {'calc': [], 'pred': []},
              'slab': {'calc': [], 'pred': []},
              'cluster': {'calc': [], 'pred': []},
              'addition': {'calc': [], 'pred': []}}

color = {'fcc': 'green',
         'bcc': 'navy',
         'sc': 'm',
         'hcp': 'y',
         'slab': 'orange',
         'cluster': 'yellow',
         'addition': 'blue'}

for i in range(27291):
    with open(os.path.join('lammps_eam', str(i), 'LAMMPS_energy')) as f:
        eam_energy.append(float(f.readline()))
    with open(os.path.join('lammps_eam', str(i), 'DFT_energy')) as f:
        dft_energy.append(float(f.readline()))
    with open(os.path.join('lammps_gap', str(i), 'LAMMPS_energy')) as f:
        gap_energy.append(float(f.readline()))
    with open(os.path.join('lammps_eam', str(i), 'log.lammps')) as f:
        for line in f:
            if 'Loop time of' in line:
                eam_time.append(float(line.split()[3]))
    with open(os.path.join('lammps_gap', str(i), 'log.lammps')) as f:
        for line in f:
            if 'Loop time of' in line:
                gap_time.append(float(line.split()[3]))
                n_atoms.append(float(line.split()[11]))

for i in range(len(n_atoms)):
    energy_eam[system[i]]['calc'].append(dft_energy[i] / n_atoms[i])
    energy_eam[system[i]]['pred'].append(eam_energy[i] / n_atoms[i])
    energy_gap[system[i]]['pred'].append(gap_energy[i] / n_atoms[i])
    eam_per_atom.append(eam_energy[i] / n_atoms[i])
    gap_per_atom.append(gap_energy[i] / n_atoms[i])
    eam_per_atom_err.append(abs(eam_energy[i] / n_atoms[i] - gap_energy[i] / n_atoms[i]))
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

for sys in ['fcc', 'bcc', 'sc', 'hcp', 'slab', 'cluster', 'addition']:
    plt.scatter(energy_eam[sys]['calc'], energy_eam[sys]['pred'],
                marker='v', color=color[sys], label=None, s=2)
    plt.scatter(energy_eam[sys]['calc'], energy_gap[sys]['pred'],
                marker='^', color=color[sys], label=None, s=2)

plt.xlabel(r'Computed Energy [eV/atom]', fontsize=17, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=17, color='k')

plt.scatter(5, 5, marker='v', color='k', label=r'GAP 2b+3b+soap', facecolor='w', s=25)
plt.scatter(5, 5, marker='^', color='k', label=r'GAP 2b+soap', facecolor='w', s=25)

plt.plot([-6.5, 0.5], [-6.5, 0.5], '-', color='k', linewidth=0.25)

for i, sys in enumerate(['fcc', 'bcc', 'sc', 'hcp', 'slab', 'cluster', 'addition']):
    plt.text(-6.5, -1-(0.2*i), sys, color=color[sys], fontsize=8)

plt.text(-3, -4.2, r'Avg GAP 2b+3b+soap time: {}s'.format(round(sum(eam_time)/len(eam_time), 5)), fontsize=8)
plt.text(-3, -4.4, r'Avg GAP 2b+soap time: {}s'.format(round(sum(gap_time)/len(gap_time), 5)), fontsize=8)

plt.text(-5.0, -1.4, r'Max 2b/3b error: {} eV/atom'.format(round(max(eam_per_atom_err), 3)), fontsize=8)
plt.text(-5.0, -1.6, r'Mean 2b/3b error: {} meV/atom'.format(
     round(sum(eam_per_atom_err)/len(eam_per_atom_err)*1000, 1)), fontsize=8)

plt.text(-5.0, -0.6, r'Max GAP error: {} eV/atom'.format(round(max(gap_per_atom_err), 3)), fontsize=8)
plt.text(-5.0, -0.8, r'Mean GAP error: {} meV/atom'.format(
     round(sum(gap_per_atom_err)/len(gap_per_atom_err)*1000, 1)), fontsize=8)

plt.legend(loc='upper left')

plt.xlim(-7, 1)
plt.ylim(-7, 1)

plt.tight_layout()
plt.savefig('eam_vs_gap.png', dpi=300)
plt.close()
