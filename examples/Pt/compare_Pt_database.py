import os
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from datetime import datetime
from pymatgen.io.ase import AseAtomsAdaptor


def parse_quip(filename):
    start_parse = False
    with open(filename, 'r') as ff:
        data = []
        predicted_energies = []
        first_line_parsed = False
        for line in ff:
            if not first_line_parsed:
                start_time = line.split()[3]
                first_line_parsed = True
            if line.startswith('libAtoms::Finalise:'):
                if len(line.split()) == 3:
                    end_time = line.split()[2]
            if line.startswith('Energy='):
                if start_parse:
                    data.append(tmp)
                start_parse = True
                tmp = list()
                predicted_energies.append(float(line.split('=')[1]))
            if start_parse:
                if line.startswith('AT'):
                    tmp.append(line[3:])
    dt = datetime.strptime(end_time, '%H:%M:%S') - datetime.strptime(start_time, '%H:%M:%S')
    return dict({'data': data, 'predicted_energies': predicted_energies}), dt


# parsed_data, runtime = parse_quip('pure_gap/quip.result')

dft_energy = []
gap_energy = []
gap_time = []

gap_per_atom = []
gap_per_atom_err = []

gap_forces = []
dft_forces = []
force_err = []
check_err = []

n_atoms = []

with open('structures.json', 'r') as f:
    structures = json.load(f)
with open('systems.json', 'r') as f:
    system = json.load(f)

energy_gap = {'fcc': {'calc': [], 'pred': []},
              'bcc': {'calc': [], 'pred': []},
              'sc': {'calc': [], 'pred': []},
              'hcp': {'calc': [], 'pred': []},
              'slab': {'calc': [], 'pred': []},
              'cluster': {'calc': [], 'pred': []},
              'addition': {'calc': [], 'pred': []}}

forces_gap = {'fcc': {'calc': [], 'pred': []},
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

for i in range(len(structures)):
    with open(os.path.join('lammps_gap', str(i), 'log.lammps')) as f:
        found = False
        for line in f:
            if 'Loop time of' in line:
                gap_time.append(float(line.split()[3]))
                n_atoms.append(float(line.split()[11]))
                found = True
        if not found:
            print('No run in {}'.format(i))
            continue

    with open(os.path.join('lammps_gap', str(i), 'energies')) as f:
        ens = json.load(f)
        gap_energy.append(float(ens['lammps']))
        dft_energy.append(float(ens['dft']))

    with open(os.path.join('lammps_gap', str(i), 'final.dump')) as f:
        parse = False
        tmp = []
        for line in f:
            if parse:
                tmp.append([float(x) for x in line.split()[5:8]])
            if 'ITEM: ATOMS id type x y z fx fy fz' in line:
                parse = True
    gap_forces.append(tmp)

    with open(os.path.join('lammps_gap', str(i), 'in.xyz')) as f:
        parse = False
        tmp = []
        for i, line in enumerate(f):
            if i > 1:
                tmp.append([float(x) for x in line.split()[4:7]])
    dft_forces.append(tmp)

for i in range(len(gap_energy)):
    energy_gap[system[i]]['calc'].append(dft_energy[i] / n_atoms[i])
    energy_gap[system[i]]['pred'].append(gap_energy[i] / n_atoms[i])
    gap_per_atom.append(gap_energy[i] / n_atoms[i])
    gap_per_atom_err.append(abs(abs(gap_energy[i] / n_atoms[i]) - abs(dft_energy[i] / n_atoms[i])))
    if gap_per_atom_err[i] > 0.1:
        print(i, dft_energy[i], n_atoms[i], gap_per_atom_err[i], system[i])
        dft_energy[i] = read(os.path.join(str(i), 'OUTCAR')).get_potential_energy(force_consistent=True)
        print('reading recomputed DFT value: {}'.format(dft_energy[i]))
        del energy_gap[system[i]]['calc'][-1]
        energy_gap[system[i]]['calc'].append(dft_energy[i] / n_atoms[i])
        gap_per_atom_err[i] = float(abs(abs(gap_energy[i] / n_atoms[i]) - abs(dft_energy[i] / n_atoms[i])))
        if not os.path.isdir(str(i)):
            os.mkdir(str(i))
            with open(os.path.join(str(i), 'out.xyz'), 'w') as f:
                f.write(structures[i])
            atoms = read(os.path.join(str(i), 'out.xyz'))
            AseAtomsAdaptor().get_structure(atoms).to(fmt='POSCAR', filename=os.path.join(str(i), 'POSCAR'))

    df = 0
    for fa, fb in zip(np.array(gap_forces[i]), np.array(dft_forces[i])):
        df += np.linalg.norm(fa - fb) / 3
    force_err.append(df/len(gap_forces))


quit()


fcc_dft = -24.39050152/4
fcc_gap = -24.404683/4
bcc_dft = -11.97199694/2
bcc_gap = -11.989746/2
hcp_dft = -17.78139257/3
hcp_gap = -17.785577/3
sc_dft = -5.57232362
sc_gap = -5.6727757


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
    plt.scatter(energy_gap[sys]['calc'], energy_gap[sys]['pred'],
                marker='.', color=color[sys], label=None, s=0.5)

plt.xlabel(r'Computed Energy [eV/atom]', fontsize=16, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')

plt.scatter(5, 5, marker='.', color='k', label=r'GAP vs DFT', facecolor='w', s=25)

plt.scatter(fcc_dft, fcc_gap, marker='x', color='green', s=16)
plt.annotate(r'fcc', xy=(fcc_dft, fcc_gap), xytext=(fcc_dft-0.5, fcc_gap+0.5), color='green', fontsize=6,
             arrowprops=dict(facecolor='k', edgecolor='k', width=0.1,
                             headwidth=2.0, headlength=4.0, shrink=0.05))

plt.scatter(bcc_dft, bcc_gap, marker='x', color='navy', s=16)
plt.annotate(r'bcc', xy=(bcc_dft, bcc_gap), xytext=(bcc_dft, bcc_gap-0.5), color='navy', fontsize=6,
             arrowprops=dict(facecolor='k', edgecolor='k', width=0.1,
                             headwidth=2.0, headlength=4.0, shrink=0.05))

plt.scatter(hcp_dft, hcp_gap, marker='x', color='y', s=16)
plt.annotate(r'hcp', xy=(hcp_dft, hcp_gap), xytext=(hcp_dft+0.5, hcp_gap-0.5), color='y', fontsize=6,
             arrowprops=dict(facecolor='k', edgecolor='k', width=0.1,
                             headwidth=2.0, headlength=4.0, shrink=0.05))

plt.scatter(sc_dft, sc_gap, marker='x', color='m', s=16)
plt.annotate(r'sc', xy=(sc_dft, sc_gap), xytext=(sc_dft+0.5, sc_gap), color='m', fontsize=6,
             arrowprops=dict(facecolor='k', edgecolor='k', width=0.1,
                             headwidth=2.0, headlength=4.0, shrink=0.05))

plt.plot([-6.5, 0.5], [-6.5, 0.5], '-', color='k', linewidth=0.25)

for i, sys in enumerate(['fcc', 'bcc', 'sc', 'hcp', 'slab', 'cluster', 'addition']):
    plt.text(-6.1, -1-(0.2*i), sys, color=color[sys], fontsize=9)

plt.text(-3, -4.4, r'Avg GAP step time: {}s'.format(round(sum(gap_time)/len(gap_time), 5)), fontsize=8)

plt.text(-4.5, -0.6, r'Max GAP error: {} meV/atom'.format(round(max(gap_per_atom_err)*1000, 3)), fontsize=8)
plt.text(-4.5, -0.9, r'Mean GAP error: {} meV/atom'.format(
     round(sum(gap_per_atom_err)/len(gap_per_atom_err)*1000, 1)), fontsize=8)

plt.text(-4.5, -1.3, r'Mean force GAP error: {} eV/\AA'.format(round(max(force_err), 3)), fontsize=8)

plt.legend(loc='upper left')

plt.xlim(-7, 1)
plt.ylim(-7, 1)

plt.tight_layout()
plt.savefig('GAP_vs_DFT.png', dpi=300)
plt.close()
