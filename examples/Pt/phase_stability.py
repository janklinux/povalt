import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Kpoints, Outcar
from pymatgen.core.structure import Structure


plot_only = True

color = {'fcc': 'navy', 'bcc': 'm', 'sc': 'b', 'hcp': 'g'}

base_dir = os.getcwd()
run_dir = os.path.join(os.getcwd(), 'run_dir')

if not plot_only:
    gap_energy = {'fcc': {'lat_const': [], 'energy': []}, 'bcc': {'lat_const': [], 'energy': []},
                  'sc': {'lat_const': [], 'energy': []}, 'hcp': {'lat_const': [], 'energy': []}}
    dft_energy = {'fcc': {'lat_const': [], 'energy': []}, 'bcc': {'lat_const': [], 'energy': []},
                  'sc': {'lat_const': [], 'energy': []}, 'hcp': {'lat_const': [], 'energy': []}}
    for csys in ['fcc', 'bcc', 'sc', 'hcp']:
        print('Running: {:3.3s}'.format(csys), end='')
        sys.stdout.flush()
        os.chdir(base_dir)
        cell = Structure.from_file(os.path.join(csys, 'POSCAR'))
        for dv in np.arange(start=-0.6, stop=1.9, step=0.1):
            print(' {}'.format(np.round(dv, 1)), end='')
            sys.stdout.flush()
            os.chdir(run_dir)
            tmp_dir = 'delme'
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)
            os.chdir(tmp_dir)

            run_cell = Structure(lattice=np.array(cell.lattice.matrix) + np.array([[dv, 0, 0], [0, dv, 0], [0, 0, dv]]),
                                 species=cell.species, coords=cell.frac_coords, coords_are_cartesian=False)

            write('cell.xyz', images=AseAtomsAdaptor().get_atoms(run_cell), format='extxyz')

            for file in os.listdir(base_dir):
                if file.startswith('platinum.xml') or file.startswith('compress.dat'):
                    os.symlink(os.path.join('..', '..', file), file)

            os.system('quip atoms_filename=cell.xyz param_filename=platinum.xml e > quip.result')

            with open('quip.result', 'r') as f:
                for line in f:
                    if 'Energy' in line:
                        gap_energy[csys]['lat_const'].append(float(cell.lattice.matrix[2][2] + dv))
                        gap_energy[csys]['energy'].append(float(line.split('=')[1]))

            run_cell.to(fmt='POSCAR', filename='POSCAR')
            Kpoints.gamma_automatic(kpts=[6, 6, 6], shift=(0, 0, 0)).write_file('KPOINTS')
            os.symlink('../../POTCAR', 'POTCAR')
            os.symlink('../../INCAR', 'INCAR')

            os.system('nice -n 10 mpirun -n 4 vasp_std > run')

            dft_energy[csys]['lat_const'].append(float(cell.lattice.matrix[2][2] + dv))
            dft_energy[csys]['energy'].append(float(Outcar('OUTCAR').final_energy))

            os.chdir(run_dir)
            shutil.rmtree(tmp_dir)
        print('')
        sys.stdout.flush()

    with open('gap_data.json', 'w') as f:
        json.dump(obj=gap_energy, fp=f)
    with open('dft_data.json', 'w') as f:
        json.dump(obj=dft_energy, fp=f)

else:
    with open('gap_data.json', 'r') as f:
        gap_energy = json.load(f)
    with open('dft_data.json', 'r') as f:
        dft_energy = json.load(f)


os.chdir(base_dir)

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

for csys in ['fcc', 'bcc', 'sc', 'hcp']:
    plt.plot(gap_energy[csys]['lat_const'], gap_energy[csys]['energy'], '-',
             linewidth=1, color=color[csys], label=csys+'-GAP')
    plt.plot(dft_energy[csys]['lat_const'], dft_energy[csys]['energy'], '.',
             linewidth=1, color=color[csys], label=csys+'-DFT')

plt.xlabel(r'Lattice Constant [\AA]', fontsize=16, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')

plt.legend(loc='upper right', fontsize=8)

# plt.xlim(-7, 1)
# plt.ylim(-7, 1)

plt.tight_layout()
plt.savefig('phase_stability.png', dpi=300)
plt.close()