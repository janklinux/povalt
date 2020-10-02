"""
Python package for training, validation and refinement of machine learned potentials

Copyright (C) 2020, Jan Kloppenburg

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from povalt.firetasks.vasp import VaspTasks
from pymatgen.io.vasp import Potcar, Outcar


class Dimer:
    """
    Class for dimer related generators
    """

    def __init__(self, species, lattice, min_dist, max_dist, show_curve):
        """
        Checks parameters and sets up the grid

        Args:
            species: list of species to generate dimer curves for
            lattice: for VASP only, the box we live in
            min_dist: minimal atom distance
            max_dist: maximal atom distance
        """

        if not isinstance(species, list):
            raise ValueError('Argument species has to be of type list')
        self.species = species
        self.lattice = lattice
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.show_curve = show_curve
        self.base_dir = os.getcwd()
        self.grid = np.linspace(start=min_dist, stop=1, num=5, endpoint=False)
        self.grid = np.append(self.grid, np.linspace(start=1, stop=3.5, num=30, endpoint=False))
        self.grid = np.append(self.grid, np.linspace(start=3.5, stop=max_dist, num=15, endpoint=True))

    def run_dimer_aims(self):
        """
        Runs the dimers with FHIaims and shows the result when a pair is completed

        Returns:
            nothing, writes and parses everything to disk

        """

        if os.path.isfile(os.path.join(self.base_dir, 'dimer.xyz')):
            os.unlink(os.path.join(self.base_dir, 'dimer.xyz'))

        for i in range(len(self.species)):
            for j in range(i, len(self.species)):
                dists = []
                energies = []

                print('Running {} <--> {}'.format(self.species[i], self.species[j]))

                species_dir = os.path.join(os.getcwd(), str(self.species[i] + self.species[j]))

                if os.path.isdir(species_dir):
                    shutil.rmtree(species_dir)
                os.mkdir(species_dir)
                os.chdir(species_dir)

                for d in self.grid:
                    if os.path.isdir(str(np.round(d, 2))):
                        shutil.rmtree(str(np.round(d, 2)))

                for d in self.grid:
                    os.mkdir(str(np.round(d, 2)))
                    with open(os.path.join(str(np.round(d, 2)), 'geometry.in'), 'w') as f:
                        f.write('atom 0.0 0.0 0.0 {}\n'.format(self.species[i]))
                        f.write('  initial_moment 0.81\n')
                        f.write('atom 0.0 0.0 {} {}\n'.format(d, self.species[j]))
                        f.write('  initial_moment -0.81\n')

                    with open(os.path.join(str(np.round(d, 2)), 'control.in'), 'w') as f:
                        with open(os.path.join(self.base_dir, 'control.in'), 'r') as fin:
                            f.write(fin.read())

                    os.chdir(str(np.round(d, 2)))
                    os.system('mpirun -n 1 /home/jank/bin/aims | tee run.light')

                    energy = None
                    forces = []
                    parse_forces = False

                    with open('run.light', 'r') as f:
                        for line in f:
                            if '| Total energy of the DFT / Hartree-Fock s.c.f. calculation      :' in line:
                                energy = float(line.split()[11])
                            if parse_forces:
                                forces.append([float(x) for x in line.split()[2:5]])
                                if i_force == 1:
                                    parse_forces = False
                                i_force += 1
                            if 'Total atomic forces (unitary forces cleaned) [eV/Ang]:' in line:
                                forces = []
                                i_force = 0
                                parse_forces = True

                    if energy is None:
                        raise ValueError('Computational Error, no total energy found. Check what happened.')

                    dists.append(d)
                    energies.append(energy)

                    with open('../dimer.xyz', 'a') as f:
                        f.write('2\n')
                        f.write('Lattice="200.0 0.0 0.0 0.0 200.0 0.0 0.0 0.0 200.0"'
                                ' Properties=species:S:1:pos:R:3:forces:R:3:force_mask:L:1'
                                ' stress="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" '
                                ' free_energy={:6.6f} pbc="T T T"'
                                ' config_type=dimer\n'.format(energy))
                        f.write('{} 0.000000 0.000000 0.000000 {:6.6f} {:6.6f} {:6.6f} 0\n'.format(
                            self.species[i], forces[0][0], forces[0][1], forces[0][2]))
                        f.write('{} 0.000000 0.000000 {:6.6f} {:6.6f} {:6.6f} {:6.6f} 0\n'.format(
                            self.species[j], float(d), forces[1][0], forces[1][1], forces[1][2]))
                        f.write('\n')

                    os.chdir(species_dir)

                if self.show_curve:
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
                    plt.plot(dists, energies, '.-', color='navy')
                    plt.xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
                    plt.ylabel(r'Free Energy [eV]', fontsize=22, color='k')
                    plt.title(r'{} -- {}'.format(self.species[i], self.species[j]), fontsize=16)
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                os.chdir(self.base_dir)

    def run_dimer_vasp(self):
        """
        Runs the dimers with VASP and shows the result when a pair is completed if desired

        Returns:
            nothing, writes and parses everything to disk

        """

        if os.path.isfile(os.path.join(self.base_dir, 'dimer.xyz')):
            os.unlink(os.path.join(self.base_dir, 'dimer.xyz'))

        for i in range(len(self.species)):
            for j in range(i, len(self.species)):
                dists = []
                energies = []

                print('Running {} <--> {}'.format(self.species[i], self.species[j]))

                species_dir = os.path.join(os.getcwd(), str(self.species[i] + self.species[j]))

                if os.path.isdir(species_dir):
                    shutil.rmtree(species_dir)
                os.mkdir(species_dir)
                os.chdir(species_dir)

                for d in self.grid:
                    if os.path.isdir(str(np.round(d, 2))):
                        shutil.rmtree(str(np.round(d, 2)))

                for d in self.grid:
                    os.mkdir(str(np.round(d, 2)))
                    with open(os.path.join(str(np.round(d, 2)), 'POSCAR'), 'w') as f:
                        f.write('autogen by PoValT\n')
                        f.write('1.0\n')
                        f.write('{} {} {}\n'.format(self.lattice[0][0], self.lattice[0][1], self.lattice[0][2]))
                        f.write('{} {} {}\n'.format(self.lattice[1][0], self.lattice[1][1], self.lattice[1][2]))
                        f.write('{} {} {}\n'.format(self.lattice[2][0], self.lattice[2][1], self.lattice[2][2]))
                        f.write(' {} {}\n'.format(self.species[i], self.species[j]))
                        f.write(' 1 1\n')
                        f.write('cartesian\n')
                        f.write('0.0 0.0 0.0\n')
                        f.write('0.0 0.0 {}\n'.format(d))

                    with open(os.path.join(str(np.round(d, 2)), 'INCAR'), 'w') as f:
                        with open(os.path.join(self.base_dir, 'INCAR'), 'r') as fin:
                            f.write(fin.read())
                        f.write('   DIPOL = 0.0 0.0 {}'.format(d/2))

                    with open(os.path.join(str(np.round(d, 2)), 'KPOINTS'), 'w') as f:
                        with open(os.path.join(self.base_dir, 'KPOINTS'), 'r') as fin:
                            f.write(fin.read())

                    pots = Potcar([self.species[i], self.species[j]])
                    pots.write_file(os.path.join(str(np.round(d, 2)), 'POTCAR'))

                    os.chdir(str(np.round(d, 2)))
                    os.system('mpirun -n 6 /home/jank/bin/vasp_std | tee run')

                    run = Outcar('OUTCAR')
                    forces = run.read_table_pattern(
                        header_pattern=r'\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+',
                        row_pattern=r'\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s'
                                    '+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)',
                        footer_pattern=r'\s--+',
                        postprocess=lambda x: float(x),
                        last_one_only=True)

                    dists.append(d)
                    energies.append(run.final_energy)

                    with open('../dimer.xyz', 'a') as f:
                        f.write('2\n')
                        f.write('Lattice="{} {} {} {} {} {} {} {} {}"'.format(
                            self.lattice[0][0], self.lattice[0][1], self.lattice[0][2],
                            self.lattice[1][0], self.lattice[1][1], self.lattice[1][2],
                            self.lattice[2][0], self.lattice[2][1], self.lattice[2][2]) +
                                ' Properties=species:S:1:pos:R:3:forces:R:3:force_mask:L:1'
                                ' stress="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" '
                                ' free_energy={:6.6f} pbc="T T T"'
                                ' config_type=dimer\n'.format(run.final_energy))
                        f.write('{} 0.000000 0.000000 0.000000 {:6.6f} {:6.6f} {:6.6f} 0\n'.format(
                            self.species[i], forces[0][0], forces[0][1], forces[0][2]))
                        f.write('{} 0.000000 0.000000 {:6.6f} {:6.6f} {:6.6f} {:6.6f} 0\n'.format(
                            self.species[j], float(d), forces[1][0], forces[1][1], forces[1][2]))
                        f.write('\n')

                    os.chdir(species_dir)

                if self.show_curve:
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
                    plt.plot(dists, energies, '.-', color='navy')
                    plt.xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
                    plt.ylabel(r'Free Energy [eV]', fontsize=22, color='k')
                    plt.title(r'{} -- {}'.format(self.species[i], self.species[j]), fontsize=16)
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                os.chdir(self.base_dir)


class Bulk:
    """
    Class to generate bulk structures for training data, incomplete, needs implementation and testing
    """

    def __init__(self, cell_type, atom_type, ncore):
        """
        checks and init variables
        """

        if cell_type not in ['fcc', 'bcc', 'sc', 'hcp']:
            raise ValueError('Wrong cell type, please correct')
        self.cell_type = cell_type
        self.atom_type = atom_type
        self.ncore = ncore

        pass

    def generate_fcc_cell(self):
        """
        generates a FCC cell with the specified atom as only occupant

        returns:
            a relaxation workflow for vasp

        """

        metadata = {'name': 'cell generation',
                    'task': 'relaxation',
                    'cell': self.cell_type}

        return VaspTasks.get_relax_wf(structure='fcc', structure_name='FCC input cell', atom_type=self.atom_type,
                                      vasp_cmd='srun --nodes 1 vasp_std', ncore=self.ncore, metadata=metadata)

    def generate_bcc_cell(self, atom_type):
        """
        generates a BCC cell with the specified atom as only occupant
        """
        pass

    def generate_sc_cell(self, atom_type):
        """
        generates a SC cell with the specified atom as only occupant
        """
        pass

    def generate_hcp_cell(self, atom_type):
        """
        generates a HCP cell with the specified atom as only occupant
        """
        pass
