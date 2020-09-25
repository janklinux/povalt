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


class Dimer:
    """
    Class for dimer related generators
    """
    def __init__(self, species, lattice, min_dist, max_dist, n_points):
        if not isinstance(species, list):
            raise ValueError('Argument species has to be of type list')
        self.species = species
        self.lattice = lattice
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.n_points = n_points
        self.base_dir = os.getcwd()
        self.grid = np.linspace(start=min_dist, stop=max_dist, num=n_points, endpoint=True)

    def print_species(self):
        for i in range(len(self.species)):
            for j in range(i, len(self.species)):
                print('a: {} -- b: {}'.format(self.species[i], self.species[j]))

    def run_dimer_aims(self):
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
                    if os.path.isdir(str(np.round(d, 1))):
                        shutil.rmtree(str(np.round(d, 1)))

                for d in self.grid:
                    os.mkdir(str(np.round(d, 1)))
                    with open(os.path.join(str(np.round(d, 1)), 'geometry.in'), 'w') as f:
                        f.write('atom 0.0 0.0 0.0 {}\n'.format(self.species[i]))
                        f.write('  initial_moment 0.81\n')
                        f.write('atom 0.0 0.0 {} {}\n'.format(d, self.species[j]))
                        f.write('  initial_moment -0.81\n')

                    with open(os.path.join(str(np.round(d, 1)), 'control.in'), 'w') as f:
                        with open(os.path.join(self.base_dir, 'control.in'), 'r') as fin:
                            f.write(fin.read())

                    os.chdir(str(np.round(d, 1)))
                    os.system('mpirun -n 1 /home/jank/compile/FHIaims/BUILD/aims.200920.scalapack.mpi.x '
                              '| tee run.light')

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
                fig, ax1 = plt.subplots()
                color = 'red'
                ax1.plot(dists, energies, '.-', color=color)
                ax1.set_xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
                ax1.set_ylabel(r'Free Energy [eV]', fontsize=22, color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                # ax2 = ax1.twinx()
                # color = 'navy'
                # ax2.plot(dists, energies, '.-', color=color)
                # ax2.set_ylabel(r'Force in sep dir [eV/\AA]', fontsize=22, color=color)
                # ax2.tick_params(axis='y', labelcolor=color)
                plt.title(r'{}'.format(str(self.species[i]+self.species[j]), fontsize=14))
                fig.tight_layout()
                plt.show()
                # plt.savefig('dimer.png', dpi=170)
                plt.close()

                os.chdir(self.base_dir)
