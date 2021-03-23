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

import io
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from pymatgen.io.vasp import Potcar, Vasprun


class Dimer:
    """
    Class for dimer related generators
    """

    def __init__(self, species, lattice, min_dist, max_dist, show_spin_curves,
                 show_result, cores, mpi_cmd, non_collinear=False):
        """
        Checks parameters and sets up the grid

        Args:
            species (list): list of species to generate dimer curves for
            lattice (3x3 array): for VASP only, the box we live in
            min_dist (float): minimal atom distance
            max_dist (float): maximal atom distance
            show_spin_curves (bool): show the individual spin curves?
            show_result (bool): show the total result stitched curve?
            non_collinear (bool): run vasp in ncl mode?
        """

        if not isinstance(species, list):
            raise ValueError('Argument species has to be of type list')
        self.species = species
        self.lattice = lattice
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.show_spin_curves = show_spin_curves
        self.show_result = show_result
        self.cores = cores
        self.base_dir = os.getcwd()
        self.grid = np.linspace(start=min_dist, stop=max_dist, num=100, endpoint=True)
        self.ncl = non_collinear
        self.mpi_cmd = mpi_cmd

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
                for fixed_spin in [0, 1, 2, 3, 4]:
                    dists = []
                    energies = []

                    print('Running {:2s} <--> {:2s} || Spin: {:d}'.format(self.species[i], self.species[j], fixed_spin))

                    run_dir = os.path.join(os.getcwd(), str(self.species[i] + self.species[j] +
                                                            '_spin_{:d}'.format(fixed_spin)))

                    if not os.path.isdir(run_dir):
                        os.mkdir(run_dir)
                    os.chdir(run_dir)

                    for d in self.grid:
                        if os.path.isdir(str(np.round(d, 2))):
                            continue
                        os.mkdir(str(np.round(d, 2)))
                        os.chdir(str(np.round(d, 2)))

                        with open('geometry.in', 'w') as f:
                            f.write('atom 0.0 0.0 0.0 {}\n'.format(self.species[i]))
                            f.write('  initial_moment 0.81\n')
                            f.write('atom 0.0 0.0 {} {}\n'.format(d, self.species[j]))

                        forces = []
                        converged = False
                        while not converged:
                            for mix in [0.01, 0.05, 0.1]:
                                for smear in [0.05, 0.001, 0.1]:
                                    converged = False
                                    with open('control.in', 'w') as f:
                                        with open(os.path.join(self.base_dir, 'control.in'), 'r') as fin:
                                            for line in fin:
                                                f.write(re.sub('MIXING', str(mix),
                                                               re.sub('SMEAR', str(smear), line)))
                                            f.write('\n')
                                            f.write('  fixed_spin_moment {:2.2f}'.format(fixed_spin))
                                            f.write('\n\n')

                                    os.system('{} -n {} /home/jank/bin/aims > run'.format(self.mpi_cmd, self.cores))

                                    forces = None
                                    parse_forces = False
                                    with open('run', 'r') as f:
                                        for line in f:
                                            if '| Total energy of the DFT / Hartree-Fock s.c.f. calculation      :' \
                                                    in line:
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
                                            if 'Have a nice day.' in line:
                                                converged = True

                                    if converged:
                                        print('Distance: {:3.3f} Mix: {:2.2f} Smear: {:3.3f} Converged.'
                                              .format(np.round(d, 2), mix, smear))
                                        break
                                if converged:
                                    break

                            if mix == 0.1 and smear == 0.001 and not converged:
                                raise RuntimeError('DFT non-convergent with internal settings')

                        os.chdir('..')

                        if energy is None:
                            raise ValueError('DFT not converged, check settings...')

                        with open('dimer.xyz', 'a') as f:
                            f.write('2\n')
                            f.write('Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0"'
                                    ' Properties=species:S:1:pos:R:3:forces:R:3:force_mask:L:1'
                                    ' stress="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"'
                                    ' virial="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"'
                                    ' free_energy={:6.6f} pbc="T T T"'
                                    ' config_type=dimer\n'.format(energy))
                            f.write('{} 0.000000 0.000000 0.000000 {:6.6f} {:6.6f} {:6.6f} 0\n'.format(
                                self.species[i], forces[0][0], forces[0][1], forces[0][2]))
                            f.write('{} 0.000000 0.000000 {:6.6f} {:6.6f} {:6.6f} {:6.6f} 0\n'.format(
                                self.species[j], float(d), forces[1][0], forces[1][1], forces[1][2]))

                    os.chdir(run_dir)

                    if self.show_spin_curves:
                        for d in self.grid:
                            dists.append(d)
                            with open(os.path.join(str(np.round(d, 2)), 'run'), 'r') as f:
                                for line in f:
                                    if '| Total energy of the DFT / Hartree-Fock s.c.f. calculation      :' in line:
                                        energies.append(float(line.split()[11]))

                        plt.rc('text', usetex=True)
                        plt.rc('font', family='sans-serif', serif='Palatino')
                        plt.rcParams['font.family'] = 'DejaVu Sans'
                        plt.rcParams['font.sans-serif'] = 'cm'
                        plt.plot(dists, energies, '.-', color='navy')
                        plt.xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
                        plt.ylabel(r'Free Energy [eV]', fontsize=22, color='k')
                        plt.title(r'FHIaims: {:2s} -- {:2s} || Spin: {:1d}'
                                  .format(self.species[i], self.species[j], fixed_spin), fontsize=16)
                        plt.show()
                        plt.close()

                    os.chdir(self.base_dir)

        data = dict()
        for i in range(len(self.species)):
            for j in range(i, len(self.species)):
                for fixed_spin in [0, 1, 2, 3, 4]:
                    dists = []
                    forces = []
                    energies = []

                    # print('Collecting {:2s} <--> {:2s} || Spin: {:d}'
                    #       .format(self.species[i], self.species[j], fixed_spin))

                    run_dir = os.path.join(self.base_dir, str(self.species[i] + self.species[j] +
                                           '_spin_{:d}'.format(fixed_spin)))

                    os.chdir(run_dir)

                    ftmp = []
                    for d in self.grid:
                        if not os.path.isdir(str(np.round(d, 2))):
                            raise FileNotFoundError('Directory {} not found, check computations...'
                                                    .format(np.round(d, 2)))

                        dists.append(float(d))

                        parse_forces = False
                        with open(os.path.join(str(np.round(d, 2)), 'run'), 'r') as f:
                            for line in f:
                                if '| Total energy of the DFT / Hartree-Fock s.c.f. calculation      :' \
                                        in line:
                                    energies.append(float(line.split()[11]))
                                if parse_forces:
                                    forces.append([float(x) for x in line.split()[2:5]])
                                    if i_force == 1:
                                        parse_forces = False
                                    i_force += 1
                                if 'Total atomic forces (unitary forces cleaned) [eV/Ang]:' in line:
                                    forces = []
                                    i_force = 0
                                    parse_forces = True
                        ftmp.append(forces)

                    data[str(self.species[i]+self.species[j])+str(fixed_spin)] = {'dist': dists, 'energy': energies,
                                                                                  'forces': ftmp}

                    os.chdir('..')

        lowest = dict()
        for i in range(len(self.species)):
            for j in range(i, len(self.species)):
                dtmp = []
                etmp = []
                itmp = []
                ftmp = []

                for ik, d in enumerate(self.grid):
                    dtmp.append(d)

                    all_ens = []
                    all_frc = []
                    for fixed_spin in [0, 1, 2, 3, 4]:
                        all_ens.append(data[str(self.species[i] + self.species[j]) + str(fixed_spin)]['energy'][ik])
                        all_frc.append(data[str(self.species[i] + self.species[j]) + str(fixed_spin)]['forces'][ik])

                    min_idx = np.where(all_ens == np.amin(all_ens))

                    if len(min_idx[0]) != 1:
                        raise ValueError('Index error, only one minimum shall be allowed...')

                    # print(self.species[i], self.species[j], min_idx, all_ens)

                    etmp.append(all_ens[min_idx[0][0]])
                    ftmp.append(all_frc[min_idx[0][0]])
                    itmp.append(min_idx[0][0])

                lowest[str(self.species[i]+self.species[j])] = {'dist': dtmp,
                                                                'energy': etmp,
                                                                'forces': ftmp,
                                                                'spin_idx': itmp}

        if self.show_result:
            plt.rc('text', usetex=True)
            plt.rc('font', family='sans-serif', serif='Palatino')
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = 'cm'

            for i in range(len(self.species)):
                for j in range(i, len(self.species)):
                    plt.plot(lowest[str(self.species[i] + self.species[j])]['dist'],
                             lowest[str(self.species[i] + self.species[j])]['energy'],
                             '-', label=str(self.species[i] + self.species[j]))

            plt.xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
            plt.ylabel(r'Free Energy [eV]', fontsize=22, color='k')
            plt.title(r'FHIaims: || Spin compare', fontsize=16)
            plt.legend(loc='best')
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
                for fixed_spin in [0, 1, 2, 3]:
                    print('Running {} <--> {} Spin: {}'.format(self.species[i], self.species[j], fixed_spin))

                    run_dir = os.path.join(os.getcwd(), str(self.species[i] + self.species[j] +
                                                            '_spin_{:d}'.format(fixed_spin)))

                    if not os.path.isdir(run_dir):
                        os.mkdir(run_dir)
                    os.chdir(run_dir)

                    for d in self.grid:
                        if os.path.isdir(str(np.round(d, 2))):
                            continue
                        os.mkdir(str(np.round(d, 2)))
                        os.chdir(str(np.round(d, 2)))

                        with open('POSCAR', 'w') as f:
                            f.write('autogen by PoValT\n')
                            f.write('1.0\n')
                            f.write('{} {} {}\n'.format(self.lattice[0][0], self.lattice[0][1], self.lattice[0][2]))
                            f.write('{} {} {}\n'.format(self.lattice[1][0], self.lattice[1][1], self.lattice[1][2]))
                            f.write('{} {} {}\n'.format(self.lattice[2][0], self.lattice[2][1], self.lattice[2][2]))
                            if self.species[i] == self.species[j]:
                                f.write(' {}\n'.format(self.species[i]))
                                f.write(' 2\n')
                            else:
                                f.write(' {} {}\n'.format(self.species[i], self.species[j]))
                                f.write(' 1 1\n')
                            f.write('cartesian\n')
                            f.write('0.0 0.0 0.0\n')
                            f.write('0.0 0.0 {}\n'.format(np.round(d, 3)))

                        with open('INCAR', 'w') as f:
                            with open(os.path.join(self.base_dir, 'INCAR'), 'r') as fin:
                                f.write(fin.read())
                            f.write('   NUPDOWN = {}\n'.format(fixed_spin))
                            if self.ncl:
                                f.write('  MAGMOM = 0.74 -0.38 0.51 -0.12 0.83 -0.62\n')
                                f.write('  SAXIS  = 0 0 1\n')
                                f.write('  LSORBIT = .TRUE.\n')
                                f.write('  LMAXMIX = 6\n')
                                f.write('  NBANDS  = 30\n')
                            else:
                                f.write('  MAGMOM = 0.74 -0.83\n')

                        os.link(os.path.join(self.base_dir, 'KPOINTS'), 'KPOINTS')
                        pots = Potcar([self.species[i], self.species[j]])
                        pots.write_file('POTCAR')

                        if self.ncl:
                            os.system('{} -n {} vasp_ncl | tee run'.format(self.mpi_cmd, self.cores))
                        else:
                            os.system('{} -n {} vasp_std | tee run'.format(self.mpi_cmd, self.cores))

                        run = read('vasprun.xml')
                        forces = run.get_forces()

                        os.chdir('..')

                        with open('dimer.xyz', 'a') as f:
                            f.write('2\n')
                            f.write('Lattice="{} {} {} {} {} {} {} {} {}"'.format(
                                self.lattice[0][0], self.lattice[0][1], self.lattice[0][2],
                                self.lattice[1][0], self.lattice[1][1], self.lattice[1][2],
                                self.lattice[2][0], self.lattice[2][1], self.lattice[2][2]) +
                                    ' Properties=species:S:1:pos:R:3:forces:R:3:force_mask:L:1'
                                    ' stress="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" '
                                    ' free_energy={:6.6f} pbc="T T T" virial="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"'
                                    ' config_type=dimer\n'.format(run.get_potential_energy(force_consistent=True)))
                            f.write('{} 0.000000 0.000000 0.000000 {:6.6f} {:6.6f} {:6.6f} 0\n'.format(
                                self.species[i], forces[0][0], forces[0][1], forces[0][2]))
                            f.write('{} 0.000000 0.000000 {:6.6f} {:6.6f} {:6.6f} {:6.6f} 0\n'.format(
                                self.species[j], float(d), forces[1][0], forces[1][1], forces[1][2]))

                    os.chdir('..')

                if self.show_spin_curves:
                    dists = []
                    energies = []
                    for d in self.grid:
                        dists.append(d)
                        energies.append(read(os.path.join(str(np.round(d, 2)), 'vasprun.xml')).
                                        get_potential_energy(force_consistent=True))

                    plt.rc('text', usetex=True)
                    plt.rc('font', family='sans-serif', serif='Palatino')
                    plt.rcParams['font.family'] = 'DejaVu Sans'
                    plt.rcParams['font.sans-serif'] = 'cm'
                    plt.plot(dists, energies, '.-', color='navy')
                    plt.xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
                    plt.ylabel(r'Free Energy [eV]', fontsize=22, color='k')
                    plt.title(r'VASP: {} -- {}'.format(self.species[i], self.species[j]), fontsize=16)
                    plt.show()
                    plt.close()

                os.chdir(self.base_dir)

        data = dict()
        for i in range(len(self.species)):
            for j in range(i, len(self.species)):
                for fixed_spin in [0, 1, 2, 3]:
                    dists = []
                    energies = []

                    print('Collecting {:2s} <--> {:2s} || Spin: {:d}'
                          .format(self.species[i], self.species[j], fixed_spin))

                    run_dir = os.path.join(self.base_dir, str(self.species[i] + self.species[j] +
                                           '_spin_{:d}'.format(fixed_spin)))

                    os.chdir(run_dir)

                    ftmp = []
                    for d in self.grid:
                        if not os.path.isdir(str(np.round(d, 2))):
                            raise FileNotFoundError('Directory {} not found, check computations...'
                                                    .format(np.round(d, 2)))

                        dists.append(float(d))

                        run = read(os.path.join(str(np.round(d, 2)), 'vasprun.xml'))
                        if not Vasprun(os.path.join(str(np.round(d, 2)), 'vasprun.xml')).converged:
                            print(run_dir, np.round(d, 2))
                            print('not conv....')
                            quit()

                        energies.append(run.get_potential_energy(force_consistent=True))
                        ftmp.append(run.get_forces())

                    data[str(self.species[i]+self.species[j])+str(fixed_spin)] = {'dist': dists, 'energy': energies,
                                                                                  'forces': ftmp}

                    os.chdir('..')

        lowest = dict()
        for i in range(len(self.species)):
            for j in range(i, len(self.species)):
                dtmp = []
                etmp = []
                itmp = []
                ftmp = []
                atmp = []

                for ik, d in enumerate(self.grid):
                    dtmp.append(d)

                    all_ens = []
                    all_frc = []
                    for fixed_spin in [0, 1, 2, 3]:
                        all_ens.append(data[str(self.species[i] + self.species[j]) + str(fixed_spin)]['energy'][ik])
                        all_frc.append(data[str(self.species[i] + self.species[j]) + str(fixed_spin)]['forces'][ik])

                    min_idx = np.where(all_ens == np.amin(all_ens))

                    if len(min_idx[0]) != 1:
                        raise ValueError('Index error, only one minimum shall be allowed...')

                    # print(self.species[i], self.species[j], min_idx, all_ens)

                    etmp.append(all_ens[min_idx[0][0]])
                    ftmp.append(all_frc[min_idx[0][0]])
                    itmp.append(min_idx[0][0])

                    at = read(self.species[i] + self.species[j] + '_spin_' + str(itmp[-1]) +
                              '/' + str(np.round(d, 2)) + '/vasprun.xml')

                    at.new_array('force_mask', np.array([False for _ in range(len(at))]))

                    stress = at.get_stress(voigt=False)
                    vol = at.get_volume()
                    virial = -np.dot(vol, stress)

                    file = io.StringIO()
                    write(filename=file, images=at, format='extxyz', parallel=False)
                    file.seek(0)
                    xyz = file.readlines()
                    file.close()

                    xyz[1] = xyz[1].strip() + ' virial="{} {} {} {} {} {} {} {} {}" ' \
                                              'config_type=dimer_{}\n'.format(
                        virial[0][0], virial[0][1], virial[0][2],
                        virial[1][0], virial[1][1], virial[1][2],
                        virial[2][0], virial[2][1], virial[2][2],
                        str(self.species[j]+self.species[i]))

                    with open('/tmp/delmetmp', 'wt') as f:
                        for line in xyz:
                            f.write(line)
                    atmp.append(read('/tmp/delmetmp'))

                lowest[str(self.species[i]+self.species[j])] = {'dist': dtmp,
                                                                'energy': etmp,
                                                                'forces': ftmp,
                                                                'spin_idx': itmp}

                filename = ''.join([self.species[i], self.species[j]])+'_dimer.xyz'
                write(filename=filename, format='extxyz', images=atmp)

                # print(os.getcwd())
                print('wrote: ', self.species[i], self.species[j])

        if self.show_result:
            plt.rc('text', usetex=True)
            plt.rc('font', family='sans-serif', serif='Palatino')
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = 'cm'

            for i in range(len(self.species)):
                for j in range(i, len(self.species)):
                    plt.plot(lowest[str(self.species[i] + self.species[j])]['dist'],
                             lowest[str(self.species[i] + self.species[j])]['energy'],
                             '-', label=str(self.species[i] + self.species[j]))

            plt.xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
            plt.ylabel(r'Free Energy [eV]', fontsize=22, color='k')
            plt.title(r'VASP: || Spin compare', fontsize=16)
            plt.legend(loc='best')
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

        # metadata = {'name': 'cell generation',
        #             'task': 'relaxation',
        #             'cell': self.cell_type}

        # return VaspTasks.get_relax_wf(structure='fcc', structure_name='FCC input cell', atom_type=self.atom_type,
        #                               vasp_cmd='srun --nodes 1 vasp_std', ncore=self.ncore, metadata=metadata)

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
