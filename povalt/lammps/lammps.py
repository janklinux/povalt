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

import re
import os
import time
import tempfile
import subprocess
from ase.atoms import Atoms
from ase.io import read as ase_read
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from povalt.helpers import find_binary
from povalt.firetasks.vasp import StaticFW, AnalFW
from pymatgen.core.structure import Structure, Element
from custodian.custodian import Job
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.sets import MPStaticSet
from atomate.vasp.powerups import add_modify_incar
from fireworks import Workflow


class LammpsJob(Job):
    """
    Class to run LAMMPS MD as firework
    """

    def __init__(self, lammps_params, fw_spec):
        """
        Sets parameters
        Args:
            lammps_params: all LAMMPS parameters
            potential_info: file names of the potential
        """
        self.lammps_params = lammps_params
        self.fw_spec = fw_spec
        self.potential_info = fw_spec['potential_info']
        self.structure = AseAtomsAdaptor().get_atoms(lammps_params['structure'])
        self.run_dir = os.getcwd()

    def setup(self):
        self.setup_lammps_md()

    def run(self):
        return self.run_lammps_md()

    def postprocess(self):
        pass

    def setup_lammps_md(self):
        os.chdir(self.run_dir)
        pot_name = self.link_potential()
        write_lammps_data(fileobj=self.lammps_params['atoms_filename'],
                          atoms=self.structure,
                          units=self.lammps_params['units'])
        with open('lammps.in', 'w') as f:
            for line in self.lammps_params['lammps_settings']:
                f.write(re.sub('POT_FW_LABEL', self.potential_info['label'],
                               re.sub('POT_FW_NAME', pot_name, line.strip())) + '\n')

    def run_lammps_md(self):
        os.chdir(self.run_dir)
        for item in self.lammps_params:
            if self.lammps_params[item] is not None:
                self.lammps_params[item] = str(self.lammps_params[item])

        os.environ['OMP_NUM_THREADS'] = self.lammps_params['omp_threads']

        if self.lammps_params['mpi_cmd'] is not None:
            if self.lammps_params['mpi_procs'] is None:
                raise ValueError('Running in MPI you have to define mpi_procs')
            cmd = str(find_binary(self.lammps_params['mpi_cmd']).strip()) + ' -n ' + \
                  str(self.lammps_params['mpi_procs']) + ' '
        else:
            cmd = ''

        cmd += find_binary(self.lammps_params['lmp_bin']).strip() + str(' -in lammps.in ') + \
               str(self.lammps_params['lmp_params'])

        try:
            with open('std_err', 'w') as serr, open('std_out', 'w') as sout:
                p = subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)
        except FileNotFoundError:
            raise FileNotFoundError('Command execution failed, check std_err.')
        finally:
            os.environ['OMP_NUM_THREADS'] = str(1)
        return p

    def link_potential(self):
        pot_name = ' *** FILE NOT FOUND *** '
        for file in self.potential_info['files']:
            os.symlink(os.path.join(self.potential_info['path'], file),
                       os.path.join(self.run_dir, file))
            if len(file) > len(self.potential_info['label']):
                pot_name = file.split('.')[3][:-1]
        return pot_name

    def get_vasp_static_dft(self):
        """
        Generates a static DFT run for VASP
        Returns:
            the workflow
        """
        with open(os.path.join(self.run_dir, 'final_positions.atom'), 'r') as f:
            final_atoms = read_lammps_dump_text(fileobj=f, index=-1)
        lammps_result = AseAtomsAdaptor.get_structure(final_atoms)
        lattice = lammps_result.lattice
        species = [Element('Pt') for _ in range(len(lammps_result.species))]
        coords = lammps_result.frac_coords
        rerun_structure = Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=False)

        kpt_set = Kpoints.automatic_density(rerun_structure, kppa=1200, force_gamma=False)
        incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 2, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
                     'ALGO': 'Fast', 'AMIN': 0.01, 'NELM': 300, 'LAECHG': 'False'}
                     # 'IDIPOL': 3, 'LDIPOL': '.TRUE.', 'DIPOL': '0.5 0.5 0.5'}

        print(kpt_set)
        print('   *** CHECK IF SLAB OR BULK ***')

        vis = MPStaticSet(rerun_structure)
        v = vis.as_dict()
        v.update({"user_kpoints_settings": kpt_set})
        vis_static = vis.__class__.from_dict(v)

        static_wf = Workflow([StaticFW(structure=rerun_structure, vasp_input_set=vis_static,
                                       vasp_cmd='mpirun -n 4 vasp_std')], name='VASP analysis')
        run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
        return run_wf  #, Workflow([AnalFW(self.run_dir)])






class Lammps:
    """
    General class for all LAMMPS related jobs like generating and running structures
    for validation and potential refinement
    """

    def __init__(self, structure):
        """
        init parameters and find out what object we have
        Args:
            structure: the structure object we deal with
        """

        self.err_chk_time = 60  # interval to check for errors (seconds)

        if isinstance(structure, Structure):
            file = tempfile.mkstemp()[1]
            structure.to(fmt='POSCAR', filename=file)
            self.structure = ase_read(file, format='vasp')
        elif isinstance(structure, Atoms):
            self.structure = structure
        else:
            raise ValueError('Structure object has to be pymatgen structure or ase atoms object')

    def write_md(self, atoms_file, lammps_settings, units):
        """
        Writes LAMMPS input files to run MD
        Args:
            atoms_file: file to write to
            lammps_settings: settings string to write as lammps input
            units: lammps units to use

        Returns:
            nothing, writes content to files
        """

        write_lammps_data(fileobj=atoms_file, atoms=self.structure, units=units)
        with open('lammps.in', 'w') as f:
            for line in lammps_settings:
                f.write(line.strip() + '\n')

    def run(self, binary, cmd_params, output_filename, mpi_cmd=None, mpi_procs=None):
        """
        Run LAMMPS to read input and write output
        Args:
            binary: specific LAMMPS binary to use
            cmd_params: command line arguments for LAMMPS
            output_filename: file name to write std to
            mpi_cmd: mpirun / srun command, optional
            mpi_procs: number of processors to run on, only required if mpi_cmd is set

        Returns:
            nothing, writes stdout and stderr to files
        """

        os.environ['OMP_NUM_THREADS'] = str(1)

        if mpi_cmd is not None:
            if mpi_procs is None:
                raise ValueError('Running in MPI you have to define mpi_procs')
            cmd = str(find_binary(mpi_cmd).strip()) + ' -n ' + str(mpi_procs) + ' '
        else:
            cmd = ''

        cmd += find_binary(binary).strip() + str(' -in lammps.in ') + str(cmd_params)

        jobdir = os.getcwd()

        sout = open(os.path.join(jobdir, output_filename), 'w')
        serr = open(os.path.join(jobdir, 'lammps_error'), 'w')

        p = subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)

        # wait for process to end and periodically check for errors
        last_check = time.time()
        while p.poll() is None:
            time.sleep(3)
            if time.time() - last_check > self.err_chk_time:
                last_check = time.time()
                if self.found_error(os.path.join(jobdir, output_filename)):
                    p.kill()
                    raise LammpsError('Error during LAMMPS, check file lammps_error and {}'.format(output_filename))

        sout.close()
        serr.close()

    @staticmethod
    def found_error(filename):
        """
        Function to check for specific errors during LAMMPS run.

        Args:
            filename: filename to check for string pattern

        Returns:
            True if string in filename, False otherwise
        """

        patterns = ['Cannot open input script']

        with open(filename, 'r') as f:
            for line in f:
                for pat in patterns:
                    if pat in line:
                        return True


class LammpsError(Exception):
    """
    Error class for LAMMPS errors
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
