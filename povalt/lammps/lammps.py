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
import time
import tempfile
import subprocess
from ase.atoms import Atoms
from ase.io import read as ase_read
from ase.io.lammpsdata import write_lammps_data
from povalt.helpers import find_binary
from pymatgen.core.structure import Structure


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

        self.err_chk_time = 60  # interval to check for errors

        if isinstance(structure, Structure):
            file = tempfile.mkstemp()[1]
            structure.to(fmt='POSCAR', filename=file)
            self.structure = ase_read(file, format='vasp')
        elif isinstance(structure, Atoms):
            self.structure = structure
        else:
            raise ValueError('Structure object has to be pymatgen structure or ase atoms object')

    def write_md(self, output_file, lammps_settings, units):
        """
        Writes LAMMPS input files to run MD
        Args:
            output_file: file to write to
            lammps_settings: settings string to write as lammps input
            units: lammps units to use

        Returns:
            nothing, writes content to files
        """

        write_lammps_data(fileobj=output_file, atoms=self.structure, units=units)
        with open('lammps.in', 'w') as f:
            for line in lammps_settings:
                f.write(line)

    def run(self, mpi_cmd, mpi_procs, binary, omp_threads, cmd_params, input_filename, output_filename):
        """
        Run LAMMPS to read input and write output
        Args:
            mpi_cmd: mpirun / srun command
            mpi_procs: number of processors to run on
            binary: specific LAMMPS binary to use
            omp_threads: number of openmpi threads to use
            cmd_params: command line arguments for LAMMPS
            input_filename: file that contains the input settings
            output_filename: file name to write std to

        Returns:
            nothing, writes stdout and stderr to files
        """

        bin_path = find_binary(binary).strip()
        cmd = 'export OMP_NUM_THREADS=' + str(omp_threads) + '; ' + \
              str(mpi_cmd) + ' -n ' + str(mpi_procs) + ' ' \
              + bin_path + str(' -i ') + str(input_filename) + ' ' + str(cmd_params)

        jobdir = os.getcwd()

        sout = open(os.path.join(jobdir, output_filename), 'w')
        serr = open(os.path.join(jobdir, 'lammps_error'), 'w')

        p = subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)

        # wait for process to end and periodically check for errors
        last_check = time.time()
        while p.poll() is None:
            time.sleep(30)
            if time.time() - last_check > self.err_chk_time:
                last_check = time.time()
                if self.found_error(os.path.join(jobdir, 'lammps_error')):   # TODO: implement me
                    p.kill()
                    raise LammpsError('Error during LAMMPS, check file fit_error')

        # close output files
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

        patterns = ['Cannot allocate memory']

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
