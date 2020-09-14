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
import lzma
import time
import tempfile
import subprocess
from pymongo import MongoClient
from ase.atoms import Atoms
from ase.io import read as ase_read
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from povalt.helpers import find_binary
from povalt.firetasks.vasp import StaticFW
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

    def __init__(self, lammps_params, db_info, fw_spec):
        """
        Sets parameters
        Args:
            lammps_params: all LAMMPS parameters
            db_info: database info
            fw_spec: fireworks specs
        """
        self.lammps_params = lammps_params
        self.fw_spec = fw_spec
        self.structure = AseAtomsAdaptor().get_atoms(lammps_params['structure'])
        self.db_info = db_info
        self.run_dir = os.getcwd()

    def postprocess(self):
        pass

    def setup(self):
        os.chdir(self.run_dir)
        xml_label, xml_name = self.download_potential()
        if xml_name is None:
            raise ValueError('Potential download seems to have failed, check internal routines...')
        write_lammps_data(fileobj=self.lammps_params['atoms_filename'],
                          atoms=self.structure,
                          units=self.lammps_params['units'])
        with open('lammps.in', 'w') as f:
            for line in self.lammps_params['lammps_settings']:
                f.write(re.sub('POT_FW_LABEL', xml_label,
                               re.sub('POT_FW_NAME', xml_name, line.strip())) + '\n')

    def run(self):
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

    def download_potential(self):
        connection = None
        if 'ssl' in self.db_info:
            if self.db_info['ssl'].lower() == 'true':
                try:
                    connection = MongoClient(host=self.db_info['host'], port=self.db_info['port'],
                                             username=self.db_info['user'], password=self.db_info['password'],
                                             ssl=True, tlsCAFile=self.db_info['ssl_ca_certs'],
                                             ssl_certfile=self.db_info['ssl_certfile'])
                except ConnectionError:
                    raise ConnectionError('Mongodb connection failed')
            else:
                try:
                    connection = MongoClient(host=self.db_info['host'], port=self.db_info['port'],
                                             username=self.db_info['user'], password=self.db_info['password'],
                                             ssl=False)
                except ConnectionError:
                    raise ConnectionError('Mongodb connection failed')

        if connection is None:
            raise ConnectionAbortedError('Connection failure, check internal routines')

        db = connection[self.db_info['database']]
        try:
            db.authenticate(self.db_info['user'], self.db_info['password'])
        except ConnectionRefusedError:
            raise ConnectionRefusedError('Mongodb authentication failed')
        collection = db[self.db_info['potential_collection']]

        xml_name = None
        xml_label = None
        for pot in collection.find():
            for p in pot:
                if p != '_id':
                    bits = p.split(':')
                    if len(bits) == 2:
                        xml_name = bits[0] + '.' + bits[1]
                    if len(bits) == 4:
                        xml_label = p.split(':')[3][:-1]
                    with open(re.sub(':', '.', p), 'wb') as f:
                        f.write(lzma.decompress(pot[p]))
        return xml_label, xml_name

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
        incar_mod = {'EDIFF': 1E-4, 'ENCUT': 220, 'NCORE': 2, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
                     'ALGO': 'Fast', 'AMIN': 0.01, 'NELM': 100, 'LAECHG': '.FALSE.', 'LCHARG': '.FALSE.'}
                     # 'IDIPOL': 3, 'LDIPOL': '.TRUE.', 'DIPOL': '0.5 0.5 0.5'}

        print('\n')
        print(kpt_set)
        print('   *** CHECK IF SLAB OR BULK ***\n\n')

        vis = MPStaticSet(rerun_structure)
        v = vis.as_dict()
        v.update({"user_kpoints_settings": kpt_set})
        vis_static = vis.__class__.from_dict(v)

        static_wf = Workflow([StaticFW(structure=rerun_structure, vasp_input_set=vis_static,
                                       vasp_cmd='mpirun -n 4 vasp_std', name='VASP analysis',
                                       db_info=self.db_info)])
        run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
        return run_wf


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
        Writes LAMMPS input files to run MD and gets the potential from the db into a file on disk
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
