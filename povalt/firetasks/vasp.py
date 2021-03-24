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
import os
import gzip
import datetime
import subprocess
import numpy as np
from pymongo import MongoClient
from ase.io import read as aseread
from ase.io import write as asewrite
from povalt.helpers import env_chk
from fireworks import Workflow, Firework, FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from pymatgen.io.vasp import Kpoints, Outcar, Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.core.structure import Structure, Lattice
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from atomate.vasp.powerups import add_modify_incar
from custodian import Custodian
from custodian.custodian import Job
from custodian.vasp.handlers import VaspErrorHandler, MeshSymmetryErrorHandler, UnconvergedErrorHandler, \
    NonConvergingErrorHandler, PotimErrorHandler, PositiveEnergyErrorHandler, FrozenJobErrorHandler, StdErrHandler
from custodian.vasp.validators import VasprunXMLValidator, VaspFilesValidator


@explicit_serialize
class FewstepsFW(Firework):
    def __init__(self, structure, vasp_input_set, vasp_cmd, name):
        """
        Standard static calculation Firework for a structure.

        Args:
            structure (Structure): Input structure
            vasp_input_set (VaspInputSet): input set to use
            vasp_cmd (str): Command to run vasp.
            name: the name of the workflow
        """

        t = list()
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(RunVaspFewCustodian(vasp_cmd=vasp_cmd))
        super(FewstepsFW, self).__init__(t, name=name)


@explicit_serialize
class RunVaspFewCustodian(FiretaskBase):
    """
    Run VASP using custodian "on rails", fixes most runtime errors, uses not all handlers
    but default validators from custodian package

    Required params:
        vasp_cmd (str): the name of the full executable for running VASP
    """

    required_params = ['vasp_cmd']
    optional_params = []

    def run_task(self, fw_spec):
        vasp_cmd = env_chk(self['vasp_cmd'], fw_spec)
        handlers = [VaspErrorHandler(), MeshSymmetryErrorHandler(),
                    PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()]
        validators = [VasprunXMLValidator(), VaspFilesValidator()]

        c = Custodian(handlers, [VaspJob(vasp_cmd=vasp_cmd)], validators=validators, max_errors=5)
        c.run()


@explicit_serialize
class StaticFW(Firework):
    def __init__(self, structure, vasp_input_set, vasp_cmd, name, db_info, lammps_energy):
        """
        Standard static calculation Firework for a structure.

        Args:
            structure (Structure): Input structure
            vasp_input_set (VaspInputSet): input set to use
            vasp_cmd (str): Command to run vasp.
            name: the name of the workflow
            db_info: database credentials to store results
            lammps_energy: energy from LAMMPS run for this structure
        """

        t = list()
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd))
        t.append(AddToDbTask(force_thresh=float(0.05), energy_thresh=10.0,
                             db_info=db_info, lammps_energy=lammps_energy))
        super(StaticFW, self).__init__(t, name=name)


@explicit_serialize
class RunVaspCustodian(FiretaskBase):
    """
    Run VASP using custodian "on rails", fixes most runtime errors, uses default handlers
    and validators from custodian package

    Required params:
        vasp_cmd (str): the name of the full executable for running VASP
    """

    required_params = ['vasp_cmd']
    optional_params = []

    def run_task(self, fw_spec):
        vasp_cmd = env_chk(self['vasp_cmd'], fw_spec)
        handlers = [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
                    NonConvergingErrorHandler(), PotimErrorHandler(),
                    PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()]
        validators = [VasprunXMLValidator(), VaspFilesValidator()]

        c = Custodian(handlers, [VaspJob(vasp_cmd=vasp_cmd)], validators=validators, max_errors=5)
        c.run()


class VaspJob(Job):
    """
    A basic VASP job
    """

    def __init__(self, vasp_cmd):
        """
        Get/set variables for a simple VASP job
        Args:
            vasp_cmd (str): Command to run vasp
        """
        self.run_dir = os.getcwd()
        self.vasp_cmd = vasp_cmd
        self.std_out = 'vasp.out'  # compatible to handlers
        self.std_err = 'std_err.txt'  # compatible to handlers

    def setup(self):
        """
        We don't have to do anything here
        """
        pass

    def run(self):
        """
        Runs VASP

        Returns:
            open subprocess for monitoring by custodian
        """

        with open(self.std_out, 'w') as sout, open(self.std_err, 'w', buffering=1) as serr:
            p = subprocess.Popen(self.vasp_cmd.split(), stdout=sout, stderr=serr)
        return p

    def postprocess(self):
        """
        For now, gzip all files in the directory we ran VASP in
        """

        for file in os.listdir(self.run_dir):
            with open(file, 'rb') as fin:
                with gzip.open(file + '.gz', 'wb') as fout:
                    fout.write(fin.read())
            os.unlink(file)


@explicit_serialize
class AddToDbTask(FiretaskBase):
    """
    Task insert results into a database if energy and forces exceed specification

    Required:
        db_file (str): absolute path to file containing the database credentials
        force_thresh (float): Threshold for any force component above which the result is added to the training db
    """

    required_params = ['force_thresh', 'energy_thresh', 'db_info', 'lammps_energy']
    optional_params = []

    def run_task(self, fw_spec):
        """
        Does the work by connecting to db, parsing the results, checking the thresholds and adding the
        structure to the db if needed

        Args:
            fw_spec: fireworks specifics

        Returns:
            nothing

        """

        connection = None

        if 'ssl' in self['db_info']:
            if self['db_info']['ssl'].lower() == 'true':
                try:
                    connection = MongoClient(host=self['db_info']['host'], port=self['db_info']['port'],
                                             username=self['db_info']['user'], password=self['db_info']['password'],
                                             ssl=True, tlsCAFile=self['db_info']['ssl_ca_certs'],
                                             ssl_certfile=self['db_info']['ssl_certfile'])
                except ConnectionError:
                    raise ConnectionError('Mongodb connection failed')
            else:
                try:
                    connection = MongoClient(host=self['db_info']['host'], port=self['db_info']['port'],
                                             username=self['db_info']['user'], password=self['db_info']['password'],
                                             ssl=False)
                except ConnectionError:
                    raise ConnectionError('Mongodb connection failed')

        if connection is None:
            raise ConnectionAbortedError('Connection failure, check internal routines')

        db = connection[self['db_info']['database']]
        try:
            db.authenticate(self['db_info']['user'], self['db_info']['password'])
        except ConnectionRefusedError:
            raise ConnectionRefusedError('Mongodb authentication failed')
        collection = db[self['db_info']['structure_collection']]

        # get the directory we parse files in
        run_dir = os.getcwd()

        vrun = os.path.join(run_dir, 'vasprun.xml.gz')
        orun = os.path.join(run_dir, 'OUTCAR.gz')

        run = Vasprun(vrun)
        if not run.converged:
            return
        runo = Outcar(orun)
        atoms = aseread(vrun)
        xyz = ''
        file = io.StringIO()
        asewrite(filename=file, images=atoms, format='xyz')
        file.seek(0)
        for f in file:
            xyz += f
        file.close()

        forces = runo.read_table_pattern(
            header_pattern=r'\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+',
            row_pattern=r'\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)',
            footer_pattern=r'\s--+',
            postprocess=lambda x: float(x),
            last_one_only=True)

        dE = float((float(runo.final_energy) - float(self['lammps_energy'])) / run.final_structure.num_sites * 1000.0)

        if np.any(np.array(forces) > float(self['force_thresh'])) or dE > float(self['energy_thresh']):
            dft_data = dict()
            dft_data['xyz'] = xyz
            dft_data['PBE_54'] = run.potcar_symbols
            dft_data['parameters'] = run.parameters.as_dict()
            dft_data['free_energy'] = runo.final_energy  # this is the FREE energy, different from vasprun.xml in 6.+
            dft_data['final_structure'] = run.final_structure.as_dict()
            dft_data['lammps_energy'] = self['lammps_energy']
            data_name = 'Pt structure added {}  ||  automatic addition from PoValT'.format(
                datetime.datetime.now().strftime('%Y/%m/%d-%T'))
            collection.insert_one({'name': data_name, 'data': dft_data})


class VaspTasks:
    """
    DEPRECATED Class -- not in use here, kept for future integration of a automatic generator -- DO NOT USE
    """
    def __init__(self):
        """
        init
        """
        pass

    @staticmethod
    def get_relax_wf(structure, atom_type=None, structure_name=None, vasp_cmd=None, ncore=None, metadata=None):

        if structure not in ['fcc', 'bcc', 'sc', 'hcp']:
            raise ValueError('Structure can only be one of [fcc, bcc, sc, hcp] here')
        if atom_type is None:
            raise ValueError('Type of atom has to be specified')
        if structure_name is None:
            raise ValueError('Name of the structure needs to be consistently defined')
        if vasp_cmd is None:
            raise ValueError('vasp_cmd needs to be set')
        if metadata is None:
            raise ValueError('Metadata con not be None, it is used to uniquely define the workflows')
        if ncore is None:
            raise ValueError('Please set ncore parameter (move to CFG file?')

        if structure == 'fcc':
            lat = Lattice.from_parameters(2.4, 2.4, 2.4, 60, 60, 60)
            atoms = [atom_type]
            pos = [[0.0, 0.0, 0.0]]
            bulk = Structure(lat, atoms, pos)
            kpts = Kpoints.gamma_automatic(kpts=[8, 8, 8], shift=(0, 0, 0)).as_dict()
        elif structure == 'bcc':
            lat = Lattice.from_parameters(2.6, 2.6, 2.6, 109.471, 109.471, 109.471)
            atoms = [atom_type]
            pos = [[0.0, 0.0, 0.0]]
            bulk = Structure(lat, atoms, pos)
            kpts = Kpoints.gamma_automatic(kpts=[8, 8, 8], shift=(0, 0, 0)).as_dict()
        elif structure == 'sc':
            lat = Lattice.from_parameters(2.6, 2.6, 2.6, 90, 90, 90)
            atoms = [atom_type]
            pos = [[0.0, 0.0, 0.0]]
            bulk = Structure(lat, atoms, pos)
            kpts = Kpoints.gamma_automatic(kpts=[8, 8, 8], shift=(0, 0, 0)).as_dict()
        elif structure == 'hcp':
            lat = Lattice.from_parameters(2.5, 2.5, 4.0, 90, 90, 120)
            atoms = [atom_type]
            pos = [[0.0, 0.0, 0.0], [0.3333, 0.6666, 0.5]]
            bulk = Structure(lat, atoms, pos)
            kpts = Kpoints.gamma_automatic(kpts=[8, 8, 4], shift=(0, 0, 0)).as_dict()
        else:
            raise KeyboardInterrupt('This was impossible but you managed to reach it...')

        vis = MPRelaxSet(bulk)
        v = vis.as_dict()
        v.update({"user_kpoints_settings": kpts})
        vis_static = vis.__class__.from_dict(v)

        fws = [OptimizeFW(structure=bulk, vasp_input_set=vis_static, vasp_cmd=vasp_cmd,
                          db_file='', name="{} -- static".format(datetime.datetime.now().strftime('%Y/%m/%d-%T')))]

        incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': int(ncore), 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
                     'ALGO': 'Fast', 'AMIN': 0.01, 'NELM': 200, 'LAECHG': 'False', 'ISIF': 1}

        wfname = "{}: static run".format(structure_name)

        static_wf = Workflow(fws, name=wfname, metadata=metadata)
        return add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
