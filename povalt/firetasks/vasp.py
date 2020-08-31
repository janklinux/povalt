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
import datetime
import subprocess
import numpy as np
from povalt.helpers import env_chk
from fireworks import Workflow, Firework, FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from pymatgen.io.vasp import Kpoints, Outcar
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.core.structure import Structure, Lattice
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from atomate.vasp.powerups import add_modify_incar
from custodian import Custodian
from custodian.custodian import Job
from custodian.vasp.handlers import VaspErrorHandler, MeshSymmetryErrorHandler, UnconvergedErrorHandler, \
    NonConvergingErrorHandler, PotimErrorHandler, PositiveEnergyErrorHandler, FrozenJobErrorHandler, StdErrHandler


@explicit_serialize
class StaticFW(Firework):
    def __init__(self, structure=None, vasp_input_set=None, vasp_cmd=None):
        """
        Standard static calculation Firework from a structure.

        Args:
            structure (Structure): Input structure
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_cmd (str): Command to run vasp.
        """

        t = list()
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd))
        super(StaticFW, self).__init__(t)


@explicit_serialize
class RunVaspCustodian(FiretaskBase):
    """
    Run VASP using custodian "on rails", fixes most runtime errors.

    Required params:
        vasp_cmd (str): the name of the full executable for running VASP.
    """
    required_params = ['vasp_cmd']
    optional_params = []

    def run_task(self, fw_spec):
        vasp_cmd = env_chk(self['vasp_cmd'], fw_spec)
        handlers = [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
                    NonConvergingErrorHandler(), PotimErrorHandler(),
                    PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()]
        validators = []

        c = Custodian(handlers, [VaspJob(vasp_cmd=vasp_cmd)], validators=validators, max_errors=3)
        c.run()


class VaspJob(Job):
    """
    A basic vasp job.
    """
    def __init__(self, vasp_cmd):
        """
        Simple Job for VASP
        Args:
            vasp_cmd (str): Command to run vasp
        """
        self.run_dir = os.getcwd()
        self.vasp_cmd = vasp_cmd
        self.std_out = 'vasp.out'  # compatible to handlers
        self.std_err = 'std_err.txt'  # compatible to handlers

    def setup(self):
        pass

    def run(self):
        """
        Runs VASP

        Returns:
            (subprocess.Popen) for monitoring.
        """
        with open(self.std_out, 'w') as sout, open(self.std_err, 'w', buffering=1) as serr:
            p = subprocess.Popen(self.vasp_cmd.split(), stdout=sout, stderr=serr)
        return p

    def postprocess(self):
        forces = Outcar(os.path.join(self.run_dir, 'OUTCAR')).read_table_pattern(
            header_pattern=r'\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+',
            row_pattern=r'\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)',
            footer_pattern=r'\s--+',
            postprocess=lambda x: float(x),
            last_one_only=True)
        print(np.array(forces))


class VaspTasks:
    """
    Class
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
                     'ALGO': 'Fast', 'AMIN': 0.01, 'NELM': 250, 'LAECHG': 'False', 'ISIF': 1}

        wfname = "{}: static run".format(structure_name)

        static_wf = Workflow(fws, name=wfname, metadata=metadata)
        return add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
