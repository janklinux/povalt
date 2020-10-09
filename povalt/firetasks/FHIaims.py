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
import gzip
import glob
import subprocess
from pymatgen import Structure
from custodian import Custodian
from custodian.custodian import Job
from custodian.fhi_aims.handlers import AimsRelaxHandler, FrozenJobErrorHandler
from custodian.fhi_aims.validators import AimsConvergedValidator
from fireworks import FWAction, FiretaskBase, Firework, Workflow, explicit_serialize


class OptimizeFW(Firework):

    def __init__(self, control, structure, basis_set, basis_dir, aims_cmd, rerun_metadata,
                 name='FHIaims run', parents=None, aims_output='run', **kwargs):
        """
        Optimize the given structure.

        Args:
            control: control.in as list of lines
            structure (Structure): pymatgen input structure
            name (str): Name for the Firework.
            aims_cmd (str): Command to run
            aims_output (str): name of output filename
            rerun_metadata: metadata for the tight rerun
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        t = list()
        t.append(RunAimsCustodian(aims_cmd=aims_cmd, control=control, structure=structure,
                                  basis_set=basis_set, basis_dir=basis_dir, aims_output=aims_output,
                                  rerun_metadata=rerun_metadata))

        super(OptimizeFW, self).__init__(t, parents=parents, name="{}-{}".
                                         format('{} - {}'.join(structure.composition.reduced_formula), name), **kwargs)


class AimsJob(Job):
    """
    A basic job. Just runs whatever is in the directory. But conceivably
    can be a complex processing of inputs etc. with initialization.
    """

    def __init__(self, aims_cmd, control, structure, basis_set, basis_dir, rerun_metadata,
                 output_file='run', stderr_file='std_err.txt'):
        """
        This constructor is necessarily complex due to the need for
        flexibility. For standard kinds of runs, it's often better to use one
        of the static constructors. The defaults are usually fine too.

        Args:
            aims_cmd (str): Command to run aims as a list of args. For example,
                if you are using mpirun, it can be something like
                ["mpirun", "aims"]
            control: control.in as list of lines
            structure: pymatgen structure object
            basis_set: light or tight
            basis_dir: directory containing the light/tight directory of basis files
            output_file (str): Name of file to direct standard out to.
                Defaults to "vasp.out".
            stderr_file (str): Name of file to direct standard error to.
                Defaults to "std_err.txt".
        """
        self.aims_cmd = aims_cmd
        self.output_file = output_file
        self.stderr_file = stderr_file
        self.run_dir = os.getcwd()
        self.control = control
        self.structure = structure
        if basis_set.lower() not in ['light', 'tight']:
            raise ValueError('basis set can be only light or tight for now...')
        self.basis_set = basis_set
        self.basis_dir = basis_dir
        self.rerun_metadata = rerun_metadata

    def setup(self):
        """
        Performs initial setup for AimsJob
        """

        base = os.path.join(self.basis_dir, self.basis_set)
        basis = []
        elements = []
        for s in self.structure.types_of_species:
            if str(s) not in elements:
                elements.append(str(s))

        for el in elements:
            basis.append(glob.glob(base+'/*'+el+'*')[0])

        with open('control.in', 'wt') as f:
            for line in self.control:
                f.write(line)
            for fname in basis:
                with open(fname, 'r') as fin:
                    f.write(fin.read())

        self.structure.to(filename='geometry.in', fmt='aims')

    def run(self):
        """
        Performs the actual run

        Returns:
            subprocess.Popen for monitoring
        """

        with open(self.output_file, 'w') as f_std, open(self.stderr_file, "w", buffering=1) as f_err:
            p = subprocess.Popen(self.aims_cmd.split(), stdout=f_std, stderr=f_err)
        return p

    def postprocess(self):
        """
        Postprocessing is adding tight run if previous was light and gzip for now
        """
        action = []

        if self.basis_set.lower() == 'light':
            tight_wf = Workflow(OptimizeFW(aims_cmd=self.aims_cmd, control=self.control,
                                           structure=Structure.from_file('geometry.in.next_step'),
                                           basis_set='tight', basis_dir=self.basis_dir,
                                           name='tight rerun', aims_output='run.tight',
                                           rerun_metadata=self.rerun_metadata),
                                metadata=self.rerun_metadata)
            action = FWAction(additions=tight_wf)

        for file in os.listdir(self.run_dir):
            with open(file, 'rb') as fin:
                with gzip.open(file + '.gz', 'wb') as fout:
                    fout.write(fin.read())
            os.unlink(file)

        return action


@explicit_serialize
class RunAimsCustodian(FiretaskBase):
    """
    Run FHIaims

    Required params:
        aims_cmd (str): the name of the full executable for running VASP. Supports env_chk.

    Optional params:
        none for now
    """
    required_params = ['aims_cmd', 'control', 'structure', 'basis_set', 'basis_dir']
    optional_params = ['aims_output', 'rerun_metadata']

    def run_task(self, fw_spec):
        job = [AimsJob(aims_cmd=self['aims_cmd'], control=self['control'], structure=self['structure'],
                       basis_set=self['basis_set'], basis_dir=self['basis_dir'], output_file=self['aims_output'],
                       rerun_metadata=self['rerun_metadata'])]
        validators = [AimsConvergedValidator()]
        c = Custodian(handlers=[AimsRelaxHandler(), FrozenJobErrorHandler()],
                      jobs=job, validators=validators, max_errors=3)
        c.run()
