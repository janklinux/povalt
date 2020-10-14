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

import abc
from fireworks import FiretaskBase, FWAction, Workflow, Firework
from fireworks.utilities.fw_utilities import explicit_serialize
from custodian import Custodian
from custodian.fhi_aims.handlers import AimsRelaxHandler, FrozenJobErrorHandler
from custodian.fhi_aims.validators import AimsConvergedValidator
from povalt.training.training import TrainJob
from povalt.lammps.lammps import LammpsJob
from povalt.firetasks.FHIaims import AimsJob


class TrainBase(FiretaskBase):
    """
    Base class to train potentials, we inherit from it
    """
    @abc.abstractmethod
    def get_job(self):
        pass

    @abc.abstractmethod
    def get_handlers(self):
        pass

    @abc.abstractmethod
    def get_validators(self):
        pass

    def run_task(self, fw_spec):
        job = self.get_job()
        c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)
        c.run()


@explicit_serialize
class PotentialTraining(TrainBase):
    """
    Class to train a potential
    """
    required_params = ['train_params', 'for_validation', 'db_info']
    optional_params = ['al_info']

    def get_job(self):
        return TrainJob(train_params=self['train_params'], for_validation=self['for_validation'],
                        db_info=self['db_info'])

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


class LammpsBase(FiretaskBase):
    """
    Base class to run LAMMPS, for inheritance
    """
    @abc.abstractmethod
    def get_job(self, fw_spec):
        pass

    @abc.abstractmethod
    def get_handlers(self):
        pass

    @abc.abstractmethod
    def get_validators(self):
        pass

    def run_task(self, fw_spec):
        job = self.get_job(fw_spec)
        c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)
        c.run()
        return FWAction(additions=job.get_vasp_static_dft(job.get_lammps_energy()))


@explicit_serialize
class Lammps(LammpsBase):
    """
    Class to run LAMMPS
    """
    required_params = ['lammps_params', 'db_info', 'is_slab']
    optional_params = []

    def get_job(self, fw_spec):
        return LammpsJob(lammps_params=self['lammps_params'], db_info=self['db_info'],
                         is_slab=self['is_slab'], fw_spec=fw_spec)

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


class AimsBase(FiretaskBase):
    """
    Base class to run FHIaims, for inheritance
    """
    @abc.abstractmethod
    def get_job(self, fw_spec):
        pass

    @abc.abstractmethod
    def get_handlers(self):
        pass

    @abc.abstractmethod
    def get_validators(self):
        pass

    def run_task(self, fw_spec):
        job = self.get_job(fw_spec)
        c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)
        c.run()
        structure, params = job.get_relaxed_structure()
        if structure is not None:
            add_wf = Workflow([Firework(
                [Aims(aims_cmd=params['aims_cmd'], control=params['control'], structure=structure,
                      basis_set='tight', basis_dir=params['basis_dir'], rerun_metadata=params['metadata'])])],
                metadata=params['metadata'], name='automatic tight run')
            return FWAction(additions=add_wf)


@explicit_serialize
class Aims(AimsBase):
    """
    Class to run FHIaims
    """
    required_params = ['aims_cmd', 'control', 'structure', 'basis_set', 'basis_dir', 'rerun_metadata']
    optional_params = ['output_file', 'stderr_file']

    def get_job(self, fw_spec):
        return AimsJob(aims_cmd=self['aims_cmd'], control=self['control'], structure=self['structure'],
                       basis_set=self['basis_set'], basis_dir=self['basis_dir'], metadata=self['rerun_metadata'])

    def get_validators(self):
        return [AimsConvergedValidator()]

    def get_handlers(self):
        return [AimsRelaxHandler(), FrozenJobErrorHandler()]

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)
