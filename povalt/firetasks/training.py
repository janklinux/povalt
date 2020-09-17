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

# import os
import abc
from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from custodian import Custodian
from povalt.training.training import TrainJob
from povalt.lammps.lammps import LammpsJob


class TrainBase(FiretaskBase):
    """
    Base class to train potentials
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
        # if fw_spec['al_task'] is not None:
        #     os.chdir('cd {}'.format(fw_spec['al_task']['base_dir']))
        #     os.system('qlaunch -q {} rapidfire --nlaunches {}'.format(os.path.join(fw_spec['al_task']['base_dir'],
        #                                                                            'my_queue.yaml'),
        #                                                               str(fw_spec['al_task']['num_launches'])))
        # return FWAction(update_spec={'potential_info': job.get_potential_info()})


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
    Base class to run LAMMPS
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
    required_params = ['lammps_params', 'db_info']
    optional_params = []

    def get_job(self, fw_spec):
        return LammpsJob(lammps_params=self['lammps_params'], db_info=self['db_info'], fw_spec=fw_spec)

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)
