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
from fireworks import FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from custodian import Custodian
from povalt.training.training import TrainJob


class TrainBase(FiretaskBase):
    """
    Base class to train potentials
    """
    @abc.abstractmethod
    def get_job(self):
        """
        Creates custodian job for the task

        Returns:
            a custodian job
        """
        pass

    @abc.abstractmethod
    def get_handlers(self):
        """
        Returns handlers of job for the task

        Returns:
            job error handlers
        """
        pass

    @abc.abstractmethod
    def get_validators(self):
        """
        Returns the validators for the job

        Returns:
            job validators
        """
        pass

    def run_task(self, fw_spec):
        """
        Runs the task
        Args:
            fw_spec: firework specifications

        Returns:

        """

        job = self.get_job()

        c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)

        c.run()


@explicit_serialize
class PotentialTraining(TrainBase):
    """
    Class to train a potential
    """
    required_params = ['my_params']
    optional_params = ['others']

    def get_job(self):
        return TrainJob(all_params=['this be', 'alist'])

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)
