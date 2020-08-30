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

import subprocess
from custodian.custodian import Job


class TrainJob(Job):
    """
    Training Job for the potential
    """
    def __init__(self, all_params):
        """
        Class init

        Args:
            all_params: all parameters
        """

        self.all_params = all_params

    def setup(self):
        pass

    def run(self):
        """
        Runs the training routine

        Returns:

        """
        print(self.all_params)
        cmd = 'ls -al'
        try:
            with open('std_err', 'w') as serr, open('std_out', 'w') as sout:
                subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)
        except FileNotFoundError:
            raise FileNotFoundError('fille not funde')
        finally:
            print('I ran it all the way')

    def postprocess(self):
        pass
