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
import subprocess
from povalt.helpers import find_binary
from custodian.custodian import Job


class TrainJob(Job):
    """
    Training Job for the potential
    """
    def __init__(self, train_params):
        """
        Class init

        Args:
            train_params: all parameters
        """

        self.train_params = train_params

    def setup(self):
        pass

    def run(self):
        """
        Runs the training routine

        Returns:

        """
        print(self.train_params)

        if int(self.train_params['omp_threads']) > 1:
            os.environ['OMP_NUM_THREADS'] = int(self.train_params['omp_threads'])
        else:
            os.environ['OMP_NUM_THREADS'] = int(1)

        if self.train_params['mpi_cmd'] is not None:
            if self.train_params['mpi_procs'] is None:
               raise ValueError('Running MPI you have to set mpi_procs')
            cmd = find_binary(str(self.train_params['mpi_cmd'].strip())).strip() + \
                  ' -n {} '.format(self.train_params['mpi_procs'])
        else:
            cmd = ''

        arg_list =

        cmd += find_binary(self.train_params['gap_cmd']).strip()


        try:
            with open('std_err', 'w') as serr, open('std_out', 'w') as sout:
                subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)
        except FileNotFoundError:
            raise FileNotFoundError('fille not funde')
        finally:
            print('I ran it all the way')

    def postprocess(self):
        pass
