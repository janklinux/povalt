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
        self.run_dir = os.getcwd()
        self.train_params = train_params
        self.potential_name = ' *** FILE NOT FOUND *** '
        self.potential_label = ' *** FILE NOT FOUND *** '

    def setup(self):
        pass

    def run(self):
        """
        Runs the training routine

        Returns:

        """
        for item in self.train_params:
            if self.train_params[item] is not None:
                self.train_params[item] = str(self.train_params[item])

        if int(self.train_params['omp_threads']) > 1:
            os.environ['OMP_NUM_THREADS'] = str(self.train_params['omp_threads'])
        else:
            os.environ['OMP_NUM_THREADS'] = str(1)

        if self.train_params['mpi_cmd'] is not None:
            if self.train_params['mpi_procs'] is None:
               raise ValueError('Running MPI you have to set mpi_procs')
            cmd = find_binary(str(self.train_params['mpi_cmd'].strip())).strip() + \
                  ' -n {} '.format(self.train_params['mpi_procs'])
        else:
            cmd = ''

        arg_list = ' atoms_filename=' + self.train_params['atoms_filename'] + \
                   ' gap = {' + \
                   ' distance_Nb' + \
                   ' order=' + self.train_params['order'] + \
                   ' compact_clusters=' + self.train_params['compact_clusters'] + \
                   ' cutoff=' + self.train_params['nb_cutoff'] + \
                   ' n_sparse=' + self.train_params['n_sparse'] + \
                   ' covariance_type=' + self.train_params['nb_covariance_type'] + \
                   ' delta=' + self.train_params['nb_delta'] + \
                   ' theta_uniform=' + self.train_params['theta_uniform'] + \
                   ' nb_sparse_method=' + self.train_params['nb_sparse_method'] + \
                   ' : soap' + \
                   ' l_max=' + self.train_params['l_max'] + \
                   ' n_max=' + self.train_params['n_max'] + \
                   ' atom_sigma=' + self.train_params['atom_sigma'] + \
                   ' zeta=' + self.train_params['zeta'] + \
                   ' cutoff=' + self.train_params['soap_cutoff'] + \
                   ' central_weight=' + self.train_params['central_weight'] + \
                   ' config_type_n_sparse=' + self.train_params['config_type_n_sparse'] + \
                   ' delta=' + self.train_params['soap_delta'] + \
                   ' f0=' + self.train_params['f0'] + \
                   ' covariance_type=' + self.train_params['soap_covariance_type'] + \
                   ' sparse_method=' + self.train_params['soap_sparse_method'] + \
                   ' }' + \
                   ' default_sigma=' + self.train_params['default_sigma'] + \
                   ' config_type_sigma=' + self.train_params['config_type_sigma'] + \
                   ' energy_parameter_name=' + self.train_params['energy_parameter_name'] + \
                   ' force_parameter_name=' + self.train_params['force_parameter_name'] + \
                   ' force_mask_parameter_name=' + self.train_params['force_mask_parameter_name'] + \
                   ' sparse_jitter=' + self.train_params['sparse_jitter'] + \
                   ' do_copy_at_file=' + self.train_params['do_copy_at_file'] + \
                   ' sparse_separate_file=' + self.train_params['sparse_separate_file'] + \
                   ' gp_file=' + self.train_params['gp_file']

        cmd += find_binary(self.train_params['gap_cmd']).strip() + arg_list

        try:
            with open('std_err', 'w') as serr, open('std_out', 'w') as sout:
                p = subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)
        except FileNotFoundError:
            raise FileNotFoundError('Command execution failed, check std_err')
        finally:
            os.environ['OMP_NUM_THREADS'] = str(1)
            # pass  # print('I ran it all the way')

        return p

    def postprocess(self):
        pass

    def get_potential_info(self):
        pot_file = []
        for file in os.listdir(self.run_dir):
            if file.startswith(self.train_params['gp_file']):
                pot_file.append(file)
        return {'files': pot_file, 'path': self.run_dir, 'label': self.train_params['gp_file']}
