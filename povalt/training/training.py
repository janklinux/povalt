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
import subprocess
from pymongo import MongoClient
from povalt.helpers import find_binary
from custodian.custodian import Job


class TrainJob(Job):
    """
    Training Job for the potential
    """
    def __init__(self, train_params, db_info):
        """
        Class init

        Args:
            train_params: all parameters
            db_info: db info to store the potential
        """
        self.run_dir = os.getcwd()
        self.db_info = db_info
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
            ' distance_2b' + \
            ' Z1=' + self.train_params['2b_z1'] + \
            ' Z2=' + self.train_params['2b_z2'] + \
            ' cutoff=' + self.train_params['2b_cutoff'] + \
            ' n_sparse=' + self.train_params['2b_n_sparse'] + \
            ' covariance_type=' + self.train_params['2b_covariance_type'] + \
            ' delta=' + self.train_params['2b_delta'] + \
            ' theta_uniform=' + self.train_params['2b_theta_uniform'] + \
            ' sparse_method=' + self.train_params['2b_sparse_method'] + \
            ' :' + \
            ' angle_3b' + \
            ' Z_center=' + self.train_params['3b_z_center'] + \
            ' Z1=' + self.train_params['3b_z1'] + \
            ' Z2=' + self.train_params['3b_z2'] + \
            ' cutoff=' + self.train_params['3b_cutoff'] + \
            ' config_type_n_sparse=' + self.train_params['3b_config_type_n_sparse'] + \
            ' covariance_type=' + self.train_params['3b_covariance_type'] + \
            ' delta=' + self.train_params['3b_delta'] + \
            ' theta_uniform=' + self.train_params['3b_theta_uniform'] + \
            ' sparse_method=' + self.train_params['3b_sparse_method'] + \
            ' :' + \
            ' soap_turbo' + \
            ' l_max=' + self.train_params['soap_l_max'] + \
            ' alpha_max=' + self.train_params['soap_alpha_max'] + \
            ' atom_sigma_r=' + self.train_params['soap_atom_sigma_r'] + \
            ' atom_sigma_t=' + self.train_params['soap_atom_sigma_t'] + \
            ' atom_sigma_r_scaling=' + self.train_params['soap_atom_sigma_r_scaling'] + \
            ' atom_sigma_t_scaling=' + self.train_params['soap_atom_sigma_t_scaling'] + \
            ' zeta=' + self.train_params['soap_zeta'] + \
            ' rcut_hard=' + self.train_params['soap_rcuthard'] + \
            ' rcut_soft=' + self.train_params['soap_rcutsoft'] + \
            ' basis=' + self.train_params['soap_basis'] + \
            ' scaling_mode=' + self.train_params['soap_scaling_mode'] + \
            ' amplitude_scaling=' + self.train_params['soap_amplitude_scaling'] + \
            ' n_species=' + self.train_params['soap_n_species'] + \
            ' species_Z=' + self.train_params['soap_species_Z'] + \
            ' radial_enhancement=' + self.train_params['soap_radial_enhancement'] + \
            ' compress_file=' + self.train_params['soap_compress_file'] + \
            ' central_weight=' + self.train_params['soap_central_weight'] + \
            ' config_type_n_sparse=' + self.train_params['soap_config_type_n_sparse'] + \
            ' delta=' + self.train_params['soap_delta'] + \
            ' f0=' + self.train_params['soap_f0'] + \
            ' covariance_type=' + self.train_params['soap_covariance_type'] + \
            ' sparse_method=' + self.train_params['soap_sparse_method'] + \
            ' }' + \
            ' default_sigma=' + self.train_params['default_sigma'] + \
            ' config_type_sigma=' + self.train_params['config_type_sigma'] + \
            ' energy_parameter_name=' + self.train_params['energy_parameter_name'] + \
            ' force_parameter_name=' + self.train_params['force_parameter_name'] + \
            ' sparse_jitter=' + self.train_params['sparse_jitter'] + \
            ' e0=' + self.train_params['e0'] + \
            ' do_copy_at_file=' + self.train_params['do_copy_at_file'] + \
            ' sparse_separate_file=' + self.train_params['sparse_separate_file'] + \
            ' gp_file=' + self.train_params['gp_file']

        cmd += find_binary(self.train_params['gap_cmd']).strip() + arg_list

        print(cmd)

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

        pot_file = {}  # dict of filename: data
        for file in os.listdir(self.run_dir):
            if file.startswith(self.train_params['gp_file']):
                with open(file, 'r') as f:
                    pot_file[re.sub('.', ':', file)] = f.readlines()

        collection.insert_one(pot_file)

    def get_potential_info(self):
        pot_file = []
        for file in os.listdir(self.run_dir):
            if file.startswith(self.train_params['gp_file']):
                pot_file.append(file)
        return {'files': pot_file, 'path': self.run_dir, 'label': self.train_params['gp_file']}
