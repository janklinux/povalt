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
import time
import subprocess
from povalt.helpers import find_binary


class TrainPotential:
    """
    Base class for training a potential from existing database entries
    """
    def __init__(self, atoms_filename, order, compact_clusters, nb_cutoff, n_sparse, nb_covariance_type, nb_delta,
                 theta_uniform, nb_sparse_method, l_max, n_max, atom_sigma, zeta, soap_cutoff, central_weight,
                 config_type_n_sparse, soap_delta, f0, soap_covariance_type, soap_sparse_method, default_sigma,
                 config_type_sigma, energy_parameter_name, force_parameter_name, force_mask_parameter_name,
                 sparse_jitter, do_copy_at_file, sparse_separate_file, gp_file):
        """
        Set up instance with all parameters for fitting
        Args:
            atoms_filename:
            order:
            compact_clusters:
            nb_cutoff:
            n_sparse:
            nb_covariance_type:
            nb_delta:
            theta_uniform:
            nb_sparse_method:
            l_max:
            n_max:
            atom_sigma:
            zeta:
            soap_cutoff:
            central_weight:
            config_type_n_sparse:
            soap_delta:
            f0:
            soap_covariance_type:
            soap_sparse_method:
            default_sigma:
            config_type_sigma:
            energy_parameter_name:
            force_parameter_name:
            force_mask_parameter_name:
            sparse_jitter:
            do_copy_at_file:
            sparse_separate_file:
            gp_file:
        """
        self.atoms_filename = str(atoms_filename)
        self.order = str(order)
        self.compact_clusters = str(compact_clusters)
        self.nb_cutoff = str(nb_cutoff)
        self.n_sparse = str(n_sparse)
        self.nb_covariance_type = str(nb_covariance_type)
        self.nb_delta = str(nb_delta)
        self.theta_uniform = str(theta_uniform)
        self.nb_sparse_method = str(nb_sparse_method)
        self.l_max = str(l_max)
        self.n_max = str(n_max)
        self.atom_sigma = str(atom_sigma)
        self.zeta = str(zeta)
        self.soap_cutoff = str(soap_cutoff)
        self.central_weight = str(central_weight)
        self.config_type_n_sparse = str(config_type_n_sparse)
        self.soap_delta = str(soap_delta)
        self.f0 = str(f0)
        self.soap_covariance_type = str(soap_covariance_type)
        self.soap_sparse_method = str(soap_sparse_method)
        self.default_sigma = str(default_sigma)
        self.config_type_sigma = str(config_type_sigma)
        self.energy_parameter_name = str(energy_parameter_name)
        self.force_parameter_name = str(force_parameter_name)
        self.force_mask_parameter_name = str(force_mask_parameter_name)
        self.sparse_jitter = str(sparse_jitter)
        self.do_copy_at_file = str(do_copy_at_file)
        self.sparse_separate_file = str(sparse_separate_file)
        self.gp_file = str(gp_file)
        self.err_chk_time = 15  # check every N seconds during gap_fit

    def train(self):
        """
        Returns: nothing, writes potential to a file
        """

        bin_path = find_binary('gap_fit').strip()
        arg_string = ' atoms_filename=' + self.atoms_filename + ' gap = { distance_Nb order=' + self.order + \
            ' compact_clusters=' + self.compact_clusters + ' cutoff=' + self.nb_cutoff + \
            ' n_sparse=' + self.n_sparse + ' covariance_type=' + self.nb_covariance_type + ' delta=' + self.nb_delta + \
            ' theta_uniform=' + self.theta_uniform + ' sparse_method=' + self.nb_sparse_method + ' : ' + \
            ' soap l_max=' + self.l_max + ' n_max=' + self.n_max + ' atom_sigma=' + self.atom_sigma + \
            ' zeta=' + self.zeta + ' cutoff=' + self.soap_cutoff + ' central_weight=' + self.central_weight + \
            ' config_type_n_sparse=' + self.config_type_n_sparse + ' delta=' + self.soap_delta + \
            ' f0=' + self.f0 + ' covariance_type=' + self.soap_covariance_type + \
            ' sparse_method=' + self.soap_sparse_method + ' } ' + \
            ' default_sigma=' + self.default_sigma + ' config_type_sigma=' + self.config_type_sigma + \
            ' energy_parameter_name=' + self.energy_parameter_name + \
            ' force_parameter_name=' + self.force_parameter_name + \
            ' force_mask_parameter_name=' + self.force_mask_parameter_name + \
            ' sparse_jitter=' + self.sparse_jitter + ' do_copy_at_file=' + self.do_copy_at_file + \
            ' sparse_separate_file=' + self.sparse_separate_file + ' gp_file=' + self.gp_file

        cmd = bin_path + arg_string

        jobdir = os.getcwd()

        sout = open(os.path.join(jobdir, 'fit_output'), 'w')
        serr = open(os.path.join(jobdir, 'fit_error'), 'w')

        p = subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)

        # wait for process to end and periodically check for errors
        last_check = time.time()
        while p.poll() is None:
            time.sleep(30)
            if time.time() - last_check > self.err_chk_time:
                last_check = time.time()
                if self.found_error(os.path.join(jobdir, 'fit_error')):
                    p.kill()
                    raise FittingError('Error during gap_fit, check file fit_error')

        # close output files
        sout.close()
        serr.close()

    @staticmethod
    def found_error(filename):
        """
        Function to check for specific errors during fitting.

        Args:
            filename: filename to check for string pattern

        Returns:
            True if string in filename, False otherwise
        """

        patterns = ['Cannot allocate memory']

        with open(filename, 'r') as f:
            for line in f:
                for pat in patterns:
                    if pat in line:
                        return True

        return False


class FittingError(Exception):
    """
    Error class for fitting and training errors
    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
