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
from fireworks import LaunchPad
from povalt.firetasks.wf_generators import potential_trainer


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='fw_run', username='jank', password='b@sf_mongo',
                 ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)

train_params = {'atoms_filename': 'complete.xyz',
                'order': 2,
                'compact_clusters': 'T',
                'nb_cutoff': 5.0,
                'n_sparse': 20,
                'nb_covariance_type': 'ard_se',
                'nb_delta': 0.5,
                'theta_uniform': 1.0,
                'nb_sparse_method': 'uniform',
                'l_max': 8,
                'n_max': 8,
                'atom_sigma': 0.5,
                'zeta': 4,
                'soap_cutoff': 5.0,
                'central_weight': 1.0,
                'config_type_n_sparse': '{fcc:500:bcc:500:hcp:500:sc:500:slab:500:cluster:0}',
                'soap_delta': 0.1,
                'f0': 0.0,
                'soap_covariance_type': 'dot_product',
                'soap_sparse_method': 'cur_points',
                'default_sigma': '{0.002 0.2 0.2 0.2}',
                'config_type_sigma': '{dimer:0.002:0.2:0.2:0.2:'
                                     'fcc:0.002:0.2:0.2:0.2:'
                                     'bcc:0.002:0.2:0.2:0.2:'
                                     'hcp:0.002:0.2:0.2:0.2:'
                                     'sc:0.002:0.2:0.2:0.2:'
                                     'slab:0.002:0.2:0.2:0.2:'
                                     'cluster:0.002:0.2:0.2:0.2:}',
                'energy_parameter_name': 'free_energy',
                'force_parameter_name': 'forces',
                'force_mask_parameter_name': 'force_mask',
                'sparse_jitter': 1E-8,
                'do_copy_at_file': 'F',
                'sparse_separate_file': 'T',
                'gp_file': 'Pt_test.xml',
                'gap_cmd': 'gap_fit',
                'mpi_cmd': None,
                'mpi_procs': 4,
                'omp_threads': 6
                }

wf = potential_trainer(train_params=train_params)

print(wf)
lpad.add_wf(wf)
