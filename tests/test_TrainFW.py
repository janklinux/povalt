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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
from ase.io import read
from fireworks import LaunchPad, Workflow
from pymatgen.io.vasp import Xdatcar
from povalt.firetasks.wf_generators import train_potential, \
    run_lammps, train_and_run_multiple_lammps
from pymatgen.io.ase import AseAtomsAdaptor


# ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
# cl_file = os.path.expanduser('~/ssl/numphys/client.pem')

lpad = LaunchPad.auto_load()
# lpad = LaunchPad(host='numphys.org', port=27017, name='fw_run', username='jank', password='b@sf_mongo',
#                  ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)

train_params_3body = {
    'do_3_body': True,
    'atoms_filename': '/users/kloppej1/scratch/jank/fireworks/complete.xyz',
    '2b_z1': 78,
    '2b_z2': 78,
    '2b_cutoff': 4.7,
    '2b_n_sparse': 15,
    '2b_covariance_type': 'ard_se',
    '2b_delta': 0.5,
    '2b_theta_uniform': 1.0,
    '2b_sparse_method': 'uniform',
    '3b_z_center': 78,
    '3b_z1': 78,
    '3b_z2': 78,
    '3b_cutoff': 4.0,
    '3b_config_type_n_sparse': '{bcc:250:fcc:250:hcp:250:sc:250:slab:250:cluster:250}',
    '3b_covariance_type': 'pp',
    '3b_delta': 0.003,
    '3b_theta_uniform': 4.0,
    '3b_sparse_method': 'uniform',
    'soap_l_max': 8,
    'soap_alpha_max': '{{8}}',
    'soap_atom_sigma_r': '{{0.5}}',
    'soap_atom_sigma_t': '{{0.5}}',
    'soap_atom_sigma_r_scaling': '{{0.0}}',
    'soap_atom_sigma_t_scaling': '{{0.0}}',
    'soap_zeta': 6,
    'soap_rcuthard': 5.7,
    'soap_rcutsoft': 4.2,
    'soap_basis': 'poly3gauss',
    'soap_scaling_mode': 'polynomial',
    'soap_amplitude_scaling': '{{1.0}}',
    'soap_n_species': 1,
    'soap_species_Z': '{78}',
    'soap_radial_enhancement': '{{1}}',
    'soap_compress_file': '/users/kloppej1/scratch/jank/fireworks/compress.dat',
    'soap_central_weight': 1.0,
    'soap_config_type_n_sparse': '{bcc:250:fcc:250:hcp:250:sc:250:slab:250:cluster:250}',
    'soap_delta': 0.1,
    'soap_f0': 0.0,
    'soap_covariance_type': 'dot_product',
    'soap_sparse_method': 'cur_points',
    'default_sigma': '{0.002 0.2 0.2 0.2}',
    'config_type_sigma': '{bcc:0.002:0.2:0.2:0.2:fcc:0.002:0.2:0.2:0.2:'
                         'hcp:0.002:0.2:0.2:0.2:sc:0.002:0.2:0.2:0.2:'
                         'slab:0.002:0.02:0.02:0.2:cluster:0.002:0.2:0.2:0.2}',
    'energy_parameter_name': 'free_energy',
    'force_parameter_name': 'dummy',
    'force_mask_parameter_name': 'dummy',
    'e0': -0.54289024,
    'sparse_jitter': 1E-8,
    'do_copy_at_file': 'F',
    'sparse_separate_file': 'T',
    'gp_file': 'platinum.xml',
    'gap_cmd': 'gap_fit',
    'mpi_cmd': None,
    'mpi_procs': 1,
    'omp_threads': 128
    }


train_params_nbody = {
    'do_n_body': True,
    'atoms_filename': '/users/kloppej1/scratch/jank/fireworks/complete.xyz',
    'nb_order': 2,
    'nb_compact_clusters': 'T',
    'nb_cutoff': 5.0,
    'nb_n_sparse': 20,
    'nb_covariance_type': 'ard_se',
    'nb_delta': 0.5,
    'nb_theta_uniform': 1.0,
    'nb_sparse_method': 'uniform',
    'soap_l_max': 8,
    'soap_alpha_max': '{{8}}',
    'soap_atom_sigma_r': '{{0.5}}',
    'soap_atom_sigma_t': '{{0.5}}',
    'soap_atom_sigma_r_scaling': '{{0.0}}',
    'soap_atom_sigma_t_scaling': '{{0.0}}',
    'soap_zeta': 6,
    'soap_rcuthard': 5.7,
    'soap_rcutsoft': 4.2,
    'soap_basis': 'poly3gauss',
    'soap_scaling_mode': 'polynomial',
    'soap_amplitude_scaling': '{{1.0}}',
    'soap_n_species': 1,
    'soap_species_Z': '{78}',
    'soap_radial_enhancement': '{{1}}',
    'soap_compress_file': '/users/kloppej1/scratch/jank/fireworks/compress.dat',
    'soap_central_weight': 1.0,
    'soap_config_type_n_sparse': '{bcc:250:fcc:250:hcp:250:sc:250:slab:250:cluster:250}',
    'soap_delta': 0.1,
    'soap_f0': 0.0,
    'soap_covariance_type': 'dot_product',
    'soap_sparse_method': 'cur_points',
    'default_sigma': '{0.002 0.2 0.2 0.2}',
    'config_type_sigma': '{bcc:0.002:0.2:0.2:0.2:fcc:0.002:0.2:0.2:0.2:'
                         'hcp:0.002:0.2:0.2:0.2:sc:0.002:0.2:0.2:0.2:'
                         'slab:0.002:0.02:0.02:0.2:cluster:0.002:0.2:0.2:0.2}',
    'energy_parameter_name': 'free_energy',
    'force_parameter_name': 'dummy',
    'force_mask_parameter_name': 'dummy',
    'e0': -0.54289024,
    'sparse_jitter': 1E-8,
    'do_copy_at_file': 'F',
    'sparse_separate_file': 'T',
    'gp_file': 'platinum.xml',
    'gap_cmd': 'gap_fit',
    'mpi_cmd': None,
    'mpi_procs': 1,
    'omp_threads': 128
}

# print(len(train_params))

# pot_wf = potential_trainer(train_params=train_params)

# print(wf)
# lpad.add_wf(wf)

structures = Xdatcar('/home/jank/work/Aalto/vasp/training_data/liq/5000K_MD/XDATCAR').structures[5:36]

# pmg_struct = AseAtomsAdaptor().get_structure(read('/home/jank/work/Aalto/vasp/training_data/bcc/POSCAR'))
# pmg_struct = Xdatcar('/home/jank/work/Aalto/vasp/training_data/liq/5000K_MD/XDATCAR').structures[-1]

lammps_params = {
    'lammps_settings': [
        'newton on', 'boundary p p p', 'units metal', 'atom_style atomic', 'read_data atom.pos', 'mass * 195.084',
        'pair_style quip', 'pair_coeff * * POT_FW_LABEL "Potential xml_label=POT_FW_NAME" 78',
        'thermo_style custom time pe ke temp', 'thermo 1', 'velocity all zero linear',
        'min_style cg', 'minimize 1e-10 1e-12 10000 100000',
        'write_dump all atom final_positions.atom'],
    'atoms_filename': 'atom.pos',  # filename must match the name in settings above
    # 'structure': '',  # placeholder for pymatgen structure object
    'units': 'metal',  # must match settings
    'lmp_bin': 'lmp',
    'lmp_params': '',  # '-k on t 4 g 1 -sf kk',
    'mpi_cmd': 'srun',
    'mpi_procs': 8,
    'omp_threads': 16,
}

# print(len(train_params))
# print(len(lammps_params))
# md_wf = train_and_run_single_lammps(train_params=train_params, lammps_params=lammps_params)
# print(md_wf)

lpad.reset('2020-09-15')


# for i in np.arange(4.5, 5.6, 0.1):
#     train_params_nbody['nb_cutoff'] = np.round(i, 1)
#     trapot = train_potential(train_params=train_params_nbody, for_validation=True, db_file='db.json')
#     lpad.add_wf(trapot)
# quit()

# lmp_fws = []
# for s in structures:
#     lammps_params['struture'] = s.as_dict()
#     lmp_fws.append(run_lammps(lammps_params=lammps_params, db_file='db.json', al_file=None))
# lpad.add_wf(Workflow(lmp_fws))

md_wf = train_and_run_multiple_lammps(train_params=train_params_nbody, lammps_params=lammps_params,
                                      structures=structures, db_file='db.json', al_file=None)
# print(md_wf)

lpad.add_wf(md_wf)
