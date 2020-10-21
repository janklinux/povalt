import json
import numpy as np
from fireworks import LaunchPad
from povalt.training.training import TrainJob
from povalt.firetasks.wf_generators import train_potential


lpad = LaunchPad(host='195.148.22.179', port=27017, name='potential_fw', username='jank', password='mongo', ssl=False)

do_delta = True
exponents = np.arange(start=-3, stop=-0.5, step=0.2)
delta = [np.round(float(10**x), 5) for x in exponents]

do_cutoff = False
cutoffs = np.arange(start=3.0, stop=4.2, step=0.1)

do_sigma = False
sigma = np.arange(start=0.4, stop=1.2, step=0.1)

do_zeta = False
zeta = np.arange(start=3, stop=8, step=1)

with open('db.json', 'r') as f:
    db_info = json.load(f)
TrainJob(train_params={}, for_validation=True, db_info=db_info).clear_validation_database()

if do_delta:
    for d in delta:
        train_params_nbody = {
            'do_n_body': True,
            'atoms_filename': '/users/kloppej1/scratch/jank/pot_fit/Au/train.xyz',
            'nb_order': 2,
            'nb_compact_clusters': 'T',
            'nb_cutoff': 4.7,
            'nb_n_sparse': 25,
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
            'soap_rcuthard': 4.7,
            'soap_rcutsoft': 4.2,
            'soap_basis': 'poly3gauss',
            'soap_scaling_mode': 'polynomial',
            'soap_amplitude_scaling': '{{1.0}}',
            'soap_n_species': 1,
            'soap_species_Z': '{79}',
            'soap_radial_enhancement': '{{1}}',
            'soap_compress_file': '/users/kloppej1/scratch/jank/pot_fit/Au/compress.dat',
            'soap_central_weight': 1.0,
            'soap_config_type_n_sparse': '{bcc:150:fcc:150:hcp:150:sc:150:slab:150:dimer:10}',
            'soap_delta': d,
            'soap_f0': 0.0,
            'soap_covariance_type': 'dot_product',
            'soap_sparse_method': 'cur_points',
            'default_sigma': '{0.002 0.2 0.2 0.2}',
            'config_type_sigma': '{bcc:0.002:0.2:0.2:0.2:fcc:0.002:0.2:0.2:0.2:'
                                 'hcp:0.002:0.2:0.2:0.2:sc:0.002:0.2:0.2:0.2:'
                                 'slab:0.002:0.02:0.02:0.2:dimer:0.002:0.02:0.02:0.2}',
            'energy_parameter_name': 'free_energy',
            'force_parameter_name': 'forces',
            'force_mask_parameter_name': 'force_mask',
            'virial_parameter_name': 'virial',
            'sparse_jitter': 1E-8,
            'do_copy_at_file': 'F',
            'sparse_separate_file': 'T',
            'gp_file': 'aurum.xml',
            'gap_cmd': 'gap_fit',
            'mpi_cmd': None,
            'mpi_procs': 1,
            'omp_threads': 40
        }
        train_wf = train_potential(train_params=train_params_nbody, for_validation=True, db_file='db.json')
        lpad.add_wf(train_wf)


if do_cutoff:
    for c in cutoffs:
        train_params_nbody = {
            'do_n_body': True,
            'atoms_filename': '/users/kloppej1/scratch/jank/pot_fit/Pt/train.xyz',
            'nb_order': 2,
            'nb_compact_clusters': 'T',
            'nb_cutoff': np.round(c + 0.5, 1),
            'nb_n_sparse': 15,
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
            'soap_rcuthard': np.round(c + 0.5, 1),
            'soap_rcutsoft': np.round(c, 1),
            'soap_basis': 'poly3gauss',
            'soap_scaling_mode': 'polynomial',
            'soap_amplitude_scaling': '{{1.0}}',
            'soap_n_species': 1,
            'soap_species_Z': '{78}',
            'soap_radial_enhancement': '{{1}}',
            'soap_compress_file': '/users/kloppej1/scratch/jank/pot_fit/Pt/compress.dat',
            'soap_central_weight': 1.0,
            'soap_config_type_n_sparse': '{bcc:100:fcc:100:hcp:100:sc:100:slab:100:cluster:100:addition:100:dimer:10}',
            'soap_delta': 0.1,
            'soap_f0': 0.0,
            'soap_covariance_type': 'dot_product',
            'soap_sparse_method': 'cur_points',
            'default_sigma': '{0.002 0.2 0.2 0.2}',
            'config_type_sigma': '{bcc:0.002:0.2:0.2:0.2:fcc:0.002:0.2:0.2:0.2:'
                                 'hcp:0.002:0.2:0.2:0.2:sc:0.002:0.2:0.2:0.2:'
                                 'slab:0.002:0.02:0.02:0.2:cluster:0.002:0.2:0.2:0.2:'
                                 'addition:0.002:0.02:0.02:0.2:dimer:0.002:0.02:0.02:0.2}',
            'energy_parameter_name': 'free_energy',
            'force_parameter_name': 'forces',
            'force_mask_parameter_name': 'force_mask',
            'virial_parameter_name': 'virial',
            'sparse_jitter': 1E-8,
            'do_copy_at_file': 'F',
            'sparse_separate_file': 'T',
            'gp_file': 'platinum.xml',
            'gap_cmd': 'gap_fit',
            'mpi_cmd': None,
            'mpi_procs': 1,
            'omp_threads': 40
        }
        train_wf = train_potential(train_params=train_params_nbody, for_validation=True, db_file='db.json')
        lpad.add_wf(train_wf)


if do_sigma:
    for s in sigma:
        train_params_nbody = {
            'do_n_body': True,
            'atoms_filename': '/users/kloppej1/scratch/jank/pot_fit/Pt/train.xyz',
            'nb_order': 2,
            'nb_compact_clusters': 'T',
            'nb_cutoff': 4.7,
            'nb_n_sparse': 15,
            'nb_covariance_type': 'ard_se',
            'nb_delta': 0.5,
            'nb_theta_uniform': 1.0,
            'nb_sparse_method': 'uniform',
            'soap_l_max': 8,
            'soap_alpha_max': '{{8}}',
            'soap_atom_sigma_r': '{{'+str(round(s, 2))+'}}',
            'soap_atom_sigma_t': '{{'+str(round(s, 2))+'}}',
            'soap_atom_sigma_r_scaling': '{{0.0}}',
            'soap_atom_sigma_t_scaling': '{{0.0}}',
            'soap_zeta': 6,
            'soap_rcuthard': 4.7,
            'soap_rcutsoft': 4.2,
            'soap_basis': 'poly3gauss',
            'soap_scaling_mode': 'polynomial',
            'soap_amplitude_scaling': '{{1.0}}',
            'soap_n_species': 1,
            'soap_species_Z': '{78}',
            'soap_radial_enhancement': '{{1}}',
            'soap_compress_file': '/users/kloppej1/scratch/jank/pot_fit/Pt/compress.dat',
            'soap_central_weight': 1.0,
            'soap_config_type_n_sparse': '{bcc:100:fcc:100:hcp:100:sc:100:slab:100:cluster:100:addition:100:dimer:10}',
            'soap_delta': 0.1,
            'soap_f0': 0.0,
            'soap_covariance_type': 'dot_product',
            'soap_sparse_method': 'cur_points',
            'default_sigma': '{0.002 0.2 0.2 0.2}',
            'config_type_sigma': '{bcc:0.002:0.2:0.2:0.2:fcc:0.002:0.2:0.2:0.2:'
                                 'hcp:0.002:0.2:0.2:0.2:sc:0.002:0.2:0.2:0.2:'
                                 'slab:0.002:0.02:0.02:0.2:cluster:0.002:0.2:0.2:0.2:'
                                 'addition:0.002:0.02:0.02:0.2:dimer:0.002:0.02:0.02:0.2}',
            'energy_parameter_name': 'free_energy',
            'force_parameter_name': 'forces',
            'force_mask_parameter_name': 'force_mask',
            'virial_parameter_name': 'virial',
            'sparse_jitter': 1E-8,
            'do_copy_at_file': 'F',
            'sparse_separate_file': 'T',
            'gp_file': 'platinum.xml',
            'gap_cmd': 'gap_fit',
            'mpi_cmd': None,
            'mpi_procs': 1,
            'omp_threads': 40
        }
        train_wf = train_potential(train_params=train_params_nbody, for_validation=True, db_file='db.json')
        lpad.add_wf(train_wf)


if do_zeta:
    for z in zeta:
        train_params_nbody = {
            'do_n_body': True,
            'atoms_filename': '/users/kloppej1/scratch/jank/pot_fit/Pt/train.xyz',
            'nb_order': 2,
            'nb_compact_clusters': 'T',
            'nb_cutoff': 4.7,
            'nb_n_sparse': 15,
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
            'soap_zeta': int(z),
            'soap_rcuthard': 4.7,
            'soap_rcutsoft': 4.2,
            'soap_basis': 'poly3gauss',
            'soap_scaling_mode': 'polynomial',
            'soap_amplitude_scaling': '{{1.0}}',
            'soap_n_species': 1,
            'soap_species_Z': '{78}',
            'soap_radial_enhancement': '{{1}}',
            'soap_compress_file': '/users/kloppej1/scratch/jank/pot_fit/Pt/compress.dat',
            'soap_central_weight': 1.0,
            'soap_config_type_n_sparse': '{bcc:100:fcc:100:hcp:100:sc:100:slab:100:cluster:100:addition:100:dimer:10}',
            'soap_delta': 0.1,
            'soap_f0': 0.0,
            'soap_covariance_type': 'dot_product',
            'soap_sparse_method': 'cur_points',
            'default_sigma': '{0.002 0.2 0.2 0.2}',
            'config_type_sigma': '{bcc:0.002:0.2:0.2:0.2:fcc:0.002:0.2:0.2:0.2:'
                                 'hcp:0.002:0.2:0.2:0.2:sc:0.002:0.2:0.2:0.2:'
                                 'slab:0.002:0.02:0.02:0.2:cluster:0.002:0.2:0.2:0.2:'
                                 'addition:0.002:0.02:0.02:0.2:dimer:0.002:0.02:0.02:0.2}',
            'energy_parameter_name': 'free_energy',
            'force_parameter_name': 'forces',
            'force_mask_parameter_name': 'force_mask',
            'virial_parameter_name': 'virial',
            'sparse_jitter': 1E-8,
            'do_copy_at_file': 'F',
            'sparse_separate_file': 'T',
            'gp_file': 'platinum.xml',
            'gap_cmd': 'gap_fit',
            'mpi_cmd': None,
            'mpi_procs': 1,
            'omp_threads': 40
        }
        train_wf = train_potential(train_params=train_params_nbody, for_validation=True, db_file='db.json')
        lpad.add_wf(train_wf)
