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
from ase.io import read
from fireworks import LaunchPad
from pymatgen.io.vasp import Xdatcar
from povalt.firetasks.wf_generators import potential_trainer, \
    train_and_run_single_lammps, train_and_run_multiple_lammps
from pymatgen.io.ase import AseAtomsAdaptor


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='fw_run', username='jank', password='b@sf_mongo',
                 ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)

train_params = {'atoms_filename': '/home/jank/work/Aalto/vasp/training_data/potential_fit/testing/complete.xyz',
                'order': 2,
                'compact_clusters': 'T',
                'nb_cutoff': 5.0,
                'n_sparse': 20,
                'nb_covariance_type': 'ard_se',
                'nb_delta': 0.5,
                'theta_uniform': 1.0,
                'nb_sparse_method': 'uniform',
                'l_max': 8,
                'alpha_max': '{{8}}',
                'atom_sigma_r': '{{0.5}}',
                'atom_sigma_t': '{{0.5}}',
                'atom_sigma_r_scaling': '{{0.0}}',
                'atom_sigma_t_scaling': '{{0.0}}',
                'zeta': 6,
                'soap_rcuthard': 7.5,
                'soap_rcutsoft': 6.5,
                'soap_basis': 'poly3gauss',
                'soap_scaling_mode': 'polynomial',
                'soap_amplitude_scaling': '{{1.0}}',
                'soap_n_species': 1,
                'soap_species_Z': '{78}',
                'radial_enhancement': '{{1}}',
                'compress_file': '/home/jank/work/test/fireworks/compress.dat',
                'central_weight': 1.0,
                'config_type_n_sparse': '{fcc:500:bcc:500:hcp:500:sc:500:slab:500:cluster:500}',
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
                                     'cluster:0.002:0.2:0.2:0.2}',
                'energy_parameter_name': 'free_energy',
                'force_parameter_name': 'dummy',
                'force_mask_parameter_name': 'dummy',
                'sparse_jitter': 1E-8,
                'do_copy_at_file': 'F',
                'sparse_separate_file': 'T',
                'gp_file': 'Pt_test.xml',
                'gap_cmd': 'gap_fit',
                'mpi_cmd': None,
                'mpi_procs': 1,
                'omp_threads': 6
                }

# pot_wf = potential_trainer(train_params=train_params)

# print(wf)
# lpad.add_wf(wf)

# pmg_struct = AseAtomsAdaptor().get_structure(read('/home/jank/work/Aalto/vasp/training_data/bcc/POSCAR'))
pmg_struct = Xdatcar('/home/jank/work/Aalto/vasp/training_data/liq/5000K_MD/XDATCAR').structures[-1]

lammps_params = {
    'lammps_settings': [
        'newton on', 'boundary p p p', 'units metal', 'atom_style atomic', 'read_data atom.pos', 'mass * 195.084',
        'pair_style quip', 'pair_coeff * * POT_FW_LABEL "Potential xml_label=POT_FW_NAME" 78',
        'thermo_style custom time pe ke temp', 'thermo 1', 'velocity all zero linear',
        'min_style cg', 'minimize 1e-10 1e-12 10000 100000',
        'write_dump all atom final_positions.atom'],
    'atoms_filename': 'atom.pos',  # filename must match the name in settings above
    'structure': pmg_struct.as_dict(),  # pymatgen structure object
    'units': 'metal',  # must match settings
    'lmp_bin': 'lmp',
    'lmp_params': '-k on t 4 g 1 -sf kk',
    'mpi_cmd': '/usr/bin/mpirun',
    'mpi_procs': 2,
    'omp_threads': 4,
}

# md_wf = train_and_run_single_lammps(train_params=train_params, lammps_params=lammps_params)
# print(md_wf)

lpad.reset('2020-09-11')

md_wf = train_and_run_multiple_lammps(train_params=train_params, lammps_params=lammps_params, num_lammps=1,
                                      db_file='db.json', al_file='al.json')
# print(md_wf)
lpad.add_wf(md_wf)
