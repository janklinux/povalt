import os
import glob
import datetime
# import subprocess
# import numpy as np
from pymatgen import Structure
from fireworks import LaunchPad, Workflow
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.powerups import add_modify_incar


def get_relax_wf(structure, struc_name='', name='Relax_run', vasp_input_set=None,
                 vasp_cmd=None, db_file=None, user_kpoints_settings=None, tag=None, metadata=None):

    if vasp_input_set is None:
        raise ValueError('INPUTSET needs to be defined...')
    if user_kpoints_settings is None:
        raise ValueError('You have to specify the K-grid...')
    if vasp_cmd is None:
        raise ValueError('vasp_cmd needs to be set by user...')
    if tag is None:
        tag = datetime.datetime.now().strftime('%Y/%m/%d-%T')

    vis = vasp_input_set
    v = vis.as_dict()
    v.update({"user_kpoints_settings": user_kpoints_settings})
    vis_static = vis.__class__.from_dict(v)

    fws = [OptimizeFW(structure=structure, vasp_input_set=vis_static, vasp_cmd=vasp_cmd,
                      db_file=db_file, name="{} -- static".format(tag))]
    wfname = "{}: {}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


lpad = LaunchPad(host='195.148.22.179', port=27017, name='cunico_fw', username='jank', password='mongo', ssl=False)

crystal = os.getcwd().split('/')[-1].split('_')[-1]
if crystal not in ['bcc', 'fcc', 'hcp', 'sc']:
    raise ValueError('This directory is not conform with generator settings, please correct internals...')

all_files = []
for i in range(6, 13):
    # hcp:
    # fcc: scel 1-5: complete; scel 6-12: atom_frac(Cu) => 0.8
    # bcc:
    # sc:
    tmp = glob.glob('training_data/SCEL{:d}_*/**/POSCAR'.format(i), recursive=True)
    all_files.extend(tmp)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 4, 'ISMEAR': 0, 'ISPIN': 2, 'NELM': 60, 'LREAL': '.FALSE.',
             'LAECHG': '.FALSE.', 'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.'}

spin_key = {'Cu': 1, 'Ni': 2, 'Co': -2}

for file in all_files:
    s = Structure.from_file(file)
    site_properties = {'initial_moment': []}
    for sn in s.species:
        site_properties['initial_moment'].append(spin_key[str(sn)])
    spin_structure = Structure(lattice=s.lattice, species=s.species, coords=s.frac_coords,
                               coords_are_cartesian=False, site_properties=site_properties)
    input_set = MPRelaxSet(spin_structure)
    kpts = Kpoints.automatic_density(structure=spin_structure, kppa=1000).as_dict()
    bin_name = crystal + '_'
    for ts in s.composition.element_composition:
        bin_name += ts.name
    scel = file.split('/')[1]
    structure_name = '{} {} {} {}'.format(scel, s.composition.reduced_formula, s.num_sites, bin_name)
    meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
    relax_wf = get_relax_wf(structure=spin_structure, struc_name=structure_name, name='CE Relaxation',
                            vasp_input_set=input_set, user_kpoints_settings=kpts, metadata=meta,
                            vasp_cmd='srun --nodes=1 --ntasks=8 --ntasks-per-node=8 '
                                     '--mem-per-cpu=1800 --exclusive vasp_std')
    run_wf = add_modify_incar(relax_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)


# command = 'casm super --get-transf-mat --structure {}'.format(file)
# transformation = []
# parse_matrix = False
# result = subprocess.Popen(command.split(), stdout=subprocess.PIPE).communicate()
# for line in result:
#     if line is not None:
#         line = line.decode('UTF-8')
#         single = line.split('\n')
#         for sing in single:
#             if parse_matrix:
#                 transformation.append([float(x) for x in sing.split()])
#                 if len(transformation) == 3:
#                     parse_matrix = False
#             if 'The transformation' in sing:
#                 parse_matrix = True
#
# a = 1.8
# fcc_lat = np.array([[0, a, a], [a, 0, a], [a, a, 0]])
# print('from file: ', s.lattice.matrix)
# print(transformation)
# print(np.dot(fcc_lat, transformation))
#
# print(np.dot(np.linalg.inv(transformation), s.lattice.matrix))

# print(np.dot(transformation, np.linalg.inv(transformation)))
