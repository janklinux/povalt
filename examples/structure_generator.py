import os
import time
import random
import datetime

import numpy as np

from fireworks import Workflow
from fireworks import LaunchPad

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.transformations.standard_transformations import SupercellTransformation

from atomate.vasp.fireworks.core import StaticFW
from atomate.vasp.powerups import add_modify_incar


def get_strain_and_stress(old, new):
    eeps = np.empty((3, 3))
    for ii in range(3):
        for jj in range(3):
            eeps[ii, jj] = 1/2 * (
                (np.linalg.norm(new[ii]) - np.linalg.norm(old[jj])) / np.linalg.norm(old[jj]) +
                (np.linalg.norm(new[jj]) - np.linalg.norm(old[ii])) / np.linalg.norm(old[ii]))
    return eeps


def get_static_wf(structure, struc_name='', name='Static_run', vasp_input_set=None,
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

    fws = [StaticFW(structure=structure, vasp_input_set=vis_static, vasp_cmd=vasp_cmd,
                    db_file=db_file, name="{} -- static".format(tag))]

    wfname = "{}: {}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


crystal = os.getcwd().split('/')[-1]
if crystal not in ['bcc', 'fcc', 'hcp', 'sc']:
    raise ValueError('This directory is not conform with generator settings, please correct internals...')

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='testdb', username='jank', password='b@sf_mongo',
                 ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)

if crystal == 'bcc':
    scale_mat = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])  # 128 atoms in bcc
elif crystal == 'fcc':
    scale_mat = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])  # 108 atoms in fcc
elif crystal == 'hcp':
    scale_mat = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 4]])  # 108 atoms in hcp
elif crystal == 'sc':
    scale_mat = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])  # 108 atoms in hcp
else:
    raise ValueError('No scaling implemented for the chosen cell...')

prim = Structure.from_file('POSCAR')
cell = SupercellTransformation(scaling_matrix=scale_mat).apply_transformation(prim)

unit_cell_kpts = [8, 8, 8]  # unit cells have ALL been run at 8 8 8 Gamma
cell_kpts = []
for ik, k in enumerate(unit_cell_kpts):
    cell_kpts.append(int(np.ceil(k / np.sum(scale_mat[ik]))))

kpt_set = Kpoints.gamma_automatic(kpts=cell_kpts, shift=(0, 0, 0)).as_dict()
incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 8, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 1,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 200, 'LAECHG': 'False',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.'}
 
if cell.num_sites < 100 or cell.num_sites > 128:
    print('Number of atoms in cell: {}'.format(cell.num_sites))
    raise ValueError('Atoms in supercell not in the range 100 > sites > 128, adjust transformation matrix...')

print('Number of atoms in supercell: {} || k-grid set to: {}'.format(cell.num_sites, cell_kpts))

random.seed(time.time())
total_structures = 10

# quit()

i = 0
while i < total_structures:
    de = np.array([[(random.random() - 0.5) * 2.25, (random.random() - 0.5) * 0.45, (random.random() - 0.5) * 0.45],
                   [(random.random() - 0.5) * 0.45, (random.random() - 0.5) * 2.25, (random.random() - 0.5) * 0.45],
                   [(random.random() - 0.5) * 0.45, (random.random() - 0.5) * 0.45, (random.random() - 0.5) * 2.25]])

    new_lat = np.add(cell.lattice.matrix, de)

    eps = get_strain_and_stress(old=cell.lattice.matrix, new=new_lat)
    if np.any(eps > 0.15):
        print('Component > 0.15 -> too much strain, omitting structure...')
        continue

    new_crds = []
    new_species = []
    for s in cell.sites:
        new_species.append(s.specie)
        new_crds.append(np.array([(random.random() - 0.5) * 0.25 + s.coords[0],
                                  (random.random() - 0.5) * 0.25 + s.coords[1],
                                  (random.random() - 0.5) * 0.25 + s.coords[2]]))

    new_cell = Structure(lattice=new_lat, species=new_species, coords=new_crds,
                         charge=None, validate_proximity=True, to_unit_cell=False,
                         coords_are_cartesian=True, site_properties=None)

    incar_set = MPStaticSet(new_cell)
    structure_name = '{} {} in '.format(len(new_cell.sites), str(new_cell.symbol_set[0])) + crystal
    meta = {'name': structure_name,
            'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

    static_wf = get_static_wf(structure=new_cell, struc_name=structure_name, vasp_input_set=incar_set,
                              vasp_cmd='srun --nodes=1 --ntasks-per-node=128 vasp_std',
                              user_kpoints_settings=kpt_set, metadata=meta)

    run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)

    i += 1
