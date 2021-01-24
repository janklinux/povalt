import os
import time
import datetime
import numpy as np
from fireworks import Workflow, LaunchPad
from pymatgen.io.vasp import Xdatcar
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import StaticFW
from atomate.vasp.powerups import add_modify_incar


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


def check_vacuum_direction(input_data):
    a = input_data.lattice.matrix[0] / 2
    b = input_data.lattice.matrix[1] / 2
    for c in input_data.cart_coords:
        if np.linalg.norm(c - np.array([a + b])) < 6:
            return False
    return True


lpad = LaunchPad(host='195.148.22.179', port=27017, name='cu_fw', username='jank', password='mongo', ssl=False)

np.random.seed(int(time.time()))

kpt_set = Kpoints.gamma_automatic(kpts=[2, 2, 1], shift=(0, 0, 0)).as_dict()
incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 8, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 300, 'LAECHG': 'False',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.', 'IDIPOL': 3, 'LDIPOL': '.TRUE.', 'DIPOL': '0.5 0.5 0.5'}


crystal = os.getcwd().split('/')[-1].split('_')[0]
if crystal not in ['bcc', 'fcc', 'hcp', 'sc']:
    raise ValueError('This directory is not conform with generator settings, please correct internals...')

for s in Xdatcar('XDATCAR').structures[1000:2500]:
    added = False
    while not added:
        face = list(np.random.randint(low=1, high=5, size=3))
        slab = SlabGenerator(s, miller_index=face, min_slab_size=s.lattice.matrix[2][2],
                             min_vacuum_size=15.0, center_slab=True).get_slab()
        if len(slab.sites) < 200 and check_vacuum_direction(slab):
            print('adding {} atoms in face {}'.format(slab.num_sites, face))
            incar_set = MPStaticSet(slab)
            structure_name = '{} {} {} slab from MD hkl: {}'\
                .format(slab.num_sites, slab.composition.reduced_formula, crystal, face)

            meta = {'name': structure_name,
                    'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

            static_wf = get_static_wf(structure=slab, struc_name=structure_name, vasp_input_set=incar_set,
                                      vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                                      user_kpoints_settings=kpt_set, metadata=meta)

            run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
            lpad.add_wf(run_wf)
            added = True
