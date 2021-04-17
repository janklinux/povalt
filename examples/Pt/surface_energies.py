import os
import datetime
import numpy as np
from fireworks import Workflow, LaunchPad
from pymatgen import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.powerups import add_modify_incar


def get_relax_wf(structure, struc_name='', name='', vasp_input_set=None,
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
                      db_file=db_file, name="{} -- OptimizeFW".format(tag))]
    wfname = "{}: {}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


lpad = LaunchPad(host='195.148.22.179', port=27017, name='hkl_fw', username='jank', password='mongo', ssl=False)

kpt_set = Kpoints.gamma_automatic(kpts=[4, 4, 1], shift=(0, 0, 0)).as_dict()
incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 16, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 60, 'LAECHG': '.FALSE.',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.', 'IDIPOL': 3, 'LDIPOL': '.TRUE.', 'DIPOL': '0.5 0.5 0.5'}

hkl_list = []
for ih in range(5):
    for ik in range(5):
        for il in range(5):
            if np.sum([ih, ik, il]) >= 1:
                hkl_list.append([ih, ik, il])

all_slabs = []
systems = ['fcc', 'bcc', 'sc', 'hcp']
for csys in systems:
    s = Structure.from_file(os.path.join(csys, 'POSCAR'))
    for hkl in hkl_list:
        slab = SlabGenerator(s, miller_index=hkl, min_slab_size=2*s.lattice.matrix[2][2],
                             min_vacuum_size=15.0, center_slab=True).get_slab()
        if len(slab.sites) < 100:  # and check_vacuum_direction(slab):
            if slab not in all_slabs:
                all_slabs.append(slab)
                # print('adding face {}'.format(hkl))
                incar_set = MPRelaxSet(slab)
                structure_name = 'Surface energy validation for hkl: {}'.format(hkl)
                meta = {'name': structure_name,
                        'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
                relax_wf = get_relax_wf(structure=slab, struc_name=structure_name, vasp_input_set=incar_set,
                                        vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                                        user_kpoints_settings=kpt_set, metadata=meta)
                run_wf = add_modify_incar(relax_wf, modify_incar_params={'incar_update': incar_mod})
                lpad.add_wf(run_wf)
                # if np.sum(hkl) == 4:
                #     slab.to(fmt='POSCAR', filename='POSCAR')
                #     os.system('VESTA POSCAR')
