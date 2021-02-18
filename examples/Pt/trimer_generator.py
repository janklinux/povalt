import re
import time
import datetime
import numpy as np
from fireworks import LaunchPad, Workflow
from pymatgen.core.structure import Structure, Lattice
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


lpad = LaunchPad(host='195.148.22.179', port=27017, name='phonon_fw', username='jank', password='mongo', ssl=False)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 4, 'ISMEAR': 0, 'ISYM': 2, 'ISPIN': 2,
             'ALGO': 'All', 'AMIN': 0.01, 'NELM': 200, 'LAECHG': '.FALSE.',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.', 'LREAL': '.FALSE.'}

np.random.seed(int(time.time()))

for i in range(500):
    valid = False
    mid_dist = (1 + (np.random.random() - 0.5) * 0.5) * 3.7
    while not valid:
        x = np.array([0, 0, 0])
        y = np.array([mid_dist, np.sin((np.random.random()-0.5)*180/np.pi), 0])
        z = np.array([0, mid_dist, 0])
        if np.linalg.norm(y - z) > 1.0:
            s = Structure(lattice=Lattice([[10, 0, 0], [0, 10 ,0], [0, 0, 10]]), species=['Pt', 'Pt', 'Pt'],
                          coords=[x, y, z], coords_are_cartesian=True, validate_proximity=True)
            valid = True

            incar_set = MPStaticSet(s)
            structure_name = re.sub(' ', '', str(s.composition.element_composition)) + \
                             ' ' + str(s.num_sites) + ' in trimer'

            meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
            kpt_set = Kpoints.gamma_automatic([1, 1, 1], shift=[0, 0, 0])

            static_wf = get_static_wf(structure=s, struc_name=structure_name, vasp_input_set=incar_set,
                                      vasp_cmd='srun --nodes=1 --ntasks=16 --ntasks-per-node=16 '
                                               '--mem-per-cpu=1800 --exclusive vasp_std',
                                      user_kpoints_settings=kpt_set, metadata=meta)

            run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
            lpad.add_wf(run_wf)
