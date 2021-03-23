import re
import os
import time
import datetime
import numpy as np
from fireworks import LaunchPad, Workflow
from pymatgen import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.powerups import add_modify_incar


def get_static_wf(structure, struc_name='', name='', vasp_input_set=None,
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
    vs = vis.as_dict()
    vs.update({"user_kpoints_settings": user_kpoints_settings})
    vis_relax = vis.__class__.from_dict(vs)

    fws = [OptimizeFW(structure=structure, vasp_input_set=vis_relax, vasp_cmd=vasp_cmd,
                      db_file=db_file, name="{} -- relax".format(tag))]
    wfname = "{}: {}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


# lpad = LaunchPad(host='195.148.22.179', port=27017, name='cuau_fw', username='jank', password='mongo', ssl=False)

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='fw_run', username='jank', password='b@sf_mongo', ssl=True,
                 ssl_ca_certs=ca_file, ssl_certfile=cl_file)


incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 1, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 100, 'LAECHG': '.FALSE.', 'LREAL': 'AUTO',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.', 'IBRION': 1, 'EDIFFG': 0.01}

hkl_list = []
for ih in range(3, 6):
    for ik in range(3, 6):
        for il in range(3, 6):
            hkl = np.array([ih, ik, il])
            if np.sum(hkl) <= 1:
                continue
            else:
                hkl_list.append(hkl)

print('Number of hkl planes: {}'.format(len(hkl_list)))

np.random.seed(int(time.time()))
systems = ['fcc', 'bcc', 'hcp', 'sc']

for csys in systems:
    s = Structure.from_file('POSCAR_' + csys + '_Au')

    tmp_crds = []
    for ii in range(-20, 20):
        for jj in range(-20, 20):
            for kk in range(-20, 20):
                for c in s.cart_coords:
                    tmp_crds.append(c + np.dot(np.array([ii, jj, kk]), s.lattice.matrix))

    for hkl in hkl_list:
        print('csys: {}  -  hkl: {}'.format(csys, hkl))

        plane = np.zeros((3, 3))
        plane[0] = hkl
        plane[1] = np.array([-1, 0, hkl[0]/hkl[2]])
        plane[2] = np.cross(plane[0], plane[1])

        theta = np.zeros(3)

        for ia, (v, w) in enumerate(zip(plane, s.lattice.matrix)):
            theta[ia] = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))

        rotx = [[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])], [0, np.sin(theta[0]), np.cos(theta[0])]]
        roty = [[np.cos(theta[1]), 0, np.sin(theta[1])], [0, 1, 0], [-np.sin(theta[1]), 0, np.cos(theta[1])]]
        rotz = [[np.cos(theta[2]), -np.sin(theta[2]), 0], [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]]

        rotation = np.dot(rotx, np.dot(roty, rotz))

        target = np.dot(rotation, np.dot(s.lattice.matrix, np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])))
        inv_target = np.linalg.inv(target)

        new_crds = []
        new_specs = []
        for c in tmp_crds:
            tmp = np.dot(c, inv_target)
            if 0 <= tmp[0] < 1 and 0 <= tmp[1] < 1 and 0 <= tmp[2] < 1:
                new_crds.append(c)

        news = Structure(lattice=target, species=['Au' for _ in range(len(new_crds))],
                         coords=new_crds, coords_are_cartesian=True)

        remove_list = []
        for ii in range(len(news.cart_coords)):
            for jj in range(ii + 1, len(news.cart_coords)):
                if news.get_distance(ii, jj) < 1:
                    remove_list.append(jj)
        news.remove_sites(remove_list)

        for ic, c in enumerate(news.frac_coords):
            if c[1] < 0.5:
                news.replace(ic, 'Cu')

        news.sort()
        # news.to(filename='rotated.vasp', fmt='POSCAR')

        incar_set = MPRelaxSet(news)
        structure_name = re.sub(' ', '', str(news.composition.element_composition)) + ' '
        structure_name += str(news.num_sites) + ' in {}_hkl:{}'.format(csys, '-'.join(str(s) for s in hkl))

        meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
        kpt_set = Kpoints.automatic_gamma_density(structure=news, kppa=1000).as_dict()

        # vasp_cmd = 'mpirun --bind-to package:report --map-by ppr:1:core:nooversubscribe '
        # '-n 2 vasp_gpu',
        # vasp_cmd = 'srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',

        static_wf = get_static_wf(structure=news, struc_name=structure_name, vasp_input_set=incar_set,
                                  vasp_cmd='mpirun --bind-to package:report --map-by ppr:1:core:nooversubscribe '
                                           '-n 2 vasp_gpu',
                                  user_kpoints_settings=kpt_set, metadata=meta, name='Interface Relaxation')

        run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
        lpad.add_wf(run_wf)
