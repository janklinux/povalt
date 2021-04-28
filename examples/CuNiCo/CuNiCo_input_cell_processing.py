import re
import os
import time
import datetime
import numpy as np
from pymongo import MongoClient
from pymatgen import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from fireworks import LaunchPad, Workflow
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import StaticFW
from atomate.vasp.powerups import add_modify_incar
from pniggli import niggli_reduce, utils_3d


def get_cij(structure):
    eps = np.array([float(0.05*(x-0.5)) for x in np.random.random(6)])
    cij = np.array([[1+eps[0], eps[5]/2, eps[4]/2],
                    [eps[5]/2, 1+eps[1], eps[3]/2],
                    [eps[4]/2, eps[3]/2, 1+eps[2]]])
    return Lattice(np.transpose(np.dot(cij, np.transpose(structure.lattice.matrix))))


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


fcc_lattice = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
bcc_lattice = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
hcp_lattice = np.array([[1, -np.sqrt(3), 0], [1, np.sqrt(3), 0], [0, 0, 1]])
sc_lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['CuNiCo']

# lpad_cuau = LaunchPad(host='195.148.22.179', port=27017, name='cuau_fw', username='jank', password='mongo', ssl=False)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 8, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 100,
             'LAECHG': 'False', 'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.'}

n_fcc = 0
n_fsym = 0
n_bcc = 0
n_bsym = 0
n_sc = 0
n_ssym = 0
n_hcp = 0
n_hsym = 0
n_diff = 0


all_wfs = []

np.random.seed(int(time.time()))

print(data_coll.estimated_document_count())

for doc in data_coll.find({}):
    if 'CuNiCo CASM generated and relaxed structure for multiplication and rnd distortion' in doc['name']:
        bin_name = doc['name'].split('||')[1].split()[3]
        crystal = bin_name.split('_')[0]
        s = Structure.from_dict(doc['data']['final_structure'])

        if crystal == 'fcc':
            if s.get_space_group_info() == 'Fm-3m':
                n_fsym += 1
            n_fcc += 1
            # a = 1.98
            # fcc_lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
            # niggli_lat = niggli_reduce(fcc_lat)
            # print(utils_3d._get_metric(fcc_lat))
            # quit()
            #
            # print(s)
            # print(SpacegroupAnalyzer(s).find_primitive().lattice)
            # print(SpacegroupAnalyzer(s).get_space_group_operations())
            # print(SpacegroupAnalyzer(s).get_conventional_standard_structure())
            # print(SpacegroupAnalyzer(s).get_symmetry_operations())
            # print(SpacegroupAnalyzer(s).get_symmetry_dataset())
            # quit()
        elif crystal == 'bcc':
            if s.get_space_group_info() == 'Im-3m':
                n_bsym += 1
            n_bcc += 1
        elif crystal == 'sc':
            if s.get_space_group_info() == 'Pm-3m':
                n_ssym += 1
            n_sc += 1
        elif crystal == 'hcp':
            if s.get_space_group_info() == 'P1':
                n_hsym += 1
            n_hcp += 1
        else:
            n_diff += 1

        # print('Number of atoms in supercell: {} || Composition: {}'
        #       .format(s.num_sites, s.composition.element_composition))


print('fcc: {}  bcc: {}  sc: {}  hcp: {}'.format(n_fcc, n_bcc, n_sc, n_hcp))
print('fcc: {}  bcc: {}  sc: {}  hcp: {}'.format(n_fsym, n_bsym, n_ssym, n_hsym))

print('rest: {}'.format(n_diff))
