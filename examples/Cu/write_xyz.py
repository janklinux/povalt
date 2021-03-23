import io
import os
import re
import sys
import json
import pymongo
import numpy as np
from ase.io import read, write
from pymatgen import Structure


def check_vacuum_direction(input_data):
    structure = Structure.from_dict(input_data)
    a = structure.lattice.matrix[0] / 2
    b = structure.lattice.matrix[1] / 2
    for c in structure.cart_coords:
        if np.linalg.norm(c - np.array([a + b])) < 6:
            return False
    return True


np.random.seed(1410)  # fix for reproduction

read_from_db = False
force_fraction = 1  # percentage of forces to INCLUDE for training

systems = ['fcc', 'bcc', 'hcp', 'sc']  # , 'slab', 'cluster', 'addition']

train_split = {'fcc': 1, 'bcc': 1, 'hcp': 1, 'sc': 1, 'slab': 1,
               'cluster': 1, 'addition': 1}

if read_from_db:
    ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
    cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
    run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
    data_db = run_con.pot_train
    data_db.authenticate('jank', 'b@sf_mongo')
    data_coll = data_db['cuprum']
    complete_xyz = []
    crystal_system = []
    print('Starting DB read of {} entries...'.format(data_coll.estimated_document_count()))
    print('busy on: ', end='')
    sys.stdout.flush()
    ik = 0
    for doc in data_coll.find({}):
        if ik % 500 == 0:
            print(' {:d}'.format(ik), end='')
            sys.stdout.flush()
        ik += 1

        valid = True

        if 'slab' in doc['name']:
            valid = check_vacuum_direction(doc['data']['final_structure'])

        if valid:
            complete_xyz.append(doc['data']['xyz'])
            if 'slab' in doc['name']:
                crystal_system.append('slab')
            elif 'cluster' in doc['name']:
                crystal_system.append('cluster')
            else:
                crystal_system.append(doc['name'].split('||')[1].split(' ')[5])

    with open('structures.json', 'w') as f:
        json.dump(complete_xyz, f)
    with open('systems.json', 'w') as f:
        json.dump(crystal_system, f)
    print('')
else:
    with open('structures.json', 'r') as f:
        complete_xyz = json.load(f)
    with open('systems.json', 'r') as f:
        crystal_system = json.load(f)

system_count = dict()
train_selected = dict()
for sys in systems:
    system_count[sys] = crystal_system.count(sys)
    tmp = []
    for i in range(system_count[sys]):
        if i < system_count[sys] * train_split[sys]:
            tmp.append(True)
        else:
            tmp.append(False)
        train_selected[sys] = tmp
        np.random.shuffle(train_selected[sys])

print('There\'s currently {} computed structures in the database'.format(len(complete_xyz)))
# print('Including in training DB: fcc     : {:5d} [{:3.1f}%]\n'
#       '                          bcc     : {:5d} [{:3.1f}%]\n'
#       '                          sc      : {:5d} [{:3.1f}%]\n'
#       '                          hcp     : {:5d} [{:3.1f}%]\n'
#       '                          slab    : {:5d} [{:3.1f}%]\n'
#       '                          cluster : {:5d} [{:3.1f}%]\n'
#       '                          addition: {:5d} [{:3.1f}%]'.
#       format(int(system_count['fcc'] * train_split['fcc']), train_split['fcc']*100,
#              int(system_count['bcc'] * train_split['bcc']), train_split['bcc']*100,
#              int(system_count['sc'] * train_split['sc']), train_split['sc']*100,
#              int(system_count['hcp'] * train_split['hcp']), train_split['hcp']*100,
#              int(system_count['slab'] * train_split['slab']), train_split['slab']*100,
#              int(system_count['cluster'] * train_split['cluster']), train_split['cluster']*100,
#              int(system_count['addition'] * train_split['addition']), train_split['addition']*100))

processed = {'fcc': [], 'bcc': [], 'hcp': [], 'sc': [], 'slab': [], 'cluster': [], 'addition': []}


with open('/tmp/delme.xyz', 'w') as f:
    for xyz in complete_xyz:
        for line in xyz:
            f.write(line)

atoms = read('/tmp/delme.xyz', index=':')
for ia, at in enumerate(atoms):
    at.info['config_type'] = crystal_system[ia]

    mask = np.random.choice([False, True], size=len(at), p=[force_fraction, 1.-force_fraction])
    at.new_array('force_mask', mask)

    f_min = 0.1
    f_scale = 0.1
    f = np.clip(np.abs(at.get_forces()) * f_scale, a_min=f_min, a_max=None)
    at.new_array("force_component_sigma", f)

    file = io.StringIO()
    write(filename=file, images=at, format='extxyz', parallel=False)
    file.seek(0)
    processed[crystal_system[ia]].append(file.readlines())
    file.close()


for pr in processed:
    print('{}: {}'.format(pr, len(processed[pr])))


with open('train.xyz', 'w') as f:
    with open('../atom/parsed.xyz', 'r') as f_in:
        f.write(f_in.read())
    with open('../dimer/CuCu_dimer.xyz', 'r') as f_in:
        f.write(f_in.read())
    for sys in systems:
        for i, xyz in enumerate(processed[sys]):
            if train_selected[sys][i]:
                for line in xyz:
                    f.write(line)

with open('test.xyz', 'w') as f:
    for sys in systems:
        for xyz in processed[sys]:
            for line in xyz:
                f.write(line)
