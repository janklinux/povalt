import io
import os
import re
import numpy as np
from ase.io import read, write
from pymongo import MongoClient
from pymatgen import Structure
from pymatgen.io.vasp import Vasprun


wrong_structures = []
err_files = ['error_21630.xyz', 'error_21704.xyz', 'error_22088.xyz']
for ef in err_files:
    with open(os.path.join('..', ef), 'rt') as f:
        tmp = []
        for line in f:
            tmp.append(line.strip())
        wrong_structures.append(tmp)


quit()

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['platinum']
add_coll = data_db['platinum_additions']


for doc in data_coll.find({}):
    for ws in wrong_structures:
        tmp1 = []
        tmp2 = []
        for line in ws:
            if line.startswith('Pt'):
                tmp1.append(line.split()[:-3])

        file = io.StringIO()
        file.writelines(doc['data']['xyz'])
        file.seek(0)
        for line in file:
            if line.startswith('Pt'):
                tmp2.append(line.strip().split())

        if len(tmp1) == len(tmp2):
            if np.all(tmp1 == tmp2):
                print('pooping: ', doc['_id'], doc['data']['free_energy'])
                data_coll.delete_one({'_id': doc['_id']})


    # if 'phonons' in doc['name']:
    #     print('pooping: ', doc['_id'], doc['name'])
        # data_coll.delete_one({'_id': doc['_id']})
        # doc['name'] = re.sub('elastics', 'random', doc['name'])
        # data_coll.update_one({'_id': doc['_id']}, {'$set': {'name': re.sub('elastics', 'random', doc['name'])}})
        # data_coll.update_one({'_id': doc['_id']}, {'$set': {'data': doc['data']}})

    # if float(doc['data']['free_energy']) in wrong_energies:
    #     print('dingdingding: ', doc['data']['free_energy'], type(doc['data']['free_energy']))
    #     for ie, we in enumerate(wrong_energies):
    #         if float(we) == float(doc['data']['free_energy']):
    #             print('doc energy: ', doc['data']['free_energy'])
    #             print('wrong energy: ', wrong_energies[ie])
    #             print('correct energy: ', correct_energies[ie])
    #             doc['data']['free_energy'] = correct_energies[ie]
    #             print('doc energy: ', doc['data']['free_energy'])
    #             data_coll.update_one({'_id': doc['_id']}, {'$set': {'data': doc['data']}})

# data_coll.insert_one({'name': data_name, 'data': dft_data})
