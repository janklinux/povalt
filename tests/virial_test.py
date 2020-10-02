import os
import io
from ase.io import read, write
from pymatgen import Structure
from pymongo import MongoClient


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')

run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['platinum']
add_coll = data_db['platinum_additions']

for doc in data_coll.find({}):
    for d in doc['data']:
        print(d)

    s = Structure.from_dict(doc['data']['final_structure'])

    print(s)

    s.to(fmt='POSCAR', filename='/tmp/POSCAR')
    atoms = read('/tmp/POSCAR')

    print(atoms)

    write(filename='text.xyz', images=atoms, format='extxyz')

    quit()
