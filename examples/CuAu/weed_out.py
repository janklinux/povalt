import os
from pymatgen import Structure
from pymongo import MongoClient


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['CuAu']

embedded_date1 = '2021/04/12'
embedded_date2 = '2021/03/24'
embedded_date3 = '2021/03/25'

for doc in data_coll.find({}):
    if embedded_date3 in doc['name']:
        print('pooping: ', doc['_id'], doc['name'])
        # data_coll.delete_one({'_id': doc['_id']})
        # if len(Structure.from_dict(doc['data']['final_structure']).composition.elements) != 2:
