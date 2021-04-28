import os
import re
from pymongo import MongoClient


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['CuNiCo']


for doc in data_coll.find({}):
#    if 'elastic' in doc['name']:
        print('pooping: ', doc['_id'], doc['name'])
        # data_coll.delete_one({'_id': doc['_id']})
