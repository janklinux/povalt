import re
import os
from bson import ObjectId
from pymongo import MongoClient


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['cuprum']

for doc in data_coll.find({}):
    print(doc['data']['xyz'][1])
    quit()
    # if doc['name'].split()[7] not in ['fcc', 'bcc', 'hcp', 'sc']:
    #     print('What type is this? : {}'.format(doc['name'].split()[7]))
    #     continue
    # else:
    #     if doc['data']['xyz'][1][-5:-1] != 'bulk':
    #         print('Already updated or wrong type...: {}'.format(doc['data']['xyz'][1][-5:-1]))
    #         continue
    #     else:
    #         doc['data']['xyz'][1] = re.sub('bulk', doc['name'].split()[7], doc['data']['xyz'][1])
    #         data_coll.update_one({'_id': ObjectId(doc['_id'])},
    #                              {'$set': {'data.xyz': doc['data']['xyz']}})
