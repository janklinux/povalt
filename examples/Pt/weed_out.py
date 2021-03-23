import os
import re
from pymongo import MongoClient


# wrong_energies = [-38.34887538, -36.91321888, -36.00214769]
# correct_energies = [-39.18042893, -38.22870714, -37.54161237]

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['platinum']
# add_coll = data_db['platinum_additions']

for doc in data_coll.find({}):
    if 'elastics' in doc['name']:
        print('pooping: ', doc['_id'], doc['name'])
        # data_coll.delete_one({'_id': doc['_id']})
        # doc['name'] = re.sub('elastics', 'random', doc['name'])
        data_coll.update_one({'_id': doc['_id']}, {'$set': {'name': re.sub('elastics', 'random', doc['name'])}})
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
