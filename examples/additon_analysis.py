import json
from pymongo import MongoClient


with open('localssl.json', 'r') as f:
    db_info = json.load(f)

con = MongoClient(host=db_info['host'], port=db_info['port'],
                  username=db_info['user'], password=db_info['password'],
                  ssl=True, tlsCAFile=db_info['ssl_ca_certs'], ssl_certfile=db_info['ssl_certfile'])
db = con[db_info['database']]
db.authenticate(db_info['user'], db_info['password'])
coll = db[db_info['structure_collection']]

for doc in coll.find({'data.lammps_energy': {'$exists': True }}):
    print()
    print(doc['data']['free_energy'])
    print(doc['data']['lammps_energy'])
    for x in doc['data']:
        print(x)
    quit()
