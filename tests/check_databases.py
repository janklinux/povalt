import re
import lzma
import json
from pymongo import MongoClient


with open('localssl.json', 'r') as f:
    db_info = json.load(f)

con = MongoClient(host=db_info['host'], port=db_info['port'],
                  username=db_info['user'], password=db_info['password'],
                  ssl=True, tlsCAFile=db_info['ssl_ca_certs'], ssl_certfile=db_info['ssl_certfile'])

db = con[db_info['database']]
db.authenticate(db_info['user'], db_info['password'])

add_coll = db[db_info['structure_collection']]
pot_coll = db[db_info['potential_collection']]
val_coll = db[db_info['validation_collection']]


print('\n   --- AdditionalStructureDB contents #: {}\n'.format(add_coll.estimated_document_count()))
# for add in add_coll.find({}, {'_id': 0, 'data.xyz': 1, 'name': 1}):
#     print(add['data']['xyz'])
#     quit()


print('   --- ValidationDB contents #: {}\n'.format(val_coll.estimated_document_count()))
# for val in val_coll.find():
#     for v in val:
#         print(v)
# val_coll.remove({})


print('\n   --- PotentialDB contents')
for pot in pot_coll.find():
    for p in pot:
        print(p)
        # if p != '_id':
        #     with open(re.sub(':', '.', p), 'wb') as f:
        #         f.write(lzma.decompress(pot[p]))
