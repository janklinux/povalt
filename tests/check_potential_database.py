import re
import lzma
import json
from pymongo import MongoClient


with open('db.json', 'r') as f:
    db_info = json.load(f)

con = MongoClient(host=db_info['host'], port=db_info['port'],
                  username=db_info['user'], password=db_info['password'],
                  ssl=True, tlsCAFile=db_info['ssl_ca_certs'], ssl_certfile=db_info['ssl_certfile'])

db = con[db_info['database']]
db.authenticate(db_info['user'], db_info['password'])

pot_coll = db[db_info['potential_collection']]

for pot in pot_coll.find():
    for p in pot:
        with open(re.sub(':', '.', p), 'wb') as f:
            f.write(lzma.decompress(pot[p]))
