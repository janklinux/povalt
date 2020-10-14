import re
import os
import lzma
import shutil
from pymongo import MongoClient


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
conn = MongoClient(host='numphys.org', port=27017, ssl=True, tlsCAFile=ca_file, ssl_certfile=cl_file)
data_db = conn.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['validate_potentials']

print('DB Content: {} potentials'.format(data_coll.estimated_document_count()))

base_dir = os.getcwd()

xml_name = None
xml_label = None
for pot in data_coll.find():
    os.chdir(base_dir)
    for p in pot:
        if p != '_id':
            bits = p.split(':')
            if len(bits) == 2:
                xml_name = bits[0] + '.' + bits[1]
            if len(bits) == 4:
                xml_label = p.split(':')[3][:-1]

    if os.path.isdir(xml_label):
        shutil.rmtree(xml_label)
    os.mkdir(xml_label)
    os.chdir(xml_label)

    for p in pot:
        if p != '_id':
            with open(re.sub(':', '.', p), 'wb') as f:
                f.write(lzma.decompress(pot[p]))

    os.symlink('../compress.dat', 'compress.dat')
    os.symlink('../test.xyz', 'test.xyz')
    os.system('sed -i s@/users/kloppej1/scratch/jank/pot_fit/Pt/compress.dat@compress.dat@g {}'.format(xml_name))
    print('running: quip atoms_filename=test.xyz param_filename={}'.format(xml_name))
    print('GAP specification: {}'.format(xml_label))
    os.system('nice -n 10 quip atoms_filename=test.xyz param_filename={} e f > quip.result'.format(xml_name))
