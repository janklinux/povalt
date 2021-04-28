import os
import json
import shutil
from pymongo import MongoClient
from pymatgen import Structure


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['CuNiCo']


scels = []
for i in range(1, 6):
    scels.append('SCEL{}_'.format(i))


base_dir = os.getcwd()
import_list = {'bcc': [], 'fcc': [], 'sc': [], 'hcp': []}
count = {'bcc': 0, 'fcc': 0, 'sc': 0, 'hcp': 0}

for crystal in ['fcc', 'bcc', 'hcp', 'sc']:
    if os.path.isdir(crystal):
        shutil.rmtree(crystal)
    os.mkdir(crystal)


for doc in data_coll.find({}):
    for scel in scels:
        if scel in doc['name']:
            # os.mkdir((str(count)))
            crystal = doc['name'].split('||')[1].split()[3].split('_')[0]

            s = Structure.from_dict(doc['data']['final_structure'])
            comp = s.composition.element_composition

            if 'Cu' not in comp:
                xyz = doc['data']['xyz']

                out = dict()
                out['atom_type'] = []
                out['atoms_per_type'] = []
                out['coord_mode'] = 'Direct'
                out['relaxed_basis'] = []
                out['relaxed_energy'] = float(doc['data']['free_energy'])
                out['relaxed_forces'] = []
                out['relaxed_lattice'] = []

                for j in s.symbol_set:
                    out['atom_type'].append(j)
                    out['atoms_per_type'].append(str(s.species).count(j))

                for k in s.frac_coords:
                    out['relaxed_basis'].append(list(k))

                for v in s.lattice.matrix:
                    out['relaxed_lattice'].append(list(v))

                for il, line in enumerate(xyz):
                    if il > 1:
                        out['relaxed_forces'].append([float(x) for x in line.split()[4:7]])

                os.mkdir(os.path.join(crystal, str(count[crystal])))
                with open(os.path.join(crystal, str(count[crystal]), 'properties.calc.json'), 'w') as f:
                    json.dump(obj=out, fp=f, indent=2)

                import_list[crystal].append(os.path.join(base_dir, crystal, str(count[crystal]),
                                                         'properties.calc.json'))

                count[crystal] += 1


for crystal in ['fcc', 'bcc', 'hcp', 'sc']:
    with open('import_{}.list'.format(crystal), 'w') as f:
        for line in import_list[crystal]:
            f.write(line + '\n')
