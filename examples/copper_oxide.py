import os
from monty.serialization import dumpfn, loadfn
from pymatgen import MPRester, Structure


if os.path.isfile('CuO.json'):
    results = loadfn('CuO.json')
else:
    mp = MPRester(api_key='ZB4Do9ybU9DUUkdUclJ')
    strings = []
    for i in range(1, 6):
        for j in range(i+1, 6):
            strings.append('Cu'+str(i)+'O'+str(j))

    results = []
    for string in strings:
        results.append(mp.query(string, ['cif', 'material_id', 'formation_energy_per_atom', 'nsites']))

    dumpfn(results, 'CuO.json')

for res in results:
    for ss in res:
        s = Structure.from_str(ss['cif'], fmt='cif')
        print(s)
