import os
import shutil
from pymatgen import MPRester, Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.core.surface import SlabGenerator

mp = MPRester(api_key='ZB4Do9ybU9DUUkdUclJ')
res = mp.query('Au', ['cif', 'formation_energy', 'material_id'])

base = os.getcwd()

s = Structure.from_str(res[0]['cif'], fmt='cif')

idxs = [(0, 0, 1), (3, 2, 1), (2, 1, 4), (6, 3, 5)]

for idx in idxs:
    print(idx)
    d = str(idx[0])+str(idx[1])+str(idx[2])
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.mkdir(d)

    slab = SlabGenerator(s, miller_index=idx, min_slab_size=7,
                         min_vacuum_size=50.0, center_slab=True).get_slab()

    slab = SupercellTransformation(((3,0,0),(0,3,0),(0,0,1))).apply_transformation(slab)

    os.chdir(d)
    slab.to(filename='geometry.in', fmt='aims')
    os.chdir(base)
