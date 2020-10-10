import os
import sys
import time
import json
import numpy as np
from ase.cluster import wulff_construction
from pymatgen import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor


def get_surfaces_and_energies():
    basic_energies = np.random.uniform(0.1, 0.3, size=6)
    np.random.shuffle(basic_energies)

    surface_list = []
    while len(surface_list) < 6:
        candidate = list(np.random.randint(low=1, high=7, size=3))
        if candidate not in surface_list:
            surface_list.append(candidate)

    return surface_list, list(basic_energies)


np.random.seed(int(time.time()))

if os.path.isfile('all_clusters.json'):
    with open('all_clusters.json') as f:
        all_clusters = json.load(f)
else:
    all_clusters = dict()

pmg_lat = Lattice(np.array([[150, 0, 0], [0, 150, 0], [0, 0, 150]]))  # lattice is hacked in pmg to be non-periodic

for n_atoms in range(5, 251):
    if str(n_atoms) in all_clusters:
        tmp = all_clusters[str(n_atoms)]['structures']
        edft = all_clusters[str(n_atoms)]['energies']['dft']
        egap = all_clusters[str(n_atoms)]['energies']['gap']
    else:
        tmp = list()
        edft = list()
        egap = list()
    print('n_atoms: {:d} -- rep:'.format(n_atoms), end='')
    sys.stdout.flush()
    for reps in range(15):
        print(' {}'.format(reps), end='')
        sys.stdout.flush()
        faces, energies = get_surfaces_and_energies()
        try:
            cluster = AseAtomsAdaptor().get_molecule(
                wulff_construction(symbol='Pt', energies=energies, surfaces=faces,
                                   structure='fcc', size=n_atoms, rounding='below', maxiter=100))
        except RuntimeError or ValueError:
            continue

        struct = Structure(lattice=pmg_lat, species=cluster.species, coords=cluster.cart_coords,
                           coords_are_cartesian=True)

        if len(tmp) == 0:
            tmp.append(struct.as_dict())
            edft.append(False)
            egap.append(False)
            print('+', end='')
            sys.stdout.flush()
        else:
            equal = False
            for s in tmp:
                if len(Structure.from_dict(s).cart_coords) == len(struct.cart_coords):
                    for ca, cb in zip(Structure.from_dict(s).cart_coords, struct.cart_coords):
                        if len(ca) != len(cb):
                            print('len misses in ', len(ca), len(cb))
                        if np.all(ca == cb):
                            equal = True

            if not equal:
                tmp.append(struct.as_dict())
                edft.append(False)
                egap.append(False)
                print('+', end='')
                sys.stdout.flush()

    all_clusters[str(n_atoms)] = {'structures': tmp, 'energies': {'dft': edft, 'gap': egap}}
    print('')


print('\n    Summay')
print('=================')
print('n_atoms   #clusters')
for c in all_clusters:
    print('  {}          {}'.format(c, len(all_clusters[c]['structures'])))


with open('all_clusters.json', 'w') as f:
    json.dump(obj=all_clusters, fp=f)
