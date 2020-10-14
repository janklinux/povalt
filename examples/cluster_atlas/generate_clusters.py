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
        candidate = list(np.random.randint(low=1, high=6, size=3))
        if candidate not in surface_list:
            surface_list.append(candidate)

    return surface_list, list(basic_energies)


np.random.seed(int(time.time()))

new_clusters = 0

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
        # fcc: 4.158 Au | 3.967 Pt
        # bcc: 3.301 Au | 3.165 Pt
        # sc: 2.749 Au | 2.625 Pt
        try:
            cluster = AseAtomsAdaptor().get_molecule(
                wulff_construction(symbol='Pt', energies=energies, surfaces=faces, latticeconstant=2.625,
                                   structure='sc', size=n_atoms, rounding='below', maxiter=100))
        except (RuntimeError, ValueError, TypeError) as e:
            continue

        struct = Structure(lattice=pmg_lat, species=cluster.species, coords=cluster.cart_coords,
                           coords_are_cartesian=True)

        if len(tmp) == 0:
            tmp.append(struct.as_dict())
            edft.append(False)
            egap.append(False)
            print('+', end='')
            sys.stdout.flush()
            new_clusters += 1
        else:
            equal = False
            for s in tmp:
                if len(Structure.from_dict(s).cart_coords) == len(struct.cart_coords):
                    for ca, cb in zip(Structure.from_dict(s).cart_coords, struct.cart_coords):
                        if np.all(ca == cb):
                            equal = True
                            break

            if not equal:
                tmp.append(struct.as_dict())
                edft.append(False)
                egap.append(False)
                print('+', end='')
                sys.stdout.flush()
                new_clusters += 1

    all_clusters[str(n_atoms)] = {'structures': tmp, 'energies': {'dft': edft, 'gap': egap}}
    print('')

# print('\n    Summay')
# print('=================')
# print('n_atoms   #clusters')
# for c in range(15, 251):
#     print('  {:d3.3}          {:d3.3}'.format(c, len(all_clusters[str(c)]['structures'])))

print('Found {} new clusters this run.'.format(new_clusters))

with open('all_clusters.json', 'w') as f:
    json.dump(obj=all_clusters, fp=f)
