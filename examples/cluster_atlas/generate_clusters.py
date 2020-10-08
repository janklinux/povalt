import os
import sys
import time
import numpy as np
from ase.cluster import wulff_construction
from pymatgen import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from monty.serialization import dumpfn, loadfn


def get_surfaces_and_energies():
    np.random.seed(int(time.time()))
    basic_energies = np.random.uniform(0.1, 0.3, size=6)
    np.random.shuffle(basic_energies)

    surface_list = []
    while len(surface_list) < 6:
        candidate = list(np.random.randint(low=1, high=4, size=3))
        if candidate not in surface_list:
            surface_list.append(candidate)

    return surface_list, list(basic_energies)


if os.path.isfile('all_clusters.json'):
    all_clusters = loadfn(fn='all_clusters.json')
else:
    all_clusters = dict()

pmg_lat = Lattice(np.array([[150, 0, 0], [0, 150, 0], [0, 0, 150]]))  # this lattice is hacked

for n_atoms in range(15, 250):
    if n_atoms in all_clusters:
        tmp = all_clusters[n_atoms]['structures']
        edft = all_clusters[n_atoms]['energies']['dft']
        egap = all_clusters[n_atoms]['energies']['gap']
    else:
        tmp = list()
        edft = list()
        egap = list()
    print('n_atoms: {:d} -- rep:'.format(n_atoms), end='')
    sys.stdout.flush()
    for reps in range(35):
        print(' {}'.format(reps), end='')
        sys.stdout.flush()
        faces, energies = get_surfaces_and_energies()
        cluster = AseAtomsAdaptor().get_molecule(
            wulff_construction(symbol='Pt', energies=energies, surfaces=faces,
                               structure='fcc', size=n_atoms, rounding='below'))

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
            for c in tmp:
                if len(Structure.from_dict(c).cart_coords) == len(struct.cart_coords):
                    for ca, cb in zip(Structure.from_dict(c).cart_coords, struct.cart_coords):
                        if np.all(ca == cb):
                            equal = True

            if not equal:
                tmp.append(struct.as_dict())
                edft.append(False)
                egap.append(False)
                print('+', end='')
                sys.stdout.flush()

    all_clusters[n_atoms] = {'structures': tmp, 'energies': {'dft': edft, 'gap': egap}}
    print('')


print('\n    Summay')
print('=================')
print('n_atoms   #clusters')
for c in all_clusters:
    print('  {}          {}'.format(c, len(all_clusters[c]['structures'])))


dumpfn(obj=all_clusters, fn='all_clusters.json')
