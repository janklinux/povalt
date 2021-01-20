import os
import numpy as np
from ase.io.lammpsdata import read_lammps_data
from sknano.generators import FullereneGenerator
from pymatgen.io.ase import AseAtomsAdaptor


def write_geo(geo, names):
    with open('geometry.in', 'w') as f:
        for c, n in zip(geo, names):
            f.write('atom {:5.5f} {:5.5f} {:5.5f} {}\n'.format(c[0], c[1], c[2], n))


def debug_geo(geo, names):
    with open('debug.in', 'w') as f:
        for c, n in zip(geo, names):
            f.write('atom {:5.5f} {:5.5f} {:5.5f} {}\n'.format(c[0], c[1], c[2], n))


def int_sum(n):
    if not isinstance(n, int):
        raise ValueError('int_sum only for integers...')
    return int(n*(n+1)/2)

name = []
atom = []

tmp = []
for i in range(-10, 10):
    for j in range(-10, 10):
        for k in range(-10, 10):
            tmp.append(np.array([i * 2.5, j * 2.5, k * 2.5]))

for at in tmp:
    if 4 < np.linalg.norm(at) < 7:
        atom.append(at)
        name.append('Au')

# for i in range(10, 39):
#     print(i, int_sum(i))

quit()

fulls = [36, 78, 90, 100, 180]
dirs = [18, 19, 20, 21, 22]
for num, dir in zip(fulls, dirs):
    fg = FullereneGenerator(N=num)
    fg.save(fname='data')
    atoms = read_lammps_data('data')

    for at in atoms:
        at.symbol = 'Au'

    s = AseAtomsAdaptor().get_structure(atoms)

    os.mkdir(str(dir))
    with open(str(dir) + '/geometry.in', 'w') as f:
        for c, n in zip(s.cart_coords, s.species):
            f.write('atom {} {} {} {}\n'.format(2*c[0], 2*c[1], 2*c[2], n))

# s.to(fmt='aims', filename='geometry.in')

# write_geo(atom, name)

# box_dim = [6, 4, 2, 1]
# heights = [0, 2.5, 5, 7.5]
# for b, h in zip(box_dim, heights):
#     tmp = []
#     for i in range(b):
#         for j in range(b):
#             tmp.append(np.array([i * 2.5, j * 2.5, h]))
#
#     theta = np.pi / 4
#     rot = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
#
#     tmprot = []
#     for t in tmp:
#         tmprot.append(np.dot(rot, t))
#
#     for at in tmprot:
#         if at[1] < 9:
#             atom.append(at)
#             name.append('Au')
#
#     break
#
# print(len(atom))
# write_geo(atom, name)
