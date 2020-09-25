import numpy as np
from pymatgen import Structure

s = Structure.from_file('geometry.in')

center = np.empty(3)
for c in s.cart_coords:
    for i in range(3):
        center[i] += c[i]

center /= len(s.cart_coords)

mins = np.empty(3)
maxs = np.empty(3)

for i in range(3):
    tmp = []
    for j in range(len(s.cart_coords)):
        tmp.append(s.cart_coords[j][i])
    mins[i] = np.amin(tmp)
    maxs[i] = np.amax(tmp)


print(mins)
print(maxs)

print(center)

new = Structure(lattice=[[28, 0, 0], [0, 28, 0], [0, 0, 28]], species=s.species, coords=s.cart_coords,
                coords_are_cartesian=True)

new.to(filename='POSCAR', fmt='POSCAR')
