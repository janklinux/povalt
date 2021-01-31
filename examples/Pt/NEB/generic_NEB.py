import numpy as np
from pymatgen import MPRester, Structure, Lattice
from pymatgen.transformations.standard_transformations import SupercellTransformation


prim = Structure.from_str(MPRester().query(criteria='mp-126', properties=['cif'])[0]['cif'], fmt='cif')

# move atom 13 from 0.3 0.3 0.6 -> 0.3 0.3 0.3 in 5 steps, 14 origin, 13 destination
path = np.linspace(start=0.66666667, stop=0.33333333, num=5, endpoint=True)

for i, p in enumerate(path):
    cell = SupercellTransformation(scaling_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]]).apply_transformation(prim)
    cell.remove_sites([13, 14])

    cell.append(species='Pt', coords=[0.33333333, 0.33333333, p],
                coords_are_cartesian=False, validate_proximity=True)

    cell.to(fmt='POSCAR', filename='pt_bulk_{}.vasp'.format(i))


for i, p in enumerate(path):
    cell = SupercellTransformation(scaling_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]]).apply_transformation(prim)

    cell.remove_sites([13, 14, 6, 7, 8, 15, 16, 17, 24, 25, 26])

    cell.append(species='Pt', coords=[0.33333333, 0.33333333, p],
                coords_are_cartesian=False, validate_proximity=True)

    new_lattice = Lattice([cell.lattice.matrix[0], 3*cell.lattice.matrix[1], cell.lattice.matrix[2]])

    new_cell = Structure(lattice=new_lattice, species=cell.species, coords=cell.cart_coords,
                         coords_are_cartesian=True, validate_proximity=True)

    new_cell.to(fmt='POSCAR', filename='pt_slab_{}.vasp'.format(i))
