from pymatgen import Structure                                                                                                                                                   
from pymatgen.transformations.standard_transformations import SupercellTransformation                                                                                            

s = Structure.from_file('POSCAR')
cell = SupercellTransformation(scaling_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 2]]).apply_transformation(s)

atom = []
species = []

prd = 6
for i in range(3):
    for j in range(3):
        atom.append([3.0 + i*prd, 3.0 + j*prd, 8.0])
        species.append('Cu')
        atom.append([0.0 + i*prd, 3.0 + j*prd, 8.0])
        species.append('Ir')
        atom.append([3.0 + i*prd, 0.0 + j*prd, 8.0])
        species.append('Pt')
        atom.append([0.0 + i*prd, 0.0 + j*prd, 8.0])
        species.append('O')

new_lattice = []
new_species = []
new_coords = []

for la in cell.lattice.matrix:
    new_lattice.append(la)
new_lattice[2] = [0, 0, 100]

for s, c in zip(cell.species, cell.cart_coords):
    new_species.append(s)
    new_coords.append(c)

for s, c in zip(species, atom):
    new_species.append(s)
    new_coords.append(c)

new_cell = Structure(lattice=new_lattice, species=new_species, coords=new_coords, charge=None,
                     validate_proximity=False, to_unit_cell=False, coords_are_cartesian=True,
                     site_properties=None)

new_cell.to(fmt='aims', filename='substrate.in')
