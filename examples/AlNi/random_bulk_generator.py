import os
import time
import random
import datetime
import numpy as np
from fireworks import LaunchPad
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
from povalt.firetasks.wf_generators import aims_single_basis


crystal = os.getcwd().split('/')[-1]
if crystal not in ['bcc', 'fcc', 'hcp', 'sc']:
    raise ValueError('This directory is not conform with generator settings, please correct internals...')

lpad = LaunchPad(host='195.148.22.179', port=27017, name='test_fw', username='jank', password='mongo', ssl=False)

if crystal == 'bcc':
    scale_mat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
elif crystal == 'fcc':
    scale_mat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
elif crystal == 'hcp':
    scale_mat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
elif crystal == 'sc':
    scale_mat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
else:
    raise ValueError('No scaling implemented for the chosen cell...')

prim = Structure.from_file('POSCAR')
cell = SupercellTransformation(scaling_matrix=scale_mat).apply_transformation(prim)

with open('../control.in', 'r') as f:
    control = f.readlines()

if cell.num_sites > 40:
    print('Number of atoms in cell: {}'.format(cell.num_sites))
    raise ValueError('Atoms in supercell not in the range 100 > sites > 128, adjust transformation matrix...')

random.seed(time.time())
total_structures = 125

si = 0
while si < total_structures:
    new_lat = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            new_lat[i, j] = ((random.random()-0.5) * 0.3 + 1) * cell.lattice.matrix[i, j]

    new_crds = []
    new_species = []
    site_properties = dict({'initial_moment': []})
    for ic, c, s in enumerate(zip(cell.frac_coords, cell.species)):
        new_species.append(s)
        new_crds.append(np.array([(random.random()-0.5) * 0.1 + c[0],
                                  (random.random()-0.5) * 0.1 + c[1],
                                  (random.random()-0.5) * 0.1 + c[2]]))
        if s.name == 'Al':
            site_properties['initial_moment'].append(-1.0*(-1.0)**(ic+1))
        else:
            site_properties['initial_moment'].append(-1.0**(ic+1))

    new_cell = Structure(lattice=new_lat, species=new_species, coords=new_crds,
                         charge=None, validate_proximity=True, to_unit_cell=False,
                         coords_are_cartesian=False, site_properties=site_properties)

    structure_name = '{} {}'.format(len(new_cell.sites), str(new_cell.symbol_set))
    meta = {'name': structure_name,
            'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

    static_wf = aims_single_basis(aims_cmd='mpirun -n 2 aims', control=control, structure=new_cell,
                                  basis_set='light', basis_dir='/home/jank/compile/FHIaims/species_defaults',
                                  metadata=meta, name='single point light')
    lpad.add_wf(static_wf)
    si += 1
