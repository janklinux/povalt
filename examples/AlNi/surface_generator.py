import re
import time
import glob
import datetime
import numpy as np
from fireworks import LaunchPad
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
from povalt.firetasks.wf_generators import aims_single_basis


def check_vacuum_direction(input_structure):
    a = input_structure.lattice.matrix[0] / 2
    b = input_structure.lattice.matrix[1] / 2
    for c in input_structure.cart_coords:
        if np.linalg.norm(c - np.array([a + b])) < 6:
            return False
    return True


lpad = LaunchPad(host='195.148.22.179', port=27017, name='test_fw', username='jank', password='mongo', ssl=False)
# lpad.reset('2020-12-03')

with open('control.in', 'r') as f:
    control = f.readlines()

np.random.seed(int(time.time()))

inputs = glob.glob('training_data/SCEL9_*/**/geometry.in', recursive=True)
np.random.shuffle(inputs)

for inp in inputs:
    cell = Structure.from_file(inp)
    possible = False
    new_lat = []
    vac_dir = []
    vac_idx = 0
    for iv, vec in enumerate(cell.lattice.matrix):
        if len(np.nonzero(vec)[0]) == 1:
            vac_idx = iv
            vac_dir.append(1)
            new_lat.append(vec * (1 + np.random.random()))
            possible = True
        else:
            new_lat.append(vec)
            vac_dir.append(0)

    if possible:
        # grid = [6, 6, 6]
        # grid[vac_idx] = 1
        # trimmed_control = []
        # for line in control:
        #     trimmed_control.append(re.sub('k_grid', 'k_grid {} {} {}'.format(grid[0], grid[1], grid[2]), line))
        trimmed_control = control

        # trafo = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # trafo[vac_idx] = np.dot(vac_dir, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
        super_cell = cell  # SupercellTransformation(scaling_matrix=trafo).apply_transformation(structure=cell)
        site_properties = dict({'initial_moment': []})
        for s in super_cell.sites:
            if s.specie.name == 'Al':
                site_properties['initial_moment'].append(1.0)
            else:
                site_properties['initial_moment'].append(-1.0)

        slab = Structure(lattice=new_lat, species=super_cell.species, coords=super_cell.cart_coords,
                         coords_are_cartesian=True, site_properties=site_properties)

        if len(slab.sites) < 50:
            print('adding slab {}'.format(slab.composition.reduced_formula))
            structure_name = '{} {} Slab from CE'.format(len(slab.sites), str(slab.symbol_set))
            meta = {'name': structure_name,
                    'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

            static_wf = aims_single_basis(aims_cmd='mpirun -n 4 aims', control=trimmed_control, structure=slab,
                                          basis_set='light', basis_dir='/home/jank/compile/FHIaims/species_defaults',
                                          metadata=meta, name='single point light')
            lpad.add_wf(static_wf)

    else:
        continue
