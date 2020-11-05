import time
import glob
import random
import datetime
import numpy as np
from fireworks import LaunchPad
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
from povalt.firetasks.wf_generators import aims_single_basis


lpad = LaunchPad(host='195.148.22.179', port=27017, name='tddft_fw', username='jank', password='mongo', ssl=False)

with open('control.in', 'r') as f:
    control = f.readlines()

random.seed(time.time())

inputs = glob.glob('training_data/SCEL10_*/**/geometry.in', recursive=True)
np.random.shuffle(inputs)

for inp in inputs:
    cell = Structure.from_file(inp)
    # cell = SupercellTransformation(scaling_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))\
    #     .apply_transformation(base)

    if cell.num_sites > 128:
        print('Number of atoms in cell: {}'.format(cell.num_sites))
        raise ValueError('Atoms in supercell > 128, adjust transformation matrix...')

    new_lat = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            new_lat[i, j] = ((random.random()-0.5) * 0.2 + 1) * cell.lattice.matrix[i, j]

    new_crds = []
    new_species = []
    site_properties = dict({'initial_moment': []})
    for s in cell.sites:
        new_species.append(s.specie)
        new_crds.append(np.array([(random.random()-0.5) * 0.35 + s.coords[0],
                                  (random.random()-0.5) * 0.35 + s.coords[1],
                                  (random.random()-0.5) * 0.35 + s.coords[2]]))
        if s.specie.name == 'Al':
            site_properties['initial_moment'].append(1.0)
        else:
            site_properties['initial_moment'].append(-1.0)

    new_cell = Structure(lattice=new_lat, species=new_species, coords=new_crds,
                         charge=None, validate_proximity=True, to_unit_cell=False,
                         coords_are_cartesian=True, site_properties=site_properties)

    structure_name = '{} {}'.format(len(new_cell.sites), str(new_cell.symbol_set))
    meta = {'name': structure_name,
            'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

    static_wf = aims_single_basis(aims_cmd='mpirun -n 4 aims', control=control, structure=new_cell,
                                  basis_set='light', basis_dir='/home/jank/compile/FHIaims/species_defaults',
                                  metadata=meta, name='single point light')
    lpad.add_wf(static_wf)
