import os
import json
import numpy as np
from pymatgen import Structure
from fireworks import LaunchPad
from povalt.firetasks.wf_generators import aims_double_basis


if os.path.isfile('all_clusters.json'):
    with open('all_clusters.json', 'r') as f:
        all_clusters = json.load(f)
else:
    raise ValueError('No cluster file found, this can not be intentional')

lpad = LaunchPad(host='195.148.22.179', port=27017, name='atlas_fw', username='jank', password='mongo', ssl=False)

total_number = 0

print('\n')
print('    Structures        ||     Energies ')
print('====================--++--===============')
print('n_atoms   #clusters   ||    DFT     GAP')
print('----------------------++-----------------')
for c in range(15, 251):
    num_clusters = len(all_clusters[str(c)]['structures'])
    total_number += num_clusters
    edft = 0.0 if not np.all(all_clusters[str(c)]['energies']['dft']) else \
        str(np.round(np.amin(all_clusters[str(c)]['energies']['dft']), 3))
    egap = 0.0 if not np.all(all_clusters[str(c)]['energies']['gap']) else \
        str(np.round(np.amin(all_clusters[str(c)]['energies']['gap']), 3))
    print('  {:3d}       {:3d}       ||    {:4.4f}     {:4.4f}'.format(int(c), int(num_clusters), edft, egap))
print('\n')


print('total number of clusters: {}'. format(total_number))

with open('control.in', 'r') as f:
    ctrl = f.readlines()
basis_d = '/users/kloppej1/compile/FHIaims/species_defaults'
# basis_d = '/home/jank/compile/FHIaims/species_defaults'

for c in all_clusters:
    for i, d in enumerate(all_clusters[c]['energies']['dft']):
        if not d:
            non_spin_structure = Structure.from_dict(all_clusters[c]['structures'][i])
            site_properties = dict({'initial_moment': []})
            for si, s in enumerate(non_spin_structure.sites):
                site_properties['initial_moment'].append((-1.0)**si)
            structure_with_spin = Structure(lattice=non_spin_structure.lattice,
                                            species=non_spin_structure.species,
                                            coords=non_spin_structure.cart_coords,
                                            charge=None, validate_proximity=False, to_unit_cell=False,
                                            coords_are_cartesian=True, site_properties=site_properties)
            wf = aims_double_basis(aims_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 aims',
                                   control=ctrl, structure=structure_with_spin, basis_dir=basis_d,
                                   metadata={'cluster_atoms': c, 'structure_number': i}, name='initial light relax')
            lpad.add_wf(wf)
