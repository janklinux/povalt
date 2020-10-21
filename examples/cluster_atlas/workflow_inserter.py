import os
import json
import numpy as np
from fireworks import LaunchPad
from povalt.firetasks.wf_generators import AimsRelaxLightTight


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
            wf = AimsRelaxLightTight(aims_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 aims',
                                     control=ctrl, structure=all_clusters[c]['structures'][i],
                                     basis_dir=basis_d, metadata={'cluster_atoms': c, 'structure_number': i},
                                     name='initial light relax')
            lpad.add_wf(wf)
