import os
import numpy as np
from fireworks import LaunchPad
from monty.serialization import loadfn
from povalt.firetasks.wf_generators import aims_relax


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='fw_run', username='jank', password='b@sf_mongo',
                 ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)


all_clusters = loadfn(fn='all_clusters.json')

print('\n')
print('    Structures        ||     Energies ')
print('====================--++--===============')
print('n_atoms   #clusters   ||    DFT     GAP')
print('----------------------++-----------------')
for c in range(15, 250):
    num_clusters = len(all_clusters[str(c)]['structures'])
    edft = 0.0 if not np.all(all_clusters[str(c)]['energies']['dft']) else \
        str(np.round(np.amin(all_clusters[str(c)]['energies']['dft']), 3))
    egap = 0.0 if not np.all(all_clusters[str(c)]['energies']['gap']) else \
        str(np.round(np.amin(all_clusters[str(c)]['energies']['gap']), 3))
    print('  {:3d}       {:3d}       ||    {:4.4f}     {:4.4f}'.format(int(c), int(num_clusters), edft, egap))
print('\n')


quit()


with open('control.in', 'r') as f:
    ctrl = f.readlines()
basis_d = '/users/kloppej1/compile/FHIaims/species_defaults'
# basis_d = '/home/jank/compile/FHIaims/species_defaults'

for c in all_clusters:
    for i, d in enumerate(all_clusters[c]['energies']['dft']):
        if not d:
            wf = aims_relax(aims_cmd='srun aims', control=ctrl, structure=all_clusters[c]['structures'][i],
                            basis_dir=basis_d, metadata={'cluster_atoms': c, 'structure_number': i},
                            name='initial light relax')
            lpad.add_wf(wf)
