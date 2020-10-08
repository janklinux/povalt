import os
import numpy as np
from fireworks import LaunchPad, Workflow
from monty.serialization import loadfn
from povalt.firetasks.FHIaims import OptimizeFW


def get_aims_wf(control, structure, basis_set, basis_dir, aims_cmd, metadata, name='FHIaims run', parents=None):

    fw = [OptimizeFW(control=control, structure=structure, basis_set=basis_set, basis_dir=basis_dir,
                     aims_cmd=aims_cmd, name=name, parents=parents)]
    return Workflow(fw, name=name, metadata=metadata)


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='fw_run', username='jank', password='b@sf_mongo',
                 ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)


lpad.reset('2020-10-08')


all_clusters = loadfn(fn='all_clusters.json')

print('\n')
print('    Structures        ||     Energies ')
print('====================--++--===============')
print('n_atoms   #clusters   ||    DFT     GAP')
print('----------------------++-----------------')
for c in all_clusters:
    num_clusters = len(all_clusters[c]['structures'])
    edft = 'NaN' if all_clusters[c]['energies']['dft'] is None else \
        str(np.round(np.amin(all_clusters[c]['energies']['dft']), 3))
    egap = 'NaN' if all_clusters[c]['energies']['gap'] is None else \
        str(np.round(np.amin(all_clusters[c]['energies']['gap']), 3))
    print('  {}          {}       ||    {}     {}'.format(c, num_clusters, edft, egap))
print('\n')

with open('control.in', 'r') as f:
    ctrl = f.readlines()
basis_s = 'light'
basis_d = '/users/kloppej1/compile/jank/FHIaims/species_defaults'
# basis_dir = '/home/jank/compile/FHIaims/species_defaults'

for c in all_clusters:
    if all_clusters[c]['energies']['dft'] is None:
        for s in all_clusters[c]['structures']:
            wf = get_aims_wf(control=ctrl, structure=s, basis_set='light', basis_dir=basis_d,
                             aims_cmd='mpirun -n 1 aims', metadata={'mymeta': 'thisstuff'})
            lpad.add_wf(wf)
