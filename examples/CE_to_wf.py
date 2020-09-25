import os
import glob

from fireworks import Workflow, LaunchPad
from pymatgen import Structure
from atomate.fhi_aims.fireworks.core import OptimizeFW

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')


lpad = LaunchPad(host='numphys.org', port=27017, name='basf_fw', username='jank', password='b@sf_mongo',
                 ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)

# lpad.reset('2020-09-24')


def get_optimize_wf(control, structure, struc_name='', name='Relax',
                    aims_cmd=None, tag=None, metadata=None):

    fws = [OptimizeFW(control=control, structure=structure, aims_cmd=aims_cmd, name="{} Relax".format(tag))]

    wfname = "{}:{}".format(struc_name, name)
    return Workflow(fws, name=wfname, metadata=metadata)


basis_dir = '/home/jank/compile/FHIaims/species_defaults'
dir_list = glob.glob('**/geometry.in', recursive=True)

for i, d in enumerate(dir_list):
    base = d[:-12]

    s = Structure.from_file(os.path.join(base, 'geometry.in'))

    with open(os.path.join(base, 'control.in'), 'r') as f:
        my_ctrl = f.readlines()

    test_fw = get_optimize_wf(control=my_ctrl, structure=s, struc_name=base.split('/')[1],
                              tag=base.split('/')[2], name='Relax', aims_cmd='srun aims')

    lpad.add_wf(test_fw)
