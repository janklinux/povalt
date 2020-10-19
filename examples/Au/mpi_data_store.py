import os
import io
import shutil
import pymongo
import numpy as np
from mpi4py import MPI
from fireworks import LaunchPad
from ase.io import read as aseread
from ase.io import write as asewrite
from pymatgen.io.vasp import Vasprun


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

lpad = LaunchPad(host='195.148.22.179', port=27017, name='train_fw', username='jank', password='mongo', ssl=False)
ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['aurum']

all_jobs = lpad.get_wf_ids({'state': 'COMPLETED'})
offset = np.floor(len(all_jobs) / size)

if rank == 0:
    print('Processing {} jobs on {} processors...'.format(len(all_jobs), size))

cpu = 0
local_list = []
for i, j in enumerate(all_jobs):
    if rank == cpu:
        local_list.append(all_jobs[i])
    if cpu != size - 1:
        if i % offset == 0 and i != 0:
            cpu += 1

# for cpu in range(size):
#     if rank == cpu:
#         print(len(local_list))
#         fuck = float(len(local_list))

# print('{} total: {}'.format(rank, size))

all_len = len(local_list)
# comm.Reduce([len(local_list), MPI.INT], [result, MPI.INT], op=MPI.SUM, root=0)
# print('rank: {} with len: {}'.format(rank, all_len))
store = comm.allreduce(all_len, op=MPI.SUM)

# print(store)
# print('rank: {} result: {}'.format(rank, all_len))

if store != len(all_jobs):
    raise ValueError('MPI job distribution not consistent')

# print('rank {} has list: {}'.format(rank, local_list))

tmp_name = 'delme_tmpio_' + str(rank)

for wfid in local_list:
    print('Task {} processing WF {}...'.format(rank, wfid))
    fw = lpad.get_fw_by_id(wfid)
    ldir = lpad.get_launchdir(fw_id=wfid)

    if not os.path.isdir(ldir):
        raise FileNotFoundError('Are you on the right machine? '
                                'Workflow {} directory does not exist here...'.format(wfid))

    run = Vasprun(os.path.join(ldir, 'vasprun.xml.gz'))

    if not run.converged or not run.converged_electronic:
        raise ValueError('Run {} is NOT converged, something is very wrong here...'.format(wfid))

    atoms = aseread(os.path.join(ldir, 'vasprun.xml.gz'))

    print('RN: ', rank, tmp_name)
    asewrite(filename=tmp_name, images=atoms, format='xyz')
    with open(tmp_name, 'r') as f:
        for line in f:
            xyz = f.readlines()
    os.unlink(tmp_name)

    stress = atoms.get_stress(voigt=False)
    vol = atoms.get_volume()
    virial = -np.dot(vol, stress)

    print('rv: ', rank, xyz[1])

    quit()

    xyz[1] = xyz[1].strip() + ' virial="{} {} {} {} {} {} {} {} {}" config_type=bulk\n'.format(
        virial[0][0], virial[0][1], virial[0][2],
        virial[1][0], virial[1][1], virial[1][2],
        virial[2][0], virial[2][1], virial[2][2])

    dft_data = dict()
    dft_data['xyz'] = xyz
    dft_data['PBE_54'] = run.potcar_symbols
    dft_data['parameters'] = run.parameters.as_dict()
    dft_data['free_energy'] = atoms.get_potential_energy(force_consistent=True)
    dft_data['final_structure'] = run.final_structure.as_dict()

    fw_dict = lpad.get_wf_by_fw_id(fw.fw_id).as_dict()
    data_name = 'Aurum random structure  ||  ' + fw_dict['metadata']['name'] + \
                '  ||  created ' + fw_dict['metadata']['date'] + '  ||  StaticFW'

    data_coll.insert_one({'name': data_name, 'data': dft_data})
    shutil.rmtree(ldir)
    lpad.delete_wf(wfid)
