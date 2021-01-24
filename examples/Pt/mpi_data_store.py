import os
import io
import sys
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

lpad = LaunchPad(host='195.148.22.179', port=27017, name='phonon_fw', username='jank', password='mongo', ssl=False)
all_jobs = lpad.get_wf_ids({'state': 'COMPLETED'})
offset = np.floor(len(all_jobs) / size)

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['platinum']


if rank == 0:
    print('Processing {} jobs on {} processors...'.format(len(all_jobs), size))
    sys.stdout.flush()

cpu = 0
local_list = []
for i, j in enumerate(all_jobs):
    if rank == cpu:
        local_list.append(all_jobs[i])
    if cpu != size - 1:
        if i % offset == 0 and i != 0:
            cpu += 1

all_len = len(local_list)
store = comm.allreduce(all_len, op=MPI.SUM)

if store != len(all_jobs):
    raise ValueError('MPI job distribution not consistent')

for wfid in local_list:
    print('Task {} processing WF {}...'.format(rank, wfid))
    sys.stdout.flush()

    fw = lpad.get_fw_by_id(wfid)

    ldir = '/'.join(lpad.get_launchdir(fw_id=wfid).split('/')[-3:])

    if not os.path.isdir(ldir):
        raise FileNotFoundError('Are you on the right machine? '
                                'Workflow {} directory does not exist here...'.format(wfid))

    run = Vasprun(os.path.join(ldir, 'vasprun.xml.gz'))
    if not run.converged or not run.converged_electronic:
        raise ValueError('Run {} is NOT converged, something is very wrong here...'.format(wfid))

    atoms = aseread(os.path.join(ldir, 'vasprun.xml.gz'), parallel=False)
    file = io.StringIO()
    asewrite(filename=file, images=atoms, format='xyz', parallel=False)
    file.seek(0)
    xyz = file.readlines()
    file.close()

    # for Pt this is done during writing, all following projects NEED THIS HERE
    # stress = atoms.get_stress(voigt=False)
    # vol = atoms.get_volume()
    # virial = -np.dot(vol, stress)
    #
    # xyz[1] = xyz[1].strip() + ' virial="{} {} {} {} {} {} {} {} {}" config_type=bulk\n'.format(
    #     virial[0][0], virial[0][1], virial[0][2],
    #     virial[1][0], virial[1][1], virial[1][2],
    #     virial[2][0], virial[2][1], virial[2][2])

    dft_data = dict()
    dft_data['xyz'] = xyz
    dft_data['PBE_54'] = run.potcar_symbols
    dft_data['parameters'] = run.parameters.as_dict()
    dft_data['free_energy'] = atoms.get_potential_energy(force_consistent=True)
    dft_data['final_structure'] = run.final_structure.as_dict()

    fw_dict = lpad.get_wf_by_fw_id(fw.fw_id).as_dict()
    data_name = 'Pt random structure  ||  ' + fw_dict['metadata']['name'] + \
                '  ||  created ' + fw_dict['metadata']['date'] + '  ||  StaticFW'

    data_coll.insert_one({'name': data_name, 'data': dft_data})
    lpad.delete_wf(wfid, delete_launch_dirs=True)
    sys.stdout.flush()

MPI.Finalize()