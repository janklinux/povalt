import os
import io
import sys
import gzip
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

lpad = LaunchPad(host='195.148.22.179', port=27017, name='agpd_fw', username='jank', password='mongo', ssl=False)
all_jobs = lpad.get_wf_ids({'state': 'COMPLETED'})
offset = np.floor(len(all_jobs) / size)

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['AgPd']

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

    rel_step = 0
    runtime = 0

    fw = lpad.get_fw_by_id(wfid)
    ldir = lpad.get_launchdir(fw_id=wfid)
    if not os.path.isdir(ldir):
        raise FileNotFoundError('Are you on the right machine?\n'
                                'Workflow directory [{}] does not exist here...'.format(ldir))

    runs = [os.path.join(ldir, vrun)
            for vrun in sorted([file for file in os.listdir(ldir) if file.startswith('vasprun')])]
    cars = [os.path.join(ldir, ocar)
            for ocar in sorted([file for file in os.listdir(ldir) if file.startswith('OUTCAR')])]

    if len(runs) == 0:
        raise FileNotFoundError('No vaspruns found, check output in {}'.format(ldir))

    for run, car in zip(runs, cars):
        os.link(os.path.join(ldir, run), os.path.join(ldir, 'vasprun.xml.gz'))  # ase workaround
        run = Vasprun(os.path.join(ldir, 'vasprun.xml.gz'))
        atoms = aseread(os.path.join(ldir, 'vasprun.xml.gz'), parallel=False, index=':')

        step_time = []
        with gzip.open(car, 'r') as f:
            for line in f:
                if b'LOOP+:  cpu time' in line:
                    step_time.append(float(line.split()[6]))

        for ai, at in enumerate(atoms):
            rel_step += 1
            fw_dict = lpad.get_wf_by_fw_id(fw.fw_id).as_dict()
            stress = at.get_stress(voigt=False)
            vol = at.get_volume()
            virial = -np.dot(vol, stress)

            at.info['virial'] = virial
            at.info['relaxation_step'] = rel_step
            at.info['config_type'] = fw_dict['metadata']['name'].split()[3]

            file = io.StringIO()
            asewrite(filename=file, images=at, format='extxyz', parallel=False)
            file.seek(0)
            xyz = file.readlines()
            file.close()

            runtime += step_time[ai]

            dft_data = dict()
            dft_data['xyz'] = xyz
            dft_data['PBE_54'] = run.potcar_symbols
            dft_data['runtime'] = runtime
            dft_data['parameters'] = run.parameters.as_dict()
            dft_data['free_energy'] = at.get_potential_energy(force_consistent=True)
            dft_data['final_structure'] = run.final_structure.as_dict()

            data_name = 'AgPd generated interface structure ||  ' +  fw_dict['metadata']['name'] + \
                        '  ||  created ' + fw_dict['metadata']['date'] + '  ||  OptimizeFW'

            # data_coll.insert_one({'name': data_name, 'data': dft_data})
            # lpad.delete_wf(wfid, delete_launch_dirs=True)
            sys.stdout.flush()

        os.unlink(os.path.join(ldir, 'vasprun.xml.gz'))

MPI.Finalize()