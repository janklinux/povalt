import os
import sys
import json
import pymongo
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from quippy.convert import ase_to_quip
from quippy.descriptors import Descriptor


mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, enable=True)


read_db = False
if read_db and mpi_size == 1:
    ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
    cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
    run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
    data_db = run_con.pot_train
    data_db.authenticate('jank', 'b@sf_mongo')
    data_coll = data_db['platinum']
    add_coll = data_db['platinum_additions']

    i = 0
    all_structures = []
    for doc in data_coll.find({}, {'data.final_structure': 1}):
        all_structures.append(doc['data']['final_structure'])
        i += 1
        if i > 1432:
            break

    print(len(all_structures))

    with open('subset_structures.json', 'w') as f:
        json.dump(obj=all_structures, fp=f)
    quit()


if mpi_rank == 0:
    if mpi_size != 4 or len(gpus) != 4:
        print('This script needs 4 CPUs and 4 GPUs to work properly, '
              'please run it accordingly.')

with open('subset_structures.json', 'r') as f:
    all_structures = json.load(f)

all_desc = []
for i, s in enumerate(all_structures):
    quip_atoms = ase_to_quip(AseAtomsAdaptor().get_atoms(Structure.from_dict(s)))
    desc = Descriptor('soap_turbo l_max=8 alpha_max={8} atom_sigma_r={0.5} atom_sigma_t={0.5} '
                      'atom_sigma_r_scaling={0.0} atom_sigma_t_scaling={0.0} zeta=6 rcut_hard=4.7 '
                      'rcut_soft=4.2 basis="poly3gauss" scaling_mode="polynomial" amplitude_scaling={1.0} '
                      'n_species=1 species_Z=78 radial_enhancement={1} central_weight={1.0} delta=0.1 '
                      'f0=0.0 covariance_type=dot_product sparse_method=cur_points')
    q = desc.calc_descriptor(quip_atoms)
    all_desc.append(tf.constant(np.array(q), dtype=tf.float16, shape=q.shape))
    if i > 81149:
        break

# print('CPU {} running from {} to {}'.format(mpi_rank,
#                                             mpi_rank * np.floor(len(all_desc)/4),
#                                             (mpi_rank+1) * np.floor(len(all_desc)/4)))

correlation_limit = 0.65
correlated_list = []
high_bound = 0.3
low_bound = 0.85
if mpi_rank == 0:
    print('Number of descriptor matrices: {}, number of matrix.matrix operations: {}'
          .format(len(all_desc), int(np.ceil(len(all_desc)**2/2))))
    print('Starting comparison of all matrices on GPU, working... ', end='')
    sys.stdout.flush()

grid_order = []
mpi_comm.Barrier()
with tf.device('GPU:{}'.format(mpi_rank)):
    for i in range(len(all_desc)):
        if i % int(len(all_desc)/10) == 0:
            print('{:d}% [{}] '.format(int(np.round(i/len(all_desc)*100, 0)), mpi_rank), end='')
            sys.stdout.flush()
        for j in range(int(mpi_rank * np.floor(len(all_desc)/4)),
                       int((mpi_rank+1) * np.floor(len(all_desc)/4) + 1)):
            matrix = tf.matmul(all_desc[i], tf.transpose(all_desc[j]))
            if np.any(matrix < correlation_limit):
                correlated_list.append([i, j])
            if np.any(matrix < low_bound):
                low_bound = tf.reduce_min(matrix)
            if np.any(matrix > high_bound):
                high_bound = tf.reduce_max(matrix)
            local_order = np.array([0, 0, 0, 0])
            for row in matrix:
                for el in row:
                    if 0 <= el < 0.25:
                        local_order[0] += 1
                    elif 0.25 <= el < 0.5:
                        local_order[1] += 1
                    elif 0.5 <= el < 0.75:
                        local_order[2] += 1
                    else:
                        local_order[3] += 1
            grid_order.append(local_order)

mpi_comm.Barrier()
if mpi_rank == 0:
    print('')

print('Overall absolute min | max on CPU {}: {} | {}'.format(mpi_rank, low_bound, high_bound))

local_best_list = []
for p, q in correlated_list:
    if p not in local_best_list:
        local_best_list.append(p)
    if q not in local_best_list:
        local_best_list.append(q)

# print('best list len: {}'.format(len(best_list)))
if mpi_rank == 0:
    global_order = []
    best_list = local_best_list
    for i in [1, 2, 3]:
        tmp = mpi_comm.recv(source=i)
        for el in tmp:
            if el not in best_list:
                best_list.append(el)
        tmp = mpi_comm.recv(source=i)
        global_order.append(tmp)

    print('len: {}'.format(len(best_list)))
    with open('best_list.json', 'w') as f:
        json.dump(obj=sorted(best_list), fp=f)
    with open('global_order.json', 'w') as f:
        json.dump(obj=global_order, fp=f)
else:
    mpi_comm.send(local_best_list, dest=0)
    mpi_comm.send(local_order, dest=0)

MPI.Finalize()
