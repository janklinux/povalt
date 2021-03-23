import io
import os
import sys
import json
import pymongo
import numpy as np
import tensorflow as tf
# from mpi4py import MPI
from ase.io import read, write
from pymatgen import Structure
from quippy.convert import ase_to_quip
from quippy.descriptors import Descriptor


def check_vacuum_direction(input_data):
    structure = Structure.from_dict(input_data)
    a = structure.lattice.matrix[0] / 2
    b = structure.lattice.matrix[1] / 2
    for c in structure.cart_coords:
        if np.linalg.norm(c - np.array([a + b])) < 6:
            return False
    return True


# mpi_comm = MPI.COMM_WORLD
# mpi_size = mpi_comm.Get_size()
# mpi_rank = mpi_comm.Get_rank()


tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, enable=True)


np.random.seed(1410)  # fix for reproduction

do_soap = False
read_from_db = False
force_fraction = 1  # percentage of forces to INCLUDE from training


systems = ['fcc_Cu', 'bcc_Cu', 'hcp_Cu', 'sc_Cu', 'slab_Cu',
           'fcc_Au', 'bcc_Au', 'hcp_Au', 'sc_Au', 'slab_Au',
           'fcc_AuCu', 'bcc_AuCu', 'hcp_AuCu', 'sc_AuCu']

train_split = {'fcc_Cu': 0.2, 'bcc_Cu': 0.2, 'hcp_Cu': 0.2, 'sc_Cu': 0.2, 'slab_Cu': 0.75,
               'fcc_Au': 0.2, 'bcc_Au': 0.2, 'hcp_Au': 0.2, 'sc_Au': 0.2, 'slab_Au': 0.75,
               'fcc_AuCu': 1.0, 'bcc_AuCu': 1.0, 'hcp_AuCu': 1.0, 'sc_AuCu': 1.0}

if read_from_db:
    ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
    cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
    run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
    data_db = run_con.pot_train
    data_db.authenticate('jank', 'b@sf_mongo')
    data_coll = data_db['CuAu']
    complete_xyz = []
    crystal_system = []
    print('Starting DB read for CuAu of {} entries...'.format(data_coll.estimated_document_count()))
    print('busy on: ', end='')
    sys.stdout.flush()
    ik = 0
    for doc in data_coll.find({}):
        if ik % 1000 == 0:
            print(' {:d}'.format(ik), end='')
            sys.stdout.flush()
        ik += 1

        valid = True

        if 'slab' in doc['name']:
            valid = check_vacuum_direction(doc['data']['final_structure'])

        if valid:
            complete_xyz.append(doc['data']['xyz'])
            if 'slab' in doc['name']:
                crystal_system.append('slab_CuAu')
            elif 'cluster' in doc['name']:
                crystal_system.append('cluster_CuAu')
            else:
                crystal_system.append(doc['name'].split('||')[1].split(' ')[5])
    print('')

    data_coll = data_db['cuprum']
    print('Starting DB read for Cu of {} entries...'.format(data_coll.estimated_document_count()))
    print('busy on: ', end='')
    sys.stdout.flush()
    ik = 0
    for doc in data_coll.find({}):
        if ik % 1000 == 0:
            print(' {:d}'.format(ik), end='')
            sys.stdout.flush()
        ik += 1

        valid = True

        if 'slab' in doc['name']:
            valid = check_vacuum_direction(doc['data']['final_structure'])

        if valid:
            complete_xyz.append(doc['data']['xyz'])
            if 'Slab' in doc['name']:
                crystal_system.append('slab_Cu')
            elif 'Cluster' in doc['name']:
                crystal_system.append('cluster_Cu')
            else:
                crystal_system.append(doc['name'].split('||')[1].split(' ')[5]+'_Cu')
    print('')

    data_coll = data_db['aurum']
    print('Starting DB read for Au of {} entries...'.format(data_coll.estimated_document_count()))
    print('busy on: ', end='')
    sys.stdout.flush()
    ik = 0
    for doc in data_coll.find({}):
        if ik % 1000 == 0:
            print(' {:d}'.format(ik), end='')
            sys.stdout.flush()
        ik += 1

        valid = True

        if 'slab' in doc['name']:
            valid = check_vacuum_direction(doc['data']['final_structure'])

        if valid:
            complete_xyz.append(doc['data']['xyz'])
            if 'Slab' in doc['name']:
                crystal_system.append('slab_Au')
            elif 'Cluster' in doc['name']:
                crystal_system.append('cluster_Au')
            else:
                crystal_system.append(doc['name'].split('||')[1].split(' ')[5]+'_Au')
    print('')

    with open('structures.json', 'w') as f:
        json.dump(complete_xyz, f)
    with open('systems.json', 'w') as f:
        json.dump(crystal_system, f)
else:
    with open('structures.json', 'r') as f:
        complete_xyz = json.load(f)
    with open('systems.json', 'r') as f:
        crystal_system = json.load(f)


system_count = dict()
train_selected = dict()
for csys in systems:
    system_count[csys] = crystal_system.count(csys)
    tmp = []
    for i in range(system_count[csys]):
        if i < system_count[csys] * train_split[csys]:
            tmp.append(True)
        else:
            tmp.append(False)
        train_selected[csys] = tmp
        np.random.shuffle(train_selected[csys])

print('There\'s currently {} computed structures in the database'.format(len(complete_xyz)))


for sc in system_count:
    print('{}: {}'.format(sc, system_count[sc]))

processed = {'fcc_Cu': [], 'bcc_Cu': [], 'hcp_Cu': [], 'sc_Cu': [], 'slab_Cu': [],
             'fcc_Au': [], 'bcc_Au': [], 'hcp_Au': [], 'sc_Au': [], 'slab_Au': [],
             'fcc_AuCu': [], 'bcc_AuCu': [], 'hcp_AuCu': [], 'sc_AuCu': []}


suggestions = dict()
if do_soap:
    for csys in systems:
        selected_idx = []
        print('SOAP processing {}: '.format(csys))
        all_kernels = dict()
        for ci, species in enumerate(['Cu', 'Au']):
            if species not in csys:
                continue

            print('Species {}...'.format(species), end='')
            sys.stdout.flush()

            tmp = []
            for idx, st in enumerate(complete_xyz):
                if crystal_system[idx] != csys:
                    continue
                tmp.append(st)

            with open('/tmp/all_{}.xyz'.format(csys), 'w') as f:
                for st in tmp:
                    for line in st:
                        f.write(line)
            atoms = read('/tmp/all_{}.xyz'.format(csys), index=':')

            q = dict()

            d = Descriptor('soap_turbo l_max=8 alpha_max={8 8} atom_sigma_r={0.5 0.5} atom_sigma_t={0.5 0.5} '
                           'atom_sigma_r_scaling={0. 0.} atom_sigma_t_scaling={0. 0.} rcut_hard=5.9 '
                           'rcut_soft=5.4 basis=poly3gauss scaling_mode=polynomial amplitude_scaling={1.0 1.0} '
                           'n_species=2 species_Z={29 79} radial_enhancement=1 compress_file=compress.dat '
                           'central_index='+str(ci+1)+' central_weight={1.0 1.0} add_species=F')

            num_desc = 0
            qtmp = []
            for at in atoms:
                qats = ase_to_quip(at)
                qats.set_cutoff(cutoff=5.9)
                qats.calc_connect()
                desc = d.calc_descriptor(qats)
                qtmp.append(tf.constant(np.array(desc), dtype=tf.float16, shape=desc.shape))
                num_desc += desc.shape[0]
            q[species] = qtmp

            print('{} descriptors computed...'.format(num_desc), end='')
            sys.stdout.flush()

            kernel = []
            indexes = []
            with tf.device('/device:GPU:0'):  # .format(0)):
                for ik in range(len(q[species])):
                    for il in range(ik + 1, len(q[species])):
                        # print(q[species][0][ik], q[species][0][il])
                        # print(np.dot(q[species][0][ik], np.transpose(q[species][0][il])), len(q[species][0][ik]))
                        # print(q[species][0][ik].shape[0], q[species][0][il].shape[0])
                        # print(np.sum(np.dot(q[species][0][ik], np.transpose(q[species][0][il]))) /
                        #       (q[species][0][ik].shape[0] * q[species][0][il].shape[0]))
                        tf_result = tf.reduce_sum(tf.matmul(q[species][ik], tf.transpose(q[species][il])))
                        # print(tf_result)
                        # print(tf_result.numpy())
                        kernel.append(tf_result.numpy() / (q[species][ik].shape[0] * q[species][il].shape[0]))
                        # kernel.append(np.sum(np.dot(q[species][0][ik], np.transpose(q[species][0][il]))) /
                        #               (q[species][0][ik].shape[0] * q[species][0][il].shape[0]))
                        indexes.append([ik, il])
                all_kernels[species] = {'kernel': kernel, 'index': indexes}

            with tf.device('/device:CPU:0'):  # .format(0)):
                added = []
                print('{} kernels...'.format(len(all_kernels[species]['kernel'])), end='')
                sys.stdout.flush()
                for ik, (kv, idx) in enumerate(sorted(zip(all_kernels[species]['kernel'],
                                                          all_kernels[species]['index']))):
                    if kv < 0.5:
                        # if csys == 'fcc_AuCu':
                        #     print(ik, kv, len(added))
                        for ix in idx:
                            if ix not in selected_idx:
                                selected_idx.append(ix)
                                added.append(atoms[ix])
                    else:
                        for ix in idx:
                            if ix not in selected_idx:
                                selected_idx.append(ix)
                                added.append(atoms[ix])
                        if len(added) >= np.floor(system_count[csys] * train_split[csys]):
                            # print('bruch @ ', ik, kv, len(added))
                            break

                for at in added:
                    if 'stress' in at.arrays:
                        del at.arrays['stress']

                if species not in suggestions:
                    suggestions[species] = {csys: added}
                else:
                    suggestions[species].update({csys: added})

                print('selected {} xyz'.format(len(added)))

        with open('soap_{}.json'.format(csys), 'w') as f:
            json.dump(obj=all_kernels, fp=f)

else:

    for csys in systems:
        selected_idx = []
        all_kernels = dict()
        for species in ['Cu', 'Au']:
            if species not in csys:
                continue

            print('Load kernels for {}...'.format(csys))

            tmp = []
            for idx, st in enumerate(complete_xyz):
                if crystal_system[idx] != csys:
                    continue
                tmp.append(st)

            with open('/tmp/all_{}.xyz'.format(csys), 'w') as f:
                for st in tmp:
                    for line in st:
                        f.write(line)
            atoms = read('/tmp/all_{}.xyz'.format(csys), index=':')

            with open('soap_{}.json'.format(csys), 'r') as f:
                all_kernels = json.load(fp=f)

            added = []
            print('{} kernels...'.format(len(all_kernels[species]['kernel'])), end='')
            sys.stdout.flush()
            for ik, (kv, idx) in enumerate(sorted(zip(all_kernels[species]['kernel'],
                                                      all_kernels[species]['index']))):
                if kv < 0.5:
                    # if csys == 'fcc_AuCu':
                    #     print(ik, kv, len(added))
                    for ix in idx:
                        if ix not in selected_idx:
                            selected_idx.append(ix)
                            added.append(atoms[ix])
                else:
                    for ix in idx:
                        if ix not in selected_idx:
                            selected_idx.append(ix)
                            added.append(atoms[ix])
                    if len(added) >= np.floor(system_count[csys] * train_split[csys]):
                        # print('bruch @ ', ik, kv, len(added))
                        break

            for at in added:
                if 'stress' in at.arrays:
                    del at.arrays['stress']

            if species not in suggestions:
                suggestions[species] = {csys: added}
            else:
                suggestions[species].update({csys: added})

            print('selected {} xyz'.format(len(added)))


for species in suggestions:
    for csys in systems:
        if species not in csys:
            continue
        print(species, csys)
        print('Take {} structures for {} in {}'.format(len(suggestions[species][csys]), species, csys))
        for at in suggestions[species][csys]:
            at.info['config_type'] = csys

            mask = np.random.choice([False, True], size=len(at), p=[force_fraction, 1.-force_fraction])
            at.new_array('force_mask', mask)

            f_min = 0.1
            f_scale = 0.1
            f = np.clip(np.abs(at.get_forces()) * f_scale, a_min=f_min, a_max=None)
            at.new_array("force_component_sigma", f)

            file = io.StringIO()
            write(filename=file, images=at, format='extxyz', parallel=False)
            file.seek(0)
            xyz = file.readlines()

            processed[csys].append(xyz)


# for i, xyz in enumerate(complete_xyz):
#     with open('/tmp/delme', 'w') as f:
#         for line in xyz:
#             f.write(line)
#
#     atoms = read('/tmp/delme')
#     atoms.info['config_type'] = crystal_system[i]
#
#     mask = np.random.choice([False, True], size=len(atoms), p=[force_fraction, 1.-force_fraction])
#     atoms.new_array('force_mask', mask)
#
#     f_min = 0.1
#     f_scale = 0.1
#     f = np.clip(np.abs(atoms.get_forces()) * f_scale, a_min=f_min, a_max=None)
#     atoms.new_array("force_component_sigma", f)
#
#     file = io.StringIO()
#     write(filename=file, images=atoms, format='extxyz', parallel=False)
#     file.seek(0)
#     xyz = file.readlines()
#
#     processed[crystal_system[i]].append(xyz)


with open('train.xyz', 'w') as f:
    for at in ['Au', 'Cu']:
        with open(os.path.join('../training_data/atom', at, 'parsed.xyz'), 'r') as f_in:
            f.write(f_in.read())
    for at in ['CuCu', 'AuAu', 'CuAu']:
        with open(os.path.join('../training_data/dimer', '{}_dimer.xyz'.format(at)), 'r') as f_in:
            f.write(f_in.read())
    for sys in systems:
        for i, xyz in enumerate(processed[sys]):
            if train_selected[sys][i]:
                for line in xyz:
                    f.write(line)

if read_from_db:
    with open('/tmp/delme', 'w') as f:
        for xyz in complete_xyz:
            for line in xyz:
                f.write(line)
    atoms = read('/tmp/delme', index=':')
    for ia, at in enumerate(atoms):
        at.info['config_type'] = crystal_system[ia]
        if 'stress' in at.arrays:
            del at.arrays['stress']
    write(filename='test.xyz', images=atoms, format='extxyz')
