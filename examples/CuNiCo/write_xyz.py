import io
import os
import sys
import json
import pymongo
import numpy as np
import tensorflow as tf
from mpi4py import MPI
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


def rescale_forces(forces_in, max_force, scale_factor):
    scaled_forces = []
    for f_comp in np.abs(forces_in):
        for fc in f_comp:
            scaled_forces.append([max_force * scale_factor if fc <= max_force else fc * scale_factor])
    return np.array(scaled_forces).reshape(len(forces_in), 3)


mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
work_gpu = 0
gpu_map = []
for i in range(mpi_size):
    if i % np.floor(mpi_size/num_gpus) == 0 and i != 0:
        work_gpu += 1
    gpu_map.append(work_gpu)
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, enable=True)
print('CPU {} using {}'.format(mpi_rank, gpus[gpu_map[mpi_rank]]))


read_from_db = True
write_testing_data = True
do_soap = True
select_from_soap = True
force_fraction = 1  # percentage of forces to INCLUDE in training
order = 'top_bottom'  # offset || linear || top_bottom
np.random.seed(1410)  # fix for reproduction

enable_force_component_regularization = True
f_max = 1.0  # maximum unscaled force
f_scale = 0.1  # scaling when forces exceed f_max


systems = ['fcc_CoCuNi',
           'fcc_CuNi',
           'fcc_CoCu',
           'fcc_CoNi']

train_split = {'fcc_CoCuNi': 0.15,
               'fcc_CuNi': 0.15,
               'fcc_CoCu': 0.15,
               'fcc_CoNi': 0.15}

# train_split = {'fcc_Cu': 0.0, 'bcc_Cu': 0.0, 'hcp_Cu': 0.0, 'sc_Cu': 0.0, 'slab_Cu': 0.0,
#                'fcc_Au': 0.0, 'bcc_Au': 0.0, 'hcp_Au': 0.0, 'sc_Au': 0.0, 'slab_Au': 0.0, 'cluster_Au': 0.0,
#                'fcc_AuCu': 0.0, 'bcc_AuCu': 0.0, 'hcp_AuCu': 0.0, 'sc_AuCu': 0.0,
#                'embedded_AuCu': 0.1, 'cluster_AuCu': 0.1}


complete_xyz = None
crystal_system = None
if mpi_rank == 0:
    if read_from_db:
        ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
        cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
        run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True,
                                      ssl_ca_certs=ca_file, ssl_certfile=cl_file)
        data_db = run_con.pot_train
        data_db.authenticate('jank', 'b@sf_mongo')
        data_coll = data_db['CuNiCo']
        complete_xyz = []
        crystal_system = []
        print('Starting DB read for CuNiCo of {} entries...'.format(data_coll.estimated_document_count()))
        print('busy on: ', end='')
        sys.stdout.flush()
        ik = 0
        # f_out = open('data_dates', 'w')
        for doc in data_coll.find({}):
            if ik % 1000 == 0:
                print(' {:d}'.format(ik), end='')
                sys.stdout.flush()
            ik += 1

            # f_out.write(doc['name']+'\n')

            # if 'CuAu CASM generated and relaxed structure for multiplication and rnd distortion' not in doc['name']:
            #     continue

            valid = True

            if 'slab' in doc['name'].lower():
                valid = check_vacuum_direction(doc['data']['final_structure'])

            if valid:
                complete_xyz.append(doc['data']['xyz'])
                if 'slab' in doc['name'].lower():
                    crystal_system.append('slab_CuNiCo')
                elif 'cluster' in doc['name'].lower():
                    crystal_system.append('cluster_CuNiCo')
                elif 'embedded' in doc['name'].lower():
                    crystal_system.append('embedded_CuNiCo')
                else:
                    crystal_system.append(doc['name'].split('||')[1].split(' ')[5])
        print('')

        # f_out.close()

        # data_coll = data_db['cuprum']
        # print('Starting DB read for Cu of {} entries...'.format(data_coll.estimated_document_count()))
        # print('busy on: ', end='')
        # sys.stdout.flush()
        # ik = 0
        # for doc in data_coll.find({}):
        #     if ik % 1000 == 0:
        #         print(' {:d}'.format(ik), end='')
        #         sys.stdout.flush()
        #     ik += 1
        #
        #     valid = True
        #
        #     if 'slab' in doc['name'].lower():
        #         valid = check_vacuum_direction(doc['data']['final_structure'])
        #
        #     if valid:
        #         complete_xyz.append(doc['data']['xyz'])
        #         if 'slab' in doc['name'].lower():
        #             crystal_system.append('slab_Cu')
        #         elif 'cluster' in doc['name'].lower():
        #             crystal_system.append('cluster_Cu')
        #         else:
        #             crystal_system.append(doc['name'].split('||')[1].split(' ')[5]+'_Cu')
        # print('')

        with open('structures.json', 'w') as f:
            json.dump(complete_xyz, f)
        with open('systems.json', 'w') as f:
            json.dump(crystal_system, f)
    else:
        with open('structures.json', 'r') as f:
            complete_xyz = json.load(f)
        with open('systems.json', 'r') as f:
            crystal_system = json.load(f)


complete_xyz = mpi_comm.bcast(complete_xyz, root=0)
crystal_system = mpi_comm.bcast(crystal_system, root=0)

all_selected = []
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


if mpi_rank == 0:
    print('')
    print('There\'s currently {} computed structures in the database'.format(len(complete_xyz)))
    print('')


processed = list()

local_systems = []
for isys, csys in enumerate(systems):
    if isys % mpi_size == mpi_rank:
        local_systems.append(csys)

local_xyz = []
local_csys = []
for csys in local_systems:
    for idx, st in enumerate(complete_xyz):
        if crystal_system[idx] == csys:
            local_xyz.append(st)
            local_csys.append(csys)

if mpi_rank != 0:
    complete_xyz = []

# print(mpi_rank, local_systems)

suggestions = dict()
if select_from_soap:
    if do_soap:
        for csys in local_systems:
            if train_split[csys] == 0.0:
                continue
            selected_idx = []
            all_kernels = dict()
            for ci, species in enumerate(['Co', 'Ni', 'Cu']):
                if species not in csys:
                    continue
                divider = len(csys.split('_')[1])/2

                tmp = []
                for idx, st in enumerate(local_xyz):
                    if local_csys[idx] == csys:
                        tmp.append(st)

                file = io.StringIO()
                for st in tmp:
                    file.writelines(st)
                file.seek(0)
                atoms = read(filename=file, format='extxyz', index=':', parallel=False)

                added = []
                out_string = 'CPU{} SOAP processing {} {}: Species {}...'.format(mpi_rank, len(atoms), csys, species)

                q = dict()
                d = Descriptor('soap_turbo l_max=8 alpha_max={8 8 8} atom_sigma_r={0.5 0.5 0.5} '
                               'atom_sigma_t={0.5 0.5 0.5} atom_sigma_r_scaling={0.0 0.0 0.0} '
                               'atom_sigma_t_scaling={0.0 0.0 0.0} rcut_hard=5.5 rcut_soft=5.0 basis=poly3gauss '
                               'scaling_mode=polynomial amplitude_scaling={1.0 1.0 1.0} '
                               'n_species=3 species_Z={27 28 29} radial_enhancement=1 compress_file=compress.dat '
                               'central_index='+str(ci+1)+' central_weight={1.0 1.0 1.0} add_species=F')

                num_desc = 0
                qtmp = []
                for at in atoms:
                    qats = ase_to_quip(at)
                    qats.set_cutoff(cutoff=6.0)
                    qats.calc_connect()
                    desc = d.calc_descriptor(qats)
                    qtmp.append(tf.constant(np.array(desc), dtype=tf.float16, shape=desc.shape))
                    num_desc += desc.shape[0]
                q[species] = qtmp

                out_string += '{} descriptors computed...'.format(num_desc)

                kernel = []
                indexes = []
                with tf.device('/device:GPU:{}'.format(gpu_map[mpi_rank])):
                    for ik in range(len(q[species])):
                        for il in range(ik + 1, len(q[species])):
                            tf_result = tf.reduce_sum(tf.matmul(q[species][ik], tf.transpose(q[species][il])))
                            kernel.append(tf_result.numpy() / (q[species][ik].shape[0] * q[species][il].shape[0]))
                            indexes.append([ik, il])
                    all_kernels[species] = {'kernel': kernel, 'index': indexes}

                with tf.device('/device:CPU:{}'.format(mpi_rank)):
                    offset = np.floor(len(atoms) / (np.floor(len(atoms) * train_split[csys])))
                    out_string += '{} kernels...'.format(len(all_kernels[species]['kernel']))

                    if order == 'offset':
                        for ik, (kv, idx) in enumerate(sorted(zip(all_kernels[species]['kernel'],
                                                                  all_kernels[species]['index']))):
                            for ix in idx:
                                if np.mod(ik, offset) == 0:
                                    if ix not in selected_idx:
                                        selected_idx.append(ix)
                                        added.append(atoms[ix])
                            if len(added) >= np.floor(system_count[csys] * train_split[csys]):
                                break
                    elif order == 'top_bottom':
                        for ik, (kv, idx) in enumerate(sorted(zip(all_kernels[species]['kernel'],
                                                                  all_kernels[species]['index']))):
                            for ix in idx:
                                if ix not in selected_idx:
                                    selected_idx.append(ix)
                                    added.append(atoms[ix])
                            if len(added) >= np.floor(system_count[csys] * train_split[csys] / 2):
                                break

                        for ik, (kv, idx) in enumerate(reversed(sorted(zip(all_kernels[species]['kernel'],
                                                                           all_kernels[species]['index'])))):
                            for ix in idx:
                                if ix not in selected_idx:
                                    selected_idx.append(ix)
                                    added.append(atoms[ix])
                            if len(added) >= np.floor(system_count[csys] * train_split[csys]):
                                break
                    elif order == 'linear':
                        for ik, (kv, idx) in enumerate(sorted(zip(all_kernels[species]['kernel'],
                                                                  all_kernels[species]['index']))):
                            if kv < 0.5:
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
                                    break
                    else:
                        raise NotImplementedError('Invalid order specified...')

                    if species not in suggestions:
                        suggestions[species] = {csys: added}
                    else:
                        suggestions[species].update({csys: added})

                    out_string += 'selected {} {} xyz'.format(len(added), species)
                    print(out_string)
                    sys.stdout.flush()

            with open('soap_{}.json'.format(csys), 'w') as f:
                json.dump(obj=all_kernels, fp=f)

    else:
        for csys in local_systems:
            if train_split[csys] == 0.0:
                continue
            selected_idx = []
            all_kernels = dict()
            for species in ['Cu', 'Ni', 'Co']:
                if species not in csys:
                    continue

                tmp = []
                for idx, st in enumerate(local_xyz):
                    if local_csys[idx] == csys:
                        tmp.append(st)

                file = io.StringIO()
                for st in tmp:
                    file.writelines(st)
                file.seek(0)
                atoms = read(filename=file, format='extxyz', index=':', parallel=False)

                out_string = 'CPU{}: processed {} {} species {}...'.format(mpi_rank, len(atoms), csys, species)
                added = []

                with open('soap_{}.json'.format(csys), 'r') as f:
                    all_kernels = json.load(fp=f)

                offset = np.floor(len(atoms) / (np.floor(len(atoms) * train_split[csys])))
                out_string += '{} kernels...'.format(len(all_kernels[species]['kernel']))
                if order == 'offset':
                    for ik, (kv, idx) in enumerate(sorted(zip(all_kernels[species]['kernel'],
                                                              all_kernels[species]['index']))):
                        for ix in idx:
                            if np.mod(ik, offset) == 0:
                                added.append(atoms[ix])
                        if len(added) >= np.floor(system_count[csys] * train_split[csys]):
                            break
                elif order == 'top_bottom':
                    for ik, (kv, idx) in enumerate(sorted(zip(all_kernels[species]['kernel'],
                                                              all_kernels[species]['index']))):
                        for ix in idx:
                            if ix not in selected_idx:
                                selected_idx.append(ix)
                                added.append(atoms[ix])
                        if len(added) >= np.floor(system_count[csys] * train_split[csys] / 2):
                            break
                    for ik, (kv, idx) in enumerate(reversed(sorted(zip(all_kernels[species]['kernel'],
                                                                       all_kernels[species]['index'])))):
                        for ix in idx:
                            if ix not in selected_idx:
                                selected_idx.append(ix)
                                added.append(atoms[ix])
                        if len(added) >= np.floor(system_count[csys] * train_split[csys]):
                            break
                elif order == 'linear':
                    for ik, (kv, idx) in enumerate(sorted(zip(all_kernels[species]['kernel'],
                                                              all_kernels[species]['index']))):
                        for ix in idx:
                            if ix not in selected_idx:
                                selected_idx.append(ix)
                                added.append(atoms[ix])
                        if len(added) >= np.floor(system_count[csys] * train_split[csys]):
                            break
                else:
                    raise NotImplementedError('Invalid order specified...')

                if species not in suggestions:
                    suggestions[species] = {csys: added}
                else:
                    suggestions[species].update({csys: added})

                out_string += 'selected {} {} xyz'.format(len(added), species)
                print(out_string)
                sys.stdout.flush()

    for species in suggestions:
        for csys in local_systems:
            if train_split[csys] == 0.0:
                continue
            if species not in csys:
                continue
            print('CPU{} takes {} structures for {} in {}'.format(mpi_rank, len(suggestions[species][csys]),
                                                                  species, csys))
            for at in suggestions[species][csys]:
                if 'cluster' in csys:
                    at.info['virial'] = '0 0 0 0 0 0 0 0 0'

                if 'stress' in at.arrays:
                    del at.arrays['stress']

                at.info['config_type'] = csys

                mask = np.random.choice([False, True], size=len(at), p=[force_fraction, 1.-force_fraction])
                at.new_array('force_mask', mask)

                if enable_force_component_regularization:
                    f = rescale_forces(at.get_forces(), max_force=f_max, scale_factor=f_scale)
                    at.new_array("force_component_sigma", f)

                file = io.StringIO()
                write(filename=file, images=at, format='extxyz', parallel=False)
                file.seek(0)
                xyz = file.readlines()

                processed.append(xyz)

    processed = mpi_comm.gather(processed, root=0)

    if mpi_rank == 0:
        print('Writing training data...')
        with open('train.xyz', 'w') as f:
            for at in ['Cu', 'Ni', 'Co']:
                with open(os.path.join('../training_data/atoms', at, 'parsed.xyz'), 'r') as f_in:
                    f.write(f_in.read())
            for at in ['CuCu', 'CuNi', 'CuCo', 'NiNi', 'NiCo', 'CoCo']:
                with open(os.path.join('../training_data/dimers', '{}_dimer.xyz'.format(at)), 'r') as f_in:
                    f.write(f_in.read())
            for xyz in processed:
                for entry in xyz:
                    f.writelines(entry)

else:
    if mpi_rank == 0:
        print('Processing training data...')
        for csys in systems:
            print('{} {}, take {}'.format(system_count[csys], csys, int(train_split[csys]*system_count[csys])))

            tmp = []
            for idx, st in enumerate(complete_xyz):
                if crystal_system[idx] == csys:
                    tmp.append(st)

            file = io.StringIO()
            for st in tmp:
                file.writelines(st)
            file.seek(0)
            atoms = read(filename=file, format='extxyz', index=':', parallel=False)

            for ia, at in enumerate(atoms):
                if not train_selected[csys][ia]:
                    continue
                if 'cluster' in csys:
                    at.info['virial'] = '0 0 0 0 0 0 0 0 0'

                if 'stress' in at.arrays:
                    del at.arrays['stress']

                at.info['config_type'] = csys

                mask = np.random.choice([False, True], size=len(at), p=[force_fraction, 1.-force_fraction])
                at.new_array('force_mask', mask)

                if enable_force_component_regularization:
                    f = rescale_forces(at.get_forces(), max_force=f_max, scale_factor=f_scale)
                    at.new_array("force_component_sigma", f)

                file = io.StringIO()
                write(filename=file, images=at, format='extxyz', parallel=False)
                file.seek(0)
                xyz = file.readlines()
                processed.append(xyz)

        print('Writing training data...')
        with open('train.xyz', 'w') as f:
            for at in ['Cu', 'Ni', 'Co']:
                with open(os.path.join('../training_data/atoms', at, 'parsed.xyz'), 'r') as f_in:
                    f.write(f_in.read())
            for at in ['CuCu', 'CuNi', 'CuCo', 'NiNi', 'NiCo', 'CoCo']:
                with open(os.path.join('../training_data/dimers', '{}_dimer.xyz'.format(at)), 'r') as f_in:
                    f.write(f_in.read())
            for xyz in processed:
                for entry in xyz:
                    f.writelines(entry)

if write_testing_data:
    if mpi_rank == 0:
        print('\nProcessing Testing data...')
        file = io.StringIO()
        for xyz in complete_xyz:
            file.writelines(xyz)
        file.seek(0)
        atoms = read(filename=file, format='extxyz', index=':', parallel=False)

        for ia, at in enumerate(atoms):
            at.info['config_type'] = crystal_system[ia]
            if 'stress' in at.arrays:
                del at.arrays['stress']
        write(filename='test.xyz', images=atoms, format='extxyz', parallel=False)

MPI.Finalize()
