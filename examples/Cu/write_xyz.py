import io
import os
import sys
import json
import pymongo
import numpy as np
import tensorflow as tf
from ase.io import read, write
from pymatgen import Structure
from quippy.descriptors import Descriptor
from quippy.convert import ase_to_quip


def check_vacuum_direction(input_data):
    structure = Structure.from_dict(input_data)
    a = structure.lattice.matrix[0] / 2
    b = structure.lattice.matrix[1] / 2
    for c in structure.cart_coords:
        if np.linalg.norm(c - np.array([a + b])) < 6:
            return False
    return True


tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, enable=True)


np.random.seed(1410)  # fix for reproduction

read_from_db = False
force_fraction = 1  # percentage of forces to INCLUDE for training
do_soap = False


systems = ['fcc', 'bcc', 'hcp', 'sc', 'slab']  # , 'cluster', 'addition']
train_split = {'fcc': 1, 'bcc': 1, 'hcp': 1, 'sc': 1, 'slab': 1}  # , 'cluster': 1, 'addition': 1}

if read_from_db:
    ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
    cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
    run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
    data_db = run_con.pot_train
    data_db.authenticate('jank', 'b@sf_mongo')
    data_coll = data_db['cuprum']
    complete_xyz = []
    crystal_system = []
    print('Starting DB read of {} entries...'.format(data_coll.estimated_document_count()))
    print('busy on: ', end='')
    sys.stdout.flush()
    ik = 0
    for doc in data_coll.find({}):
        if ik % 500 == 0:
            print(' {:d}'.format(ik), end='')
            sys.stdout.flush()
        ik += 1

        valid = True

        if 'slab' in doc['name']:
            valid = check_vacuum_direction(doc['data']['final_structure'])

        if valid:
            complete_xyz.append(doc['data']['xyz'])
            if 'slab' in doc['name']:
                crystal_system.append('slab')
            elif 'cluster' in doc['name']:
                crystal_system.append('cluster')
            else:
                crystal_system.append(doc['name'].split('||')[1].split(' ')[5])

    with open('structures.json', 'w') as f:
        json.dump(complete_xyz, f)
    with open('systems.json', 'w') as f:
        json.dump(crystal_system, f)
    print('')
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
# print('Including in training DB: fcc     : {:5d} [{:3.1f}%]\n'
#       '                          bcc     : {:5d} [{:3.1f}%]\n'
#       '                          sc      : {:5d} [{:3.1f}%]\n'
#       '                          hcp     : {:5d} [{:3.1f}%]\n'
#       '                          slab    : {:5d} [{:3.1f}%]\n'
#       '                          cluster : {:5d} [{:3.1f}%]\n'
#       '                          addition: {:5d} [{:3.1f}%]'.
#       format(int(system_count['fcc'] * train_split['fcc']), train_split['fcc']*100,
#              int(system_count['bcc'] * train_split['bcc']), train_split['bcc']*100,
#              int(system_count['sc'] * train_split['sc']), train_split['sc']*100,
#              int(system_count['hcp'] * train_split['hcp']), train_split['hcp']*100,
#              int(system_count['slab'] * train_split['slab']), train_split['slab']*100,
#              int(system_count['cluster'] * train_split['cluster']), train_split['cluster']*100,
#              int(system_count['addition'] * train_split['addition']), train_split['addition']*100))

suggestions = dict()
if do_soap:
    for csys in systems:
        selected_idx = []
        print('SOAP processing {}: '.format(csys))
        all_kernels = dict()
        for ci, species in enumerate(['Cu']):
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

            d = Descriptor('soap_turbo l_max=8 alpha_max=8 atom_sigma_r=0.5 atom_sigma_t=0.5  '
                           'atom_sigma_r_scaling=0.0 atom_sigma_t_scaling=0.0 rcut_hard=5.7 '
                           'rcut_soft=5.2 basis=poly3gauss scaling_mode=polynomial amplitude_scaling=1.0 '
                           'n_species=1 species_Z=29 radial_enhancement=1 compress_file=compress.dat '
                           'central_weight=1.0 add_species=F')

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
                        tf_result = tf.reduce_sum(tf.matmul(q[species][ik], tf.transpose(q[species][il])))
                        kernel.append(tf_result.numpy() / (q[species][ik].shape[0] * q[species][il].shape[0]))
                        indexes.append([ik, il])
                all_kernels[species] = {'kernel': kernel, 'index': indexes}

            with tf.device('/device:CPU:0'):  # .format(0)):
                added = []
                print('{} kernels...'.format(len(all_kernels[species]['kernel'])), end='')
                sys.stdout.flush()
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
        for species in ['Cu']:
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

            for at in added:
                if 'stress' in at.arrays:
                    del at.arrays['stress']

            if species not in suggestions:
                suggestions[species] = {csys: added}
            else:
                suggestions[species].update({csys: added})

            print('selected {} xyz'.format(len(added)))


processed = {'fcc': [], 'bcc': [], 'hcp': [], 'sc': [], 'slab': []}  # , 'cluster': [], 'addition': []}
for species in suggestions:
    for csys in systems:
        print('Take {} structures for {} in {}'.format(len(suggestions[species][csys]), species, csys))
        for at in suggestions[species][csys]:
            at.info['config_type'] = csys

            mask = np.random.choice([True, False], size=len(at), p=[force_fraction, 1.-force_fraction])
            at.new_array('force_mask', mask)

            if csys == 'cluster':
                stress = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                stress = at.get_stress(voigt=False)

            vol = at.get_volume()
            virial = -np.dot(vol, stress)

            at.info['virial'] = virial

            f_min = 0.1
            f_scale = 0.1
            f = np.clip(np.abs(at.get_forces()) * f_scale, a_min=f_min, a_max=None)
            at.new_array("force_component_sigma", f)

            file = io.StringIO()
            write(filename=file, images=at, format='extxyz', parallel=False)
            file.seek(0)
            xyz = file.readlines()

            processed[csys].append(xyz)


for pr in processed:
    print('{}: {}'.format(pr, len(processed[pr])))


with open('train.xyz', 'w') as f:
    with open('../training_data/atom/parsed.xyz', 'r') as f_in:
        f.write(f_in.read())
    with open('../training_data/dimer/CuCu_dimer.xyz', 'r') as f_in:
        f.write(f_in.read())
    for csys in systems:
        for i, xyz in enumerate(processed[csys]):
            if train_selected[csys][i]:
                for line in xyz:
                    f.write(line)

with open('test.xyz', 'w') as f:
    for csys in systems:
        for xyz in processed[csys]:
            for line in xyz:
                f.write(line)
