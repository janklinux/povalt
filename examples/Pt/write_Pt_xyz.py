import io
import os
import sys
import json
import pymongo
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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


tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, enable=True)


np.random.seed(1410)  # fix for reproduction

read_from_db = False
write_test_data = False
force_fraction = 0  # percentage of forces to EXCLUDE from training
show_dimer = False
do_soap = False
select_from_soap = False
order = 'top_bottom'  # offset | linear | top_bottom


systems = ['fcc', 'bcc', 'hcp', 'sc', 'slab', 'cluster', 'addition', 'phonons', 'trimer', 'elastics', 'random']

# train_split = {'fcc': 1.0, 'bcc': 0.9, 'hcp': 0.9, 'sc': 0.9, 'slab': 1.0,
#                'cluster': 1.0, 'phonons': 1.0, 'addition': 1.0, 'trimer': 0.0,
#                'elastics': 1.0, 'random': 0.0}

train_split = {'fcc': 1.0, 'bcc': 1.0, 'hcp': 1.0, 'sc': 1.0, 'slab': 1.0,
               'cluster': 1.0, 'phonons': 1.0, 'addition': 1.0, 'trimer': 1.0,
               'elastics': 1.0, 'random': 1.0}


with open('train.xyz', 'w') as f:
    f.write('1\n')
    f.write('Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3:forces:R:3:force_mask:L:1 '
            'energy=-0.52810937 stress="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" '
            'virial="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" '
            'free_energy=-0.54289024 pbc="T T T" '
            'config_type=isolated_atom\n')
    f.write('Pt 0.0 0.0 0.0 0.0 0.0 0.0 0\n')

dimer_curve = list()
force_curve = list()
zcrds = np.round(np.arange(0.7, 8.1, 0.1, dtype=np.float16), 1)

for crd in zcrds:
    atoms = read(os.path.join('/home/jank/work/Aalto/GAP_data/Pt/training_data/dimer/manual_run',
                              str(crd), 'vasprun.xml'))
    dimer_curve.append(atoms.get_potential_energy(force_consistent=True))
    atoms.info['config_type'] = 'dimer'

    xyz = ''
    file = io.StringIO()
    write(filename=file, images=atoms, format='extxyz')
    file.seek(0)
    for f in file:
        xyz += f
    file.close()

    tmp = ''
    tmp_line = []
    for line in xyz:
        tmp += line
        if '\n' in line:
            tmp_line.append(tmp)
            tmp = ''

    s_pos = 0
    l_pos = 0
    s_append = False
    l_append = False
    stress = []
    lattice = []
    for i, el in enumerate(tmp_line[1].split()):
        if s_append:
            if '"' in el:
                el = el[:-1]
            stress.append(float(el))
            if i > s_pos + 7:
                s_append = False
        if 'stress' in el:
            s_pos = i
            stress.append(float(el.split('"')[1]))
            s_append = True
        if l_append:
            if '"' in el:
                el = el[:-1]
            lattice.append(float(el))
            if i > l_pos + 7:
                l_append = False
        if 'Lattice' in el:
            l_pos = i
            lattice.append(float(el.split('"')[1]))
            l_append = True

    stress = np.array([stress[0:3], stress[3:6], stress[6:9]])
    lattice = np.array([lattice[0:3], lattice[3:6], lattice[6:9]])
    vol = np.abs(np.dot(lattice[2], np.cross(lattice[0], lattice[1])))
    virial = - np.dot(vol, stress)

    wtmp = ''
    for bit in tmp_line[1].split(' '):
        if 'Properties' in bit:
            bit += ':force_mask:L:1:force_component_sigma:R:3 '
            bit += 'virial="{} {} {} {} {} {} {} {} {}"'.format(
                virial[0][0], virial[0][1], virial[0][2],
                virial[1][0], virial[1][1], virial[1][2],
                virial[2][0], virial[1][2], virial[2][2])
        wtmp += bit + ' '
    tmp_line[1] = wtmp.strip() + '\n'
    tmp_line[0] = tmp_line[0].strip() + '\n'

    forces = []
    for j in range(2, len(tmp_line)):
        forces.append([float(x) for x in tmp_line[j].split()[4:7]])

    f_min = 0.1
    f_scale = 0.1
    f = np.clip(np.abs(forces) * f_scale, a_min=f_min, a_max=None)

    for j in range(2, len(tmp_line)):
        tmp_line[j] = tmp_line[j].strip() + '   0  {:5.5f} {:5.5f} {:5.5f}\n'.format(f[j-2][0], f[j-2][1], f[j-2][2])

    with open('train.xyz', 'a') as f:
        for line in tmp_line:
            f.write(line)

    force_curve.append(atoms.get_forces()[1][2])


if show_dimer:
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', serif='Palatino')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = 'cm'
    fig, ax1 = plt.subplots()
    color = 'red'
    ax1.plot(zcrds, dimer_curve, '.-', color=color)
    ax1.set_xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
    ax1.set_ylabel(r'Free Energy [eV]', fontsize=22, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'navy'
    ax2.plot(zcrds, force_curve, '.-', color=color)
    ax2.set_ylabel(r'Force in sep dir [eV/\AA]', fontsize=22, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()
    plt.close()


if read_from_db:
    ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
    cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
    run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
    data_db = run_con.pot_train
    data_db.authenticate('jank', 'b@sf_mongo')
    data_coll = data_db['platinum']
    add_coll = data_db['platinum_additions']

    complete_xyz = []
    crystal_system = []
    print('Starting DB read...')
    ik = 0
    total_slabs = 0
    added_slabs = 0
    print('busy on src:', end='')
    sys.stdout.flush()
    for doc in data_coll.find({}, {'_id': 0, 'data.xyz': 1, 'name': 1, 'data.final_structure': 1}):
        if ik % 1000 == 0:
            print(' {:d}'.format(ik), end='')
            sys.stdout.flush()
        ik += 1

        valid = True
        if 'slab' in doc['name'].lower():
            total_slabs += 1
            valid = check_vacuum_direction(doc['data']['final_structure'])

        if valid:
            complete_xyz.append(doc['data']['xyz'])
            if 'slab' in doc['name'].lower():
                added_slabs += 1
                crystal_system.append('slab')
            elif 'cluster' in doc['name'].lower():
                crystal_system.append('cluster')
            elif 'phonons' in doc['name'].lower():
                crystal_system.append('phonons')
            elif 'elastics' in doc['name'].lower():
                crystal_system.append('elastics')
            else:
                crystal_system.append(doc['name'].split('||')[1].split(' ')[5].lower())
    print('')
    print('busy on add:', end='')
    sys.stdout.flush()
    ik = 0
    for doc in add_coll.find({}, {'_id': 0, 'data.xyz': 1, 'name': 1}):
        if ik % 100 == 0:
            print(' {:d}'.format(ik), end='')
            sys.stdout.flush()
        ik += 1
        complete_xyz.append(doc['data']['xyz'])
        crystal_system.append('addition')
    print('')

    with open('structures.json', 'w') as f:
        json.dump(complete_xyz, f)
    with open('systems.json', 'w') as f:
        json.dump(crystal_system, f)
    print('Slabs total: {} -- added: {}'.format(total_slabs, added_slabs))
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


print('')
print('There\'s currently {} computed structures in the database'.format(len(complete_xyz)))
for sc in system_count:
    print('{}: {}'.format(sc, system_count[sc]))
print('')


all_xyz = dict()
for csys in systems:
    tmp = []
    for idx, st in enumerate(complete_xyz):
        if crystal_system[idx] == csys:
            tmp.append(st)
    all_xyz[csys] = tmp

processed = dict()
for csys in systems:
    processed[csys] = []

if select_from_soap:
    suggestions = dict()
    if do_soap:
        for csys in systems:
            if train_split[csys] == 0.0:
                continue
            selected_idx = []
            print('SOAP processing {}: '.format(csys))
            all_kernels = dict()
            for ci, species in enumerate(['Pt']):
                print('Species {} in {}...'.format(species, csys), end='')
                sys.stdout.flush()

                file = io.StringIO()
                for st in all_xyz[csys]:
                    file.writelines(st)
                file.seek(0)
                atoms = read(filename=file, format='extxyz', index=':')

                q = dict()
                d = Descriptor('soap_turbo l_max=8 alpha_max=8 atom_sigma_r=0.5 atom_sigma_t=0.5  '
                               'atom_sigma_r_scaling=0.0 atom_sigma_t_scaling=0.0 rcut_hard=5.2 '
                               'rcut_soft=4.7 basis=poly3gauss scaling_mode=polynomial amplitude_scaling=1.0 '
                               'n_species=1 species_Z=78 radial_enhancement=1 compress_file=compress.dat '
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
                    offset = np.floor(len(atoms) / (np.floor(len(atoms) * train_split[csys])))
                    print('{} kernels...'.format(len(all_kernels[species]['kernel'])), end='')
                    sys.stdout.flush()
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
                    else:
                        raise NotImplementedError('Invalid order specified...')

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
            added = []
            for species in ['Pt']:
                if train_split[csys] == 0.0:
                    continue

                file = io.StringIO()
                for st in all_xyz[csys]:
                    file.writelines(st)
                file.seek(0)
                atoms = read(filename=file, format='extxyz', index=':')

                if train_split[csys] == 1.0:
                    for at in atoms:
                        added.append(at)
                    if species not in suggestions:
                        suggestions[species] = {csys: added}
                    else:
                        suggestions[species].update({csys: added})
                    print('selected {} xyz from {} [100%]'.format(len(added), csys))
                    continue

                print('Load kernels for {} {}...'.format(len(all_xyz[csys]), csys), end='')
                sys.stdout.flush()

                with open('soap_{}.json'.format(csys), 'r') as f:
                    all_kernels = json.load(fp=f)

                offset = np.floor(len(atoms) / (np.floor(len(atoms) * train_split[csys])))
                print('{} kernels...'.format(len(all_kernels[species]['kernel'])), end='')
                sys.stdout.flush()
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
                else:
                    raise NotImplementedError('Invalid order specified...')

                if species not in suggestions:
                    suggestions[species] = {csys: added}
                else:
                    suggestions[species].update({csys: added})

                print('selected {} xyz'.format(len(added)))

    print('\n')
    for species in suggestions:
        for csys in systems:
            if train_split[csys] == 0.0:
                continue
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

                if 'stress' in at.arrays:
                    del at.arrays['stress']

                f_min = 0.1
                f_scale = 0.1
                f = np.clip(np.abs(at.get_forces()) * f_scale, a_min=f_min, a_max=None)
                at.new_array("force_component_sigma", f)

                file = io.StringIO()
                write(filename=file, images=at, format='extxyz', parallel=False)
                file.seek(0)
                xyz = file.readlines()

                processed[csys].append(xyz)

    with open('train.xyz', 'a') as f:
        for csys in systems:
            if train_split[csys] == 0.0:
                continue
            for i, xyz in enumerate(processed[csys]):
                for line in xyz:
                    f.write(line)

else:

    train_selected = dict()
    for csys in systems:
        tmp = []
        for i in range(system_count[csys]):
            if i < system_count[csys] * train_split[csys]:
                tmp.append(True)
            else:
                tmp.append(False)
        train_selected[csys] = tmp
        np.random.shuffle(train_selected[csys])

    file = io.StringIO()
    for xyz in complete_xyz:
        file.writelines(xyz)
    atoms = read(filename=file, format='extxyz', index=':')

    for ia, at in enumerate(atoms):
        if train_split[crystal_system[ia]] == 0.0:
            continue
        if crystal_system[ia] == 'cluster':
            stress = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            stress = at.get_stress(voigt=False)

        vol = at.get_volume()
        virial = -np.dot(vol, stress)

        at.info['virial'] = virial
        at.info['config_type'] = crystal_system[ia]

        if 'stress' in at.arrays:
            del at.arrays['stress']

        f_min = 0.1
        f_scale = 0.1
        f = np.clip(np.abs(at.get_forces()) * f_scale, a_min=f_min, a_max=None)
        at.new_array("force_component_sigma", f)

        mask = np.random.choice([True, False], size=len(at), p=[force_fraction, 1.-force_fraction])
        at.new_array('force_mask', mask)

        file = io.StringIO()
        write(filename=file, images=at, format='extxyz', parallel=False)
        file.seek(0)
        xyz = file.readlines()

        processed[crystal_system[ia]].append(xyz)

    print('\nTraining Data Contents:')
    for pr in processed:
        print('{}: {}'.format(pr, int(system_count[pr]*train_split[pr])))

    with open('train.xyz', 'a') as f:
        for csys in systems:
            for i, xyz in enumerate(processed[csys]):
                if train_selected[csys][i]:
                    for line in xyz:
                        f.write(line)


if write_test_data:
    print('\nProcessing Testing Data...')
    with open('/tmp/delme.xyz', 'w') as f:
        for xyz in complete_xyz:
            for line in xyz:
                f.write(line)
    atoms = read('/tmp/delme.xyz', index=':')
    processed = dict()
    for csys in systems:
        processed[csys] = []

    for ia, at in enumerate(atoms):
        if crystal_system[ia] not in processed:
            continue
        if crystal_system[ia] == 'cluster':
            stress = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            stress = at.get_stress(voigt=False)

        if crystal_system[ia] == 'trimer':
            continue

        vol = at.get_volume()
        virial = -np.dot(vol, stress)

        at.info['virial'] = virial
        at.info['config_type'] = crystal_system[ia]

        f_min = 0.1
        f_scale = 0.1
        f = np.clip(np.abs(at.get_forces()) * f_scale, a_min=f_min, a_max=None)
        at.new_array("force_component_sigma", f)

        file = io.StringIO()
        write(filename=file, images=at, format='extxyz', parallel=False)
        file.seek(0)
        xyz = file.readlines()

        processed[crystal_system[ia]].append(xyz)

    print('\nTest Data Contents:')
    for pr in processed:
        print('{}: {}'.format(pr, len(processed[pr])))

    with open('test.xyz', 'w') as f:
        for csys in systems:
            for xyz in processed[csys]:
                for line in xyz:
                    f.write(line)
