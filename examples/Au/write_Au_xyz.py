import os
import re
import sys
import json
import numpy as np
import pymongo


np.random.seed(1410)  # fix for reproduction

read_from_db = False
force_fraction = 0  # percentage of forces to EXCLUDE from training
do_soap = False

systems = ['fcc', 'bcc', 'hcp', 'sc', 'slab', 'cluster', 'addition']

train_split = {'fcc': 1,
               'bcc': 1,
               'hcp': 1,
               'sc': 1,
               'slab': 1,
               'cluster': 0.9,
               'addition': 0.9}

if read_from_db:
    ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
    cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
    run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
    data_db = run_con.pot_train
    data_db.authenticate('jank', 'b@sf_mongo')
    data_coll = data_db['aurum']
    complete_xyz = []
    crystal_system = []
    print('Starting DB read of {} entries...'.format(data_coll.estimated_document_count()))
    print('busy on: ', end='')
    sys.stdout.flush()
    ik = 0
    for doc in data_coll.find({}, {'_id': 0, 'data.xyz': 1, 'name': 1}):
        if ik % 500 == 0:
            print(' {:d}'.format(ik), end='')
            sys.stdout.flush()
        ik += 1
        complete_xyz.append(doc['data']['xyz'])
        if 'Slab' in doc['name']:
            crystal_system.append('slab')
        elif 'Cluster' in doc['name']:
            crystal_system.append('cluster')
        else:
            crystal_system.append(doc['name'].split('||')[1].split(' ')[5])
    # ik = 0
    # for doc in add_coll.find({}, {'_id': 0, 'data.xyz': 1, 'name': 1}):
    #     if ik % 100 == 0:
    #         print('busy on add: {:d}'.format(ik))
    #     ik += 1
    #     complete_xyz.append(doc['data']['xyz'])
    #     crystal_system.append('addition')
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
for sys in systems:
    system_count[sys] = crystal_system.count(sys)
    tmp = []
    for i in range(system_count[sys]):
        if i < system_count[sys] * train_split[sys]:
            tmp.append(True)
        else:
            tmp.append(False)
        train_selected[sys] = tmp
        np.random.shuffle(train_selected[sys])

print('There\'s currently {} computed structures in the database'.format(len(complete_xyz)))
print('Including in training DB: fcc     : {:5d} [{:3.1f}%]\n'
      '                          bcc     : {:5d} [{:3.1f}%]\n'
      '                          sc      : {:5d} [{:3.1f}%]\n'
      '                          hcp     : {:5d} [{:3.1f}%]\n'
      '                          slab    : {:5d} [{:3.1f}%]\n'
      '                          cluster : {:5d} [{:3.1f}%]\n'
      '                          addition: {:5d} [{:3.1f}%]'.
      format(int(system_count['fcc'] * train_split['fcc']), train_split['fcc']*100,
             int(system_count['bcc'] * train_split['bcc']), train_split['bcc']*100,
             int(system_count['sc'] * train_split['sc']), train_split['sc']*100,
             int(system_count['hcp'] * train_split['hcp']), train_split['hcp']*100,
             int(system_count['slab'] * train_split['slab']), train_split['slab']*100,
             int(system_count['cluster'] * train_split['cluster']), train_split['cluster']*100,
             int(system_count['addition'] * train_split['addition']), train_split['addition']*100))
# print('This will need approximately {} GB of memory during training.'.format(np.round(
#     (system_count['fcc'] * train_split['fcc'] + system_count['bcc'] * train_split['bcc'] +
#      system_count['sc'] * train_split['sc'] + system_count['hcp'] * train_split['hcp'] +
#      system_count['slab'] * train_split['slab'] + system_count['cluster'] * train_split['cluster'] +
#      system_count['addition'] * train_split['addition']) * 8 * 1000 * len(systems) * 150 / 2**30 * 1.1, 2)))
# ( num systems * include % ) * #systems * 150 * 8 bytes * 1000 / GB + 10%
# |          dim1           | * |    dim2    | * numerics

processed = {'fcc': [], 'bcc': [], 'hcp': [], 'sc': [], 'slab': [], 'cluster': [], 'addition': []}

for i, xyz in enumerate(complete_xyz):
    xyz[1] = re.sub('bulk', crystal_system[i], xyz[1])
    xyz[1] = re.sub('forces:R:3', 'forces:R:3:force_mask:L:1', xyz[1])

    force_flag = np.zeros(len(xyz[2:]))
    for j in range(int(force_fraction * len(force_flag))):
        force_flag[j] = 1
    np.random.shuffle(force_flag)
    for j in range(2, len(xyz)):
        xyz[j] = xyz[j].strip() + '   {}\n'.format(int(force_flag[j - 2]))

    processed[crystal_system[i]].append(xyz)


with open('train.xyz', 'w') as f:
    with open('atom/parsed.xyz', 'r') as f_in:
        f.write(f_in.read())
    with open('dimer/AuAu/dimer.xyz', 'r') as f_in:
        f.write(f_in.read())
    for sys in systems:
        for i, xyz in enumerate(processed[sys]):
            if train_selected[sys][i]:
                for line in xyz:
                    f.write(line)

with open('test.xyz', 'w') as f:
    for sys in systems:
        for xyz in processed[sys]:
            for line in xyz:
                f.write(line)


if do_soap:
    with open('input', 'w') as f:
        for line in ['input_file = complete.xyz', 'n_species = 1', 'species = Pt', 'rcut = 5.0', 'buffer = 0.5',
                     'atom_sigma_r = 0.3', 'atom_sigma_t = 0.5', 'atom_sigma_r_scaling = 0.05',
                     'atom_sigma_t_scaling = 0.025', 'amplitude_scaling = 2', 'n_max = 8', 'l_max = 10',
                     'which_atom = 1', 'ase_format = .true.', 'nf = 4.', 'central_weight = 1.',
                     'scaling_mode = polynomial', 'timing = .true.', 'write_soap = .true.']:
            f.write(line + '\n')
    os.system('turbogap > turbogap.run')

    train_soaps = []
    with open('soap.dat', 'r') as f:
        for line in f:
            if len(line.split()) > 2:
                train_soaps.append([float(x) for x in line.split()])

    print(len(train_soaps), len(train_soaps[4]))

    train_soaps = np.array(train_soaps)

    print(len(train_soaps), len(train_soaps[4]))
