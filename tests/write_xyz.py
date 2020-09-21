import io
import os
import json
import pymongo

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write


read_from_db = True
force_fraction = 0.98  # percentage of forces to EXCLUDE from training
do_soap = False

systems = ['fcc', 'bcc', 'hcp', 'sc', 'slab', 'cluster', 'addition']

train_split = dict()
split = {'fcc': 0.55,
         'bcc': 0.55,
         'hcp': 0.55,
         'sc': 0.55,
         'slab': 0.85,
         'cluster': 0.85,
         'addition': 1.0}

for sys in systems:
    train_split[sys] = split[sys]

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')

run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['platinum']
add_coll = data_db['platinum_additions']


with open('complete.xyz', 'w') as f:
    f.write('1\n')
    f.write('Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3:forces:R:3:force_mask:L:1 '
            'energy=-0.52810937 stress="-8.1156408296031e-06 3.659415525032732e-06 1.5484996896042586e-06 '
            '3.659415525032732e-06 -8.890405598908113e-08 -4.473464352776056e-06 1.5484996896042586e-06 '
            '-4.473464352776056e-06 -3.0529280832706558e-06" free_energy=-0.54289024 pbc="T T T" '
            'config_type=isolated_atom\n')
    f.write('Pt 0.0 0.0 0.0 0.0 0.0 0.0 0\n')

dimer_curve = list()
force_curve = list()
zcrds = np.round(np.arange(0.7, 8.1, 0.1, dtype=np.float16), 1)

for crd in zcrds:
    atoms = read(os.path.join('/home/jank/work/Aalto/vasp/training_data/dimer', str(crd), 'vasprun.xml'))
    dimer_curve.append(atoms.get_potential_energy(force_consistent=True))
    atoms.info['config_type'] = 'dimer'

    xyz = ''
    file = io.StringIO()
    write(filename=file, images=atoms, format='xyz')
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
#    tmp_line[1] = tmp_line[1].strip() + ' config_type=random\n'
    wtmp = ''
    for bit in tmp_line[1].split(' '):
        if 'Properties' in bit:
            bit += ':force_mask:L:1'
        wtmp += bit + ' '
    tmp_line[1] = wtmp.strip() + '\n'
    tmp_line[0] = tmp_line[0].strip() + '\n'

    force_flag = np.zeros(len(tmp_line[2:]))
    for j in range(int(force_fraction * len(force_flag))):
        force_flag[j] = 1
    np.random.shuffle(force_flag)

    for j in range(2, len(tmp_line)):
        tmp_line[j] = tmp_line[j].strip() + '     0\n'

    with open('complete.xyz', 'a') as f:
        for line in tmp_line:
            f.write(line)

    force_curve.append(atoms.get_forces()[1][2])

show_dimer = False
if show_dimer:
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', serif='Palatino')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = 'cm'
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 3
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.minor.width'] = 3
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.major.width'] = 3
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.minor.width'] = 3
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['axes.linewidth'] = 3
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
    complete_xyz = []
    crystal_system = []
    print('Starting DB read...')
    ik = 0
    for doc in data_coll.find({}, {'_id': 0, 'data.xyz': 1, 'name': 1}):
        if ik % 1000 == 0:
            print('busy on src: {:d}'.format(ik))
        ik += 1
        complete_xyz.append(doc['data']['xyz'])
        if 'Slab' in doc['name']:
            crystal_system.append('slab')
        elif 'Cluster' in doc['name']:
            crystal_system.append('cluster')
        else:
            crystal_system.append(doc['name'].split('||')[1].split(' ')[5])
    ik = 0
    for doc in add_coll.find({}, {'_id': 0, 'data.xyz': 1, 'name': 1}):
        if ik % 100 == 0:
            print('busy on add: {:d}'.format(ik))
        ik += 1
        complete_xyz.append(doc['data']['xyz'])
        crystal_system.append('addition')
    with open('structures.json', 'w') as f:
        json.dump(complete_xyz, f)
    with open('systems.json', 'w') as f:
        json.dump(crystal_system, f)
else:
    with open('structures.json', 'r') as f:
        complete_xyz = json.load(f)
    with open('systems.json', 'r') as f:
        crystal_system = json.load(f)


np.random.seed(1410)  # fix for reproduction

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
print('This will need approximately {} GB of memory during training.'.format(np.round(
    (system_count['fcc'] * train_split['fcc'] + system_count['bcc'] * train_split['bcc'] +
     system_count['sc'] * train_split['sc'] + system_count['hcp'] * train_split['hcp'] +
     system_count['slab'] * train_split['slab'] + system_count['cluster'] * train_split['cluster'] +
     system_count['addition'] * train_split['addition']) * 8 * 1000 * len(systems) * 150 / 2**30 * 1.1, 2)))
# ( num systems * include % ) * #systems * 150 * 8 bytes * 1000 / GB + 10%
# |          dim1           | * |    dim2    | * numerics

np.random.seed(1410)  # fix for reproduction

processed = {'fcc': [],
             'bcc': [],
             'hcp': [],
             'sc': [],
             'slab': [],
             'cluster': [],
             'addition': []}

force_flag = []
for i, xyz in enumerate(complete_xyz):
    tmp = ''
    tmp_line = []
    for line in xyz:
        tmp += line
        if '\n' in line:
            tmp_line.append(tmp)
            tmp = ''
    tmp_line[1] = tmp_line[1].strip() + ' config_type={}\n'.format(crystal_system[i])
    wtmp = ''
    for bit in tmp_line[1].split(' '):
        if 'Properties' in bit:
            bit += ':force_mask:L:1'
        wtmp += bit + ' '
    tmp_line[1] = wtmp.strip() + '\n'
    tmp_line[0] = tmp_line[0].strip() + '\n'

    force_flag = np.zeros(len(tmp_line[2:]))
    for j in range(int(force_fraction * len(force_flag))):
        force_flag[j] = 1
    np.random.shuffle(force_flag)

    for j in range(2, len(tmp_line)):
        tmp_line[j] = tmp_line[j].strip() + '     {}\n'.format(int(force_flag[j - 2]))

    processed[crystal_system[i]].append(tmp_line)


with open('complete.xyz', 'a') as f:
    for sys in systems:
        for i, xyz in enumerate(processed[sys]):
            if train_selected[sys][i]:
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
