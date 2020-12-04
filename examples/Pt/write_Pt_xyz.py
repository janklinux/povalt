import io
import os
import json
import pymongo
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from pymatgen import Structure


def check_vacuum_direction(input_data):
    structure = Structure.from_dict(input_data)
    a = structure.lattice.matrix[0] / 2
    b = structure.lattice.matrix[1] / 2
    for c in structure.cart_coords:
        if np.linalg.norm(c - np.array([a + b])) < 6:
            return False
    return True


np.random.seed(1410)  # fix for reproduction

read_from_db = True
force_fraction = 0  # percentage of forces to EXCLUDE from training
show_dimer = False
do_soap = False

systems = ['fcc', 'bcc', 'hcp', 'sc', 'slab', 'cluster', 'addition', 'phonons']

train_split = {'fcc': 0.5,
               'bcc': 0.5,
               'hcp': 0.5,
               'sc': 0.5,
               'slab': 0.9,
               'cluster': 0.9,
               'phonons': 0.9,
               'addition': 1.0}

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
    atoms = read(os.path.join('/home/jank/work/Aalto/GAP_data/Pt/training_data/dimer', str(crd), 'vasprun.xml'))
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
            bit += ':force_mask:L:1 '
            bit += 'virial="{} {} {} {} {} {} {} {} {}"'.format(
                virial[0][0], virial[0][1], virial[0][2],
                virial[1][0], virial[1][1], virial[1][2],
                virial[2][0], virial[1][2], virial[2][2])
        wtmp += bit + ' '
    tmp_line[1] = wtmp.strip() + '\n'
    tmp_line[0] = tmp_line[0].strip() + '\n'

    force_flag = np.zeros(len(tmp_line[2:]))
    for j in range(int(force_fraction * len(force_flag))):
        force_flag[j] = 1
    np.random.shuffle(force_flag)

    for j in range(2, len(tmp_line)):
        tmp_line[j] = tmp_line[j].strip() + '     0\n'

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
    for doc in data_coll.find({}, {'_id': 0, 'data.xyz': 1, 'name': 1, 'data.final_structure': 1}):
        if ik % 1000 == 0:
            print('busy on src: {:d}'.format(ik))
        ik += 1

        valid = True
        if 'Slab' in doc['name']:
            total_slabs += 1
            valid = check_vacuum_direction(doc['data']['final_structure'])

        if valid:
            complete_xyz.append(doc['data']['xyz'])
            if 'Slab' in doc['name']:
                added_slabs += 1
                crystal_system.append('slab')
            elif 'Cluster' in doc['name']:
                crystal_system.append('cluster')
            elif 'phonons' in doc['name']:
                crystal_system.append('phonons')
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
    print('Slabs, total: {} -- added: {}'.format(total_slabs, added_slabs))
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
print('Including in training DB: fcc     : {:5d} [{:3.1f}%]\n'
      '                          bcc     : {:5d} [{:3.1f}%]\n'
      '                          sc      : {:5d} [{:3.1f}%]\n'
      '                          hcp     : {:5d} [{:3.1f}%]\n'
      '                          slab    : {:5d} [{:3.1f}%]\n'
      '                          cluster : {:5d} [{:3.1f}%]\n'
      '                          phonons : {:5d} [{:3.1f}%]\n'
      '                          addition: {:5d} [{:3.1f}%]'.
      format(int(system_count['fcc'] * train_split['fcc']), train_split['fcc']*100,
             int(system_count['bcc'] * train_split['bcc']), train_split['bcc']*100,
             int(system_count['sc'] * train_split['sc']), train_split['sc']*100,
             int(system_count['hcp'] * train_split['hcp']), train_split['hcp']*100,
             int(system_count['slab'] * train_split['slab']), train_split['slab']*100,
             int(system_count['cluster'] * train_split['cluster']), train_split['cluster']*100,
             int(system_count['phonons'] * train_split['phonons']), train_split['phonons']*100,
             int(system_count['addition'] * train_split['addition']), train_split['addition']*100))
# print('This will need approximately {} GB of memory during training.'.format(np.round(
#     (system_count['fcc'] * train_split['fcc'] + system_count['bcc'] * train_split['bcc'] +
#      system_count['sc'] * train_split['sc'] + system_count['hcp'] * train_split['hcp'] +
#      system_count['slab'] * train_split['slab'] + system_count['cluster'] * train_split['cluster'] +
#      system_count['addition'] * train_split['addition']) * 8 * 1000 * len(systems) * 150 / 2**30 * 1.1, 2)))
# ( num systems * include % ) * #systems * 150 * 8 bytes * 1000 / GB + 10%
# |          dim1           | * |    dim2    | * numerics

processed = {'fcc': [],
             'bcc': [],
             'hcp': [],
             'sc': [],
             'slab': [],
             'cluster': [],
             'phonons': [],
             'addition': []}

skips = 0
force_flag = []

for i, xyz in enumerate(complete_xyz):
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
    for ii, el in enumerate(tmp_line[1].split()):
        if s_append:
            if '"' in el:
                el = el[:-1]
            stress.append(float(el))
            if ii > s_pos + 7:
                s_append = False
        if 'stress' in el:
            s_pos = ii
            stress.append(float(el.split('"')[1]))
            s_append = True
        if l_append:
            if '"' in el:
                el = el[:-1]
            lattice.append(float(el))
            if ii > l_pos + 7:
                l_append = False
        if 'Lattice' in el:
            l_pos = ii
            lattice.append(float(el.split('"')[1]))
            l_append = True

    if len(stress) == 0:
        stress = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    stress = np.array([stress[0:3], stress[3:6], stress[6:9]])
    lattice = np.array([lattice[0:3], lattice[3:6], lattice[6:9]])
    vol = np.abs(np.dot(lattice[2], np.cross(lattice[0], lattice[1])))
    virial = -np.dot(vol, stress)

    tmp_line[1] = tmp_line[1].strip() + ' config_type={}\n'.format(crystal_system[i])
    wtmp = ''
    for bit in tmp_line[1].split(' '):
        if 'Properties' in bit:
            bit += ':force_mask:L:1 '
            bit += 'virial="{} {} {} {} {} {} {} {} {}"'.format(
                virial[0][0], virial[0][1], virial[0][2],
                virial[1][0], virial[1][1], virial[1][2],
                virial[2][0], virial[1][2], virial[2][2])
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


with open('train.xyz', 'a') as f:
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
