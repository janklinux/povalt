import os
import re
import numpy as np
from monty.serialization import loadfn


np.random.seed(1410)  # fix for reproduction

force_fraction = 0.0  # percentage of forces to EXCLUDE from training
do_soap = False

systems = ['bulk', 'moldyn']
train_split = {'bulk': 0.8,
               'moldyn': 0.8}

with open('train.xyz', 'w') as fout:
    for a in ['Al', 'Ni']:
        with open(os.path.join('atoms', a, 'parsed.xyz')) as fin:
            fout.write(fin.read())

    for a in ['AlAl', 'AlNi', 'NiNi']:
        with open(os.path.join('dimers', a, 'dimer.xyz')) as fin:
            fout.write(fin.read())

    data = loadfn('AlNi_complete.json')

    system_count = dict()
    for csys in systems:
        system_count[csys] = data['xyz'].count(csys)

    print(system_count)
    quit()


    train_selected = dict()
    for sys in systems:
        tmp = []
        for i in range(system_count[sys]):
            if i < system_count[sys] * train_split[sys]:
                tmp.append(True)
            else:
                tmp.append(False)
            train_selected[sys] = tmp
        np.random.shuffle(train_selected[sys])

    with open('test.xyz', 'w') as ftest:
        for sys in systems:
            for i, t in enumerate(train_selected[sys]):
                if t:
                    fout.write(data[i]['xyz'])
                else:
                    ftest.write(data[i]['xyz'])


with open('MD/parsed.xyz', 'r') as fin:
    with open('train.xyz', 'a') as fout:
        for line in fin:
            fout.write(re.sub('cluster', 'moldyn', line))


with open('no_dimers.xyz', 'w') as f:
    for d in data:
        f.write(d['xyz'])
    with open('MD/parsed.xyz', 'r') as f_in:
        for line in f_in:
            f.write(re.sub('cluster', 'moldyn', line))


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