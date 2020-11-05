import os
import re
import glob
import gzip
import numpy as np
from monty.serialization import loadfn


def get_parsed_data(base_dir):
    tmp_data = []
    current_dir = os.getcwd()
    os.chdir(base_dir)
    for file in glob.glob('**/parsed.xyz.gz', recursive=True):
        wtmp = []
        with gzip.open(file, 'rt') as in_file:
            for wline in in_file:
                wtmp.append(wline)
        tmp_data.append(wtmp)
    os.chdir(current_dir)
    return tmp_data


np.random.seed(1410)  # fix for reproduction

force_fraction = 0.0  # percentage of forces to EXCLUDE from training
do_soap = False

with open('train.xyz', 'w') as fout:
    for a in ['Al', 'Ni']:
        with open(os.path.join('atoms', a, 'parsed.xyz')) as fin:
            fout.write(fin.read())

    for a in ['AlAl', 'AlNi', 'NiNi']:
        with open(os.path.join('dimers', a, 'dimer.xyz')) as fin:
            fout.write(fin.read())

    data = loadfn('AlNi_complete.json')
    for d in data:
        fout.write(d['xyz'])

    # with open('MD/parsed.xyz', 'r') as fin:  # MD has no stresses computed -> no virials
    #     for line in fin:
    #         fout.write(re.sub('cluster', 'moldyn', line))

    ce_data = get_parsed_data('/home/jank/work/Aalto/GAP_data/AlNi/training_data/run_dir')
    for xyz in ce_data:
        for line in xyz:
            fout.write(re.sub('CEbulk', 'random', line))

with open('validation.xyz', 'w') as f:
    for d in data:
        f.write(d['xyz'])
    # with open('MD/parsed.xyz', 'r') as f_in:  # MD has no stresses computed -> no virials
    #     for line in f_in:
    #         f.write(re.sub('cluster', 'moldyn', line))
    for xyz in ce_data:
        for line in xyz:
            f.write(re.sub('CEbulk', 'random', line))


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
