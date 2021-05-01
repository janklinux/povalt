import re
import os
import lzma
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pymongo import MongoClient
from datetime import datetime


def get_command_line(filename):
    with open(filename, 'r') as f:
        complete_line = ''
        for line in f:
            if 'command_line>' in line:
                complete_line += line.strip() + ' '

    return_line = ''
    for ib, char in enumerate(re.sub('_', '-', complete_line[24:-19])):
        return_line += char
        if ib % 115 == 0 and ib != 0:
            return_line += '\n'
    return_line = re.sub('{', '', return_line)
    return_line = re.sub('}', '', return_line)
    return return_line


def parse_quip(filename):
    start_parse = False
    with open(filename, 'r') as ff:
        data = []
        predicted_energies = []
        first_line_parsed = False
        for line in ff:
            if not first_line_parsed:
                start_time = line.split()[3]
                first_line_parsed = True
            if line.startswith('libAtoms::Finalise:'):
                if len(line.split()) == 3:
                    end_time = line.split()[2]
            if line.startswith('Energy='):
                if start_parse:
                    data.append(tmp)
                start_parse = True
                tmp = list()
                predicted_energies.append(float(line.split('=')[1]))
            if start_parse:
                if line.startswith('AT'):
                    tmp.append(line[3:])
    dt = datetime.strptime(end_time, '%H:%M:%S') - datetime.strptime(start_time, '%H:%M:%S')
    return dict({'data': data, 'predicted_energies': predicted_energies}), dt


def parse_xyz(filename):
    with open(filename, 'r') as ff:
        data = []
        reference_energies = []
        atoms = []
        line_count = 0
        for line in ff:
            line_count += 1
            if line_count == 1:
                n_atoms = int(line)
            if line_count == 2:
                info = line
                lbits = line.split()
                for ii, bit in enumerate(lbits):
                    if 'free_energy' in bit:
                        reference_energies.append(float(bit.split('=')[1]))
            if 2 < line_count <= n_atoms + 2:
                atoms.append(line)
            if line_count == n_atoms + 2:
                line_count = 0
                tmp = list()
                tmp.append(n_atoms)
                tmp.append(info)
                for a in atoms:
                    tmp.append(a)
                data.append(tmp)
                atoms = []
    return dict({'data': data, 'reference_energies': reference_energies})


def get_differences(in_result, in_reference):
    result_data = []
    reference_data = []

    for ida, res in enumerate(in_result['data']):
        coord = []
        forces = []
        n_atoms = 0
        line_count = 0
        for line in res:
            line_count += 1
            if line_count == 1:
                n_atoms = int(line)
            if 2 < line_count <= n_atoms + 2:
                coord.append([float(x) for x in line.split()[1:4]])
                forces.append([float(x) for x in line.split()[13:17]])
            if line_count == n_atoms + 2:
                line_count = 0
                tmp = dict()
                tmp['free_energy'] = in_result['predicted_energies'][ida]
                tmp['coords'] = coord
                tmp['forces'] = forces
                result_data.append(tmp)
                coord = []
                forces = []

    for ref in in_reference['data']:
        energy = 0
        coord = []
        forces = []
        n_atoms = 0
        line_count = 0
        for line in ref:
            line_count += 1
            if line_count == 1:
                n_atoms = int(line)
            if line_count == 2:
                lbits = line.split()
                for ii, bit in enumerate(lbits):
                    if 'free_energy' in bit:
                        energy = float(bit.split('=')[1])
            if 2 < line_count <= n_atoms + 2:
                coord.append([float(x) for x in line.split()[1:4]])
                forces.append([float(x) for x in line.split()[4:7]])
            if line_count == n_atoms + 2:
                line_count = 0
                tmp = dict()
                tmp['free_energy'] = energy
                tmp['coords'] = coord
                tmp['forces'] = forces
                reference_data.append(tmp)
                energy = 0
                coord = []
                forces = []

    energy_diff = []
    force_diff = []
    reference_per_atom = []
    prediction_per_atom = []

    print('parsed sizes: data: {}, ref: {}'.format(len(result_data), len(reference_data)))

    for i in range(np.min([len(result_data), len(reference_data)])):
        for fa, fb in zip(np.array(result_data[i]['coords']), np.array(reference_data[i]['coords'])):
            if not np.all(fa == fb):
                print('not the same: ', fa, fb)
                quit()

        reference_per_atom.append(reference_data[i]['free_energy'] / len(result_data[i]['coords']))
        prediction_per_atom.append(result_data[i]['free_energy'] / len(result_data[i]['coords']))

        energy_diff.append(np.abs(np.abs(reference_data[i]['free_energy'] / len(result_data[i]['coords'])) -
                                  np.abs(result_data[i]['free_energy'] / len(result_data[i]['coords']))))

        df = 0
        for fa, fb in zip(np.array(result_data[i]['forces']), np.array(reference_data[i]['forces'])):
            df += np.linalg.norm(fa - fb) / 3
        force_diff.append(df/len(result_data[i]['forces']))

    return reference_per_atom, prediction_per_atom, energy_diff, force_diff


def scatterplot(result_energy, reference_energy, quip_time, max_energy_error,
                avg_energy_error, force_error, gap_name, gap_file):
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
    plt.scatter(result_energy, reference_energy, marker='.', color='navy', label=None, s=0.5)
    plt.xlabel(r'Computed Energy [eV/atom]', fontsize=16, color='k')
    plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')
    plt.scatter(5, 5, marker='.', color='k', label=r'GAP vs DFT', facecolor='w', s=25)
    plt.plot([-6.5, 0.5], [-6.5, 0.5], '-', color='k', linewidth=0.25)
    plt.text(-3.0, -0.6, r'Max error: {} eV/atom'.format(round(max_energy_error, 3)), fontsize=8)
    plt.text(-3.0, -0.9, r'Mean error: {} meV/atom'.format(round(avg_energy_error*1000, 1)), fontsize=8)
    plt.text(-3.0, -1.3, r'Mean force error: {} eV/\AA'.format(round(force_error, 3)), fontsize=8)
    plt.text(-1.0, -2.4, r'QUIP runtime: {}'.format(quip_time), fontsize=8)
    plt.text(-1.0, -3.1, get_command_line(gap_file), fontsize=4)
    plt.legend(loc='upper left')
    plt.xlim(-3.75, 1)
    plt.ylim(-3.75, 1)
    plt.tight_layout()
    plt.savefig('../GAP_vs_DFT-' + gap_name + '.png', dpi=600)
    plt.close()


mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
conn = MongoClient(host='numphys.org', port=27017, ssl=True, tlsCAFile=ca_file, ssl_certfile=cl_file)
data_db = conn.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['validate_potentials']

reference = parse_xyz('test.xyz')

num_pots = data_coll.estimated_document_count()
offset = int(num_pots / mpi_size)

base_dir = os.getcwd()
all_names = []
all_labels = []
if mpi_rank == 0:
    print('DB Content: {} potentials'.format(num_pots))
    xml_name = None
    xml_label = None
    for pot in data_coll.find():
        os.chdir(base_dir)
        for p in pot:
            if p != '_id':
                bits = p.split(':')
                if len(bits) == 2:
                    xml_name = bits[0] + '.' + bits[1]
                if len(bits) == 4:
                    xml_label = p.split(':')[3][:-1]

        if os.path.isdir(xml_label):
            shutil.rmtree(xml_label)
        os.mkdir(xml_label)
        os.chdir(xml_label)

        for p in pot:
            if p != '_id':
                with open(re.sub(':', '.', p), 'wb') as fp:
                    fp.write(lzma.decompress(pot[p]))

        all_names.append(xml_name)
        all_labels.append(xml_label)

    cpu = 0
    name = [[] for _ in range(mpi_size)]
    label = [[] for _ in range(mpi_size)]
    for ip, (aname, alabel) in enumerate(zip(all_names, all_labels)):
        name[cpu].append(aname)
        label[cpu].append(alabel)
        if cpu != mpi_size - 1:
            if ip % offset == 0 and ip != 0:
                cpu += 1
else:
    name = None
    label = None

name = mpi_comm.scatter(name, root=0)
label = mpi_comm.scatter(label, root=0)

all_len = len(label)
summed_num = mpi_comm.allreduce(all_len, op=MPI.SUM)
if summed_num != num_pots:
    raise ValueError('MPI job distribution not consistent')

assert(len(name) == len(label))

for name, label in zip(name, label):
    os.chdir(os.path.join(base_dir, label))
    os.symlink('../compress.dat', 'compress.dat')
    os.symlink('../test.xyz', 'test.xyz')
    os.system('sed -i s@/users/kloppej1/scratch/jank/pot_fit/Au/compress.dat@compress.dat@g {}'.format(name))
    print('CPU {} running: quip atoms_filename=test.xyz param_filename={} for {}'.format(mpi_rank, name, label))
    os.system('nice -n 10 quip atoms_filename=test.xyz param_filename={} e f > quip.result'.format(name))

    result, runtime = parse_quip('quip.result')
    os.system('nice -n 15 xz -z9e quip.result &')
    eref, epred, de, dforce = get_differences(in_result=result, in_reference=reference)

    scatterplot(result_energy=epred, reference_energy=eref,
                quip_time=runtime, max_energy_error=np.amax(de), avg_energy_error=np.sum(de)/len(de),
                force_error=np.sum(dforce)/len(dforce), gap_name=label, gap_file=name)

MPI.Finalize()