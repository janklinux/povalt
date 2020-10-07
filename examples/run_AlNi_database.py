import os
import sys
import json
import shutil
from ase.io.lammpsdata import write_lammps_data
from pymatgen.io.ase import AseAtomsAdaptor
from monty.serialization import loadfn


structures = loadfn(fn='AlNi_complete.json')
base_dir = os.path.join(os.getcwd(), 'lammps_gap')

print('Running:', end='')
sys.stdout.flush()
for i, struct in enumerate(structures):
    tmp = ''
    for chr in struct['xyz']:
        if chr == '\n':
            break
        else:
            tmp += chr
    if int(tmp) < 2:
        print('Structure {} is single-atom, skipping'.format(i))
        continue
    os.chdir(base_dir)
    if os.path.isdir(str(i)):
        shutil.rmtree(str(i))
    os.mkdir(str(i))
    os.chdir(str(i))

    atoms = AseAtomsAdaptor().get_atoms(struct['structure'])

    tmp = ''
    lines = []
    for char in struct['xyz']:
        if '\n' in char:
            lines.append(tmp)
            tmp = ''
        else:
            tmp += char

    dft_free_energy = None
    for line in lines:
        if 'energy' in line:
            for e in line.split():
                if 'energy' in e:
                    if 'free' in e:
                        dft_free_energy = e.split('=')[1]
                    else:
                        dft_energy = e.split('=')[1]

    if dft_free_energy is None:
        raise ValueError('Check inputs for correct values, vasp_free_energy is undefined...')

    write_lammps_data(fileobj='atom.pos', atoms=atoms, units='metal')

    for file in os.listdir(base_dir):
        if file.startswith('AlNi.xml'):
            os.symlink(os.path.join('..', file), file)

    os.symlink('../compress.dat', 'compress.dat')
    shutil.copy('../lammps.in', 'lammps.in')
    os.system('/home/jank/bin/lmp -in lammps.in > /dev/zero')

    lammps_energy = None
    with open('log.lammps', 'rt') as f:
        parse = False
        for line in f:
            if parse:
                lammps_energy = line.split()[1]
                parse = False
            if 'Time PotEng KinEng Temp' in line:
                parse = True

    if lammps_energy is None:
        raise ValueError('LAMMPS energy not found, check settings and run...')

    dump = {'dft': dft_free_energy,
            'lammps': lammps_energy}

    with open('energies', 'w') as f:
        json.dump(obj=dump, fp=f)

    if i % 100 == 0:
        print(' {}'.format(i), end='')
        sys.stdout.flush()

print('\n')
