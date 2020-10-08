import os
import json
import shutil
from ase.io import read
from ase.io.lammpsdata import write_lammps_data


with open('structures.json', 'r') as f:
    structures = json.load(f)

base_dir = os.path.join(os.getcwd(), 'lammps_gap')

for i, struct in enumerate(structures):
    os.chdir(base_dir)
    if os.path.isdir(str(i)):
        shutil.rmtree(str(i))
    os.mkdir(str(i))
    os.chdir(str(i))

    dft_free_energy = None
    with open('in.xyz', 'wt') as f:
        for line in struct:
            f.write(line)
    with open('in.xyz', 'r') as f:
        for line in f:
            if 'energy' in line:
                for e in line.split():
                    if 'energy' in e:
                        if 'free' in e:
                            dft_free_energy = e.split('=')[1]
                        else:
                            dft_energy = e.split('=')[1]

    if dft_free_energy is None:
        raise ValueError('Check inputs for correct values, vasp_free_energy is undefined...')

    atoms = read(filename='in.xyz')
    write_lammps_data(fileobj='atom.pos', atoms=atoms, units='metal')

    for file in os.listdir(base_dir):
        if file.startswith('platinum.xml'):
            os.symlink(os.path.join('..', file), file)

    os.symlink('../compress.dat', 'compress.dat')
    os.symlink('../lammps.in', 'lammps.in')
    os.system('/home/jank/bin/lmp -in lammps.in > /dev/zero')
    with open('log.lammps', 'rt') as f:
        parse = False
        for line in f:
            if parse:
                lammps_energy = line.split()[1]
                parse = False
            if 'Time PotEng KinEng Temp' in line:
                parse = True

    dump = {'dft': dft_free_energy,
            'lammps': lammps_energy}
    with open('energies', 'w') as f:
        json.dump(obj=dump, fp=f)

    if i % 100 == 0:
        print('Running {}'.format(i))
