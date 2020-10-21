import os
import json
import shutil
from ase.io.lammpsdata import write_lammps_data
from pymatgen.io.ase import AseAtomsAdaptor
from monty.serialization import loadfn


structures = loadfn(fn='CuPtIr_complete.json')
base_dir = os.path.join(os.getcwd(), 'lammps_gap')

print('Running...')

k = 0
for i, struct in enumerate(structures):
    if i % 100 == 0:
        print('   Structure #{:d5.5}, actual run #{:d5.5} of {:d5.5}'.format(i, k, len(structures)))

    tmp = ''
    lines = []
    for char in struct['xyz']:
        if '\n' in char:
            lines.append(tmp)
            tmp = ''
        else:
            tmp += char

    if int(lines[0]) < 3:  # need three atoms at least
        continue

    atoms = AseAtomsAdaptor().get_atoms(struct['structure'])

    tmp = []
    for a in atoms.get_chemical_symbols():
        if a not in tmp:
            tmp.append(a)

    if len(tmp) != 3:  # need all three different species for GAP
        continue

    os.chdir(base_dir)
    if os.path.isdir(str(i)):
        shutil.rmtree(str(i))
    os.mkdir(str(i))
    os.chdir(str(i))

    write_lammps_data(fileobj='atom.pos', atoms=atoms, units='metal')

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

    for file in os.listdir(base_dir):
        if file.startswith('CuPtIr.xml'):
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

    k += 1

print('\n')
