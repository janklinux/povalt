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

    with open('/tmp/stru', 'wt') as f:
        for line in struct:
            f.write(line)
    with open('/tmp/stru', 'r') as f:
        for line in f:
            if 'energy' in line:
                for e in line.split():
                    if 'energy' in e:
                        if 'free' in e:
                            vasp_free_energy = e.split('=')[1]
                        else:
                            vasp_energy = e.split('=')[1]
    with open('DFT_energy', 'wt') as f:
        f.write(vasp_free_energy)
    atoms = read(filename='/tmp/stru')
    write_lammps_data(fileobj='atom.pos', atoms=atoms, units='metal')
    # os.symlink('../Pt_Zhou04.eam.alloy', 'Pt_Zhou04.eam.alloy')
    os.symlink('../platinum.xml', 'platinum.xml')
    os.symlink('../platinum.xml.sparseX.GAP_2020_9_28_180_10_57_12_4571',
               'platinum.xml.sparseX.GAP_2020_9_28_180_10_57_12_4571')
    os.symlink('../platinum.xml.sparseX.GAP_2020_9_28_180_10_57_12_4572',
               'platinum.xml.sparseX.GAP_2020_9_28_180_10_57_12_4572')
    # os.symlink('../platinum.xml.sparseX.', 'platinum.xml.sparseX.')
    os.symlink('../compress.dat', 'compress.dat')
    os.symlink('../lammps.in', 'lammps.in')
    # os.system('/usr/bin/time /home/jank/bin/lmp -in lammps.in > run.time')
    os.system('/home/jank/bin/lmp -in lammps.in > /dev/zero')
    with open('log.lammps', 'rt') as f:
        parse = False
        for line in f:
            if parse:
                lammps_energy = line.split()[1]
                parse = False
            if 'Time PotEng KinEng Temp' in line:
                parse = True
    with open('LAMMPS_energy', 'wt') as f:
        f.write(lammps_energy)

    if i % 100 == 0:
        print('Running {}'.format(i))
