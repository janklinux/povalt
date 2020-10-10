import os
import json
import shutil
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.lammpsdata import write_lammps_data


with open('all_clusters.json', 'r') as f:
    all_clusters = json.load(f)

base_dir = os.path.join(os.getcwd(), 'run')

for c in all_clusters:
    os.chdir(base_dir)
    if not os.path.isdir(c):
        os.mkdir(c)
    os.chdir(c)
    sub_dir = os.getcwd()
    for i, s in enumerate(all_clusters[c]['structures']):
        os.chdir(sub_dir)
        if os.path.isdir(str(i)):
            shutil.rmtree(str(i))
        os.mkdir(str(i))
        os.chdir(str(i))

        atoms = AseAtomsAdaptor().get_atoms(Structure.from_dict(s))
        write_lammps_data(fileobj='atom.pos', atoms=atoms, units='metal')

        for file in os.listdir(base_dir):
            if file.startswith('platinum.xml'):
                os.symlink(os.path.join('..', file), file)

        os.symlink('../compress.dat', 'compress.dat')
        os.symlink('../lammps.in', 'lammps.in')
        os.system('lmp_mpi -in lammps.in > /dev/zero')
