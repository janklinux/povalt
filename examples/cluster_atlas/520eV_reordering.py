import os
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from fireworks import LaunchPad


run_all = True

if run_all:
    with open('../anna_clusters.json', 'r') as f:
        original = json.load(f)

    energy = {'300': [], '520': [], 'gap': []}
    base_dir = os.getcwd()
    found = [False for _ in range(len(original))]
    fw_meta_list = []
    ldir_list = []
    out_path = ['' for _ in range(len(original))]
    lpad = LaunchPad.auto_load()

    for fwid in lpad.get_fw_ids({'state': 'COMPLETED'}):
        os.chdir(base_dir)
        fw = lpad.get_fw_by_id(fwid)
        fw_meta_list.append(lpad.get_wf_by_fw_id(fw.fw_id).as_dict())
        ldir = '/'.join(lpad.get_launchdir(fw_id=fwid).split('/')[-3:])
        ldir_list.append(ldir)
        atoms = read(ldir + '/OUTCAR.gz')
        tmp = io.StringIO()
        write(filename=tmp, images=atoms, format='xyz')
        tmp.seek(0)

        list_1 = list()
        for i, line in enumerate(tmp):
            if i > 1:
                list_1.append([np.round(float(x), 3) for x in line.strip().split()[1:4]])
        list_1 = np.array(list_1)

        for ic, cluster in enumerate(original):
            list_2 = list()
            for i, line in enumerate(cluster['xyz']):
                if i > 1:
                    list_2.append([np.round(float(x), 3) for x in line.strip().split()[1:4]])
            list_2 = np.array(list_2)

            same = True
            if not len(list_1) == len(list_2):
                same = False

            for a, b in zip(list_1, list_2):
                if np.linalg.norm(a - b) > 0.01:
                    same = False

            if same:
                found[ic] = True
                if not os.path.exists(os.path.join(os.path.join('rerun', cluster['path']))):
                    os.makedirs(os.path.join(os.path.join('rerun', cluster['path'])))
                write(filename=os.path.join('rerun', cluster['path'], '520eV_rerun.xyz'),
                      images=atoms, format='xyz')
                with open('/tmp/delme.xyz', 'w') as f:
                    for line in cluster['xyz']:
                        f.write(line)
                old_atoms = read('/tmp/delme.xyz')
                energy['300'].append(old_atoms.get_potential_energy(force_consistent=True)
                                     / len(old_atoms.get_chemical_symbols()))
                energy['520'].append(atoms.get_potential_energy(force_consistent=True)
                                     / len(atoms.get_chemical_symbols()))

                os.chdir('/tmp')
                os.symlink('/home/jank/work/Aalto/GAP_data/Pt/80-20_seed1410_2b-soap/lammps_gap/compress.dat',
                           'compress.dat')
                os.symlink('/home/jank/work/Aalto/GAP_data/Pt/80-20_seed1410_2b-soap/lammps_gap/platinum.xml',
                           'platinum.xml')
                os.symlink('/home/jank/work/Aalto/GAP_data/Pt/80-20_seed1410_2b-soap/lammps_gap/platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5051',
                           'platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5051')
                os.symlink('/home/jank/work/Aalto/GAP_data/Pt/80-20_seed1410_2b-soap/lammps_gap/platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5052',
                           'platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5052')
                print('up: ', len(atoms.get_chemical_symbols()), len(old_atoms.get_chemical_symbols()))
                os.system('nice -n 10 quip atoms_filename=delme.xyz param_filename=platinum.xml e > quip.result')
                os.unlink('delme.xyz')
                with open('quip.result', 'r') as f:
                    for line in f:
                        if 'Energy' in line:
                            energy['gap'].append(float(line.split('=')[1]))
                os.unlink('/tmp/compress.dat')
                os.unlink('/tmp/platinum.xml')
                os.unlink('/tmp/platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5051')
                os.unlink('/tmp/platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5052')

    for i, f in enumerate(found):
        if not f:
            atoms = None
            old_atoms = None
            os.chdir(base_dir)
            if not os.path.exists(os.path.join(os.path.join('rerun', original[i]['path']))):
                os.makedirs(os.path.join(os.path.join('rerun', original[i]['path'])))
            atoms = read(os.path.join(ldir_list[i], 'OUTCAR.gz'))
            print('inbet: ', ldir_list[i], len(atoms.get_chemical_symbols()))
            write(filename=os.path.join('rerun', original[i]['path'], '520eV_rerun.xyz'),
                  images=atoms, format='xyz')
            energy['520'].append(atoms.get_potential_energy(force_consistent=True)
                                 / len(atoms.get_chemical_symbols()))
            os.chdir('/tmp')
            with open('delme.xyz', 'w') as fl:
                for line in original[i]['xyz']:
                    fl.write(line)
            old_atoms = read('delme.xyz')
            print('inold: ', os.getcwd(), len(old_atoms.get_chemical_symbols()))
            energy['300'].append(old_atoms.get_potential_energy(force_consistent=True)
                                 / len(old_atoms.get_chemical_symbols()))
            os.symlink('/home/jank/work/Aalto/GAP_data/Pt/80-20_seed1410_2b-soap/lammps_gap/compress.dat',
                       'compress.dat')
            os.symlink('/home/jank/work/Aalto/GAP_data/Pt/80-20_seed1410_2b-soap/lammps_gap/platinum.xml',
                       'platinum.xml')
            os.symlink(
                '/home/jank/work/Aalto/GAP_data/Pt/80-20_seed1410_2b-soap/lammps_gap/platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5051',
                'platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5051')
            os.symlink(
                '/home/jank/work/Aalto/GAP_data/Pt/80-20_seed1410_2b-soap/lammps_gap/platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5052',
                'platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5052')
            print('dn: ', len(atoms.get_chemical_symbols()), len(old_atoms.get_chemical_symbols()))
            os.system('nice -n 10 quip atoms_filename=delme.xyz param_filename=platinum.xml e > quip.result')
            os.unlink('delme.xyz')
            with open('quip.result', 'r') as fl:
                for line in fl:
                    if 'Energy' in line:
                        energy['gap'].append(float(line.split('=')[1]) / len(old_atoms.get_chemical_symbols()))
            os.unlink('/tmp/compress.dat')
            os.unlink('/tmp/platinum.xml')
            os.unlink('/tmp/platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5051')
            os.unlink('/tmp/platinum.xml.sparseX.GAP_2020_10_5_180_14_58_38_5052')

    with open('plot_data.json', 'w') as fo:
        json.dump(obj=energy, fp=fo)
else:
    with open('plot_data.json', 'r') as f:
        energy = json.load(f)


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

plt.plot(energy['300'], energy['gap'], '.', s=1, linewidth=1, color='g', label='300 eV')
plt.plot(energy['520'], energy['gap'], '.', s=1, linewidth=1, color='m', label='520 eV')

plt.xlabel(r'DFT Energy [eV/Atom]', fontsize=16, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')

plt.legend(loc='upper right', fontsize=8)

# plt.xlim(-7, 1)
# plt.ylim(-7, 1)

plt.tight_layout()
plt.savefig('compare_cutoffs.png', dpi=300)
plt.close()
