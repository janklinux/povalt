import io
import os
import json
import glob
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from ase.io import write
from ase.io.lammpsrun import read_lammps_dump_text
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from fireworks import LaunchPad
from povalt.firetasks.wf_generators import run_lammps


run_quip = False
make_wflows = False


def parse_quip(filename):
    start_parse = False
    with open(filename, 'r') as fff:
        intern_data = []
        predicted_energies = []
        first_line_parsed = False
        for fline in fff:
            if not first_line_parsed:
                start_time = fline.split()[3]
                first_line_parsed = True
            if fline.startswith('libAtoms::Finalise:'):
                if len(fline.split()) == 3:
                    end_time = fline.split()[2]
            if fline.startswith('Energy='):
                if start_parse:
                    intern_data.append(atmp)
                start_parse = True
                atmp = list()
                predicted_energies.append(float(fline.split('=')[1]))
            if start_parse:
                if fline.startswith('AT'):
                    atmp.append(fline[3:])
    dt = datetime.strptime(end_time, '%H:%M:%S') - datetime.strptime(start_time, '%H:%M:%S')
    return intern_data, predicted_energies, dt


root_dir = os.getcwd()

with open('all_clusters.json', 'r') as f:
    all_clusters = json.load(f)


if run_quip:
    base_dir = os.path.join(os.getcwd(), 'quip')
    os.chdir(base_dir)
    out = io.StringIO()
    for c in all_clusters:
        for s in all_clusters[c]['structures']:
            atoms = AseAtomsAdaptor().get_atoms(Structure.from_dict(s))
            if len(atoms.get_chemical_symbols()) > 1:
                write(filename=out, images=atoms, append=True, format='xyz')

    out.seek(0)
    with open('clusters.xyz', 'w') as f:
        for out_line in out:
            f.write(out_line)
    os.system('sed -i s/"Pt"/"Au"/g clusters.xyz')
    os.system('nice -n 10 quip atoms_filename=clusters.xyz param_filename=aurum.xml e > quip.result')


if make_wflows:
    os.chdir(root_dir)
    lpad = LaunchPad().auto_load()
    lammps_params = {
        'lammps_settings': [
            'units metal', 'newton on', 'dimension   3', 'boundary    p p p', 'atom_style  atomic',
            'atom_modify map array', 'read_data   atom.pos', 'timestep    1',
            'thermo_style custom time pe ke temp', 'mass 1 196.967',
            'pair_style      quip',
            'pair_coeff * *  POT_FW_NAME "Potential xml_label=POT_FW_LABEL" 79',
            'velocity all zero linear', 'thermo 1',
            'neigh_modify every 1 delay 0 check yes',
            'min_style          cg', 'minimize           1e-10 1e-12 10000 100000',
            'write_dump all custom final.dump id type x y z fx fy fz'],
        'atoms_filename': 'atom.pos',  # filename must match the name in settings above
        # 'structure': '',  # placeholder for pymatgen structure object
        'units': 'metal',  # must match settings
        'lmp_bin': 'lmp',
        'lmp_params': '',  # '-k on t 4 g 1 -sf kk',
        'mpi_cmd': None,
        'mpi_procs': 1,
        'omp_threads': 1,
    }

    structure_list = []
    for c in all_clusters:
        for s in all_clusters[c]['structures']:
            tmp = Structure.from_dict(s)
            au_struct = Structure(lattice=tmp.lattice, species=['Au' for _ in range(len(tmp.sites))],
                                  coords=tmp.cart_coords, coords_are_cartesian=True)
            lmp_fw = run_lammps(lammps_params=lammps_params, structure=au_struct,
                                db_file='db.json', al_file=None)
            lpad.add_wf(lmp_fw)

    quit()


os.chdir(root_dir)

xyz, energies, runtime = parse_quip('quip/quip.result')
num_atoms = []
guess_energy = []
for i, (d, e) in enumerate(zip(xyz, energies)):
    num_atoms.append(int(d[0]))
    guess_energy.append(float(e) / float(d[0]))


gold_dist = []
guess_dist = []
platinum_dist = []

gold_order = dict()
guess_order = dict()
platinum_order = dict()


for na, ge in zip(num_atoms, guess_energy):
    if na not in guess_order:
        guess_order[na] = []
    guess_order[na].append(ge)

# print(len(guess_order))
# for i in guess_order:
#     if len(guess_order[i]) > 130:
#         print(i)

srt = sorted(guess_order[85])
for en in srt:
    guess_dist.append(en - srt[0])


pt_time = []
pt_energy = []
pt_n_atoms = []

for file in glob.glob('/home/jank/work/Aalto/GAP_data/Pt/cluster_atlas/lammps/**/parsed.json', recursive=True):
    with open(os.path.join('/'.join(file.split('/')[:-1]), 'final.dump')) as f:
        read_atoms = True
        check_atoms = False
        for line in f:
            if check_atoms:
                if int(line) == 1:
                    read_atoms = False
                    check_atoms = False
                    continue
                else:
                    check_atoms = False
            if 'ITEM: NUMBER OF ATOMS' in line:
                check_atoms = True
        f.seek(0)
        if read_atoms:
            atoms = read_lammps_dump_text(fileobj=f)
            with open(file, 'r') as ff:
                data = json.load(ff)
            pt_time.append(data['time'])
            pt_energy.append(data['lammps'] / data['n_atoms'])
            pt_n_atoms.append(data['n_atoms'])
            if data['n_atoms'] not in platinum_order:
                platinum_order[data['n_atoms']] = {'energy': [], 'structure': []}
            platinum_order[data['n_atoms']]['energy'].append(float(data['lammps'] / data['n_atoms']))
            platinum_order[data['n_atoms']]['structure'].append(atoms)


au_time = []
au_energy = []
au_n_atoms = []

for file in glob.glob('/home/jank/work/Aalto/GAP_data/Au/cluster_atlas/lammps/**/parsed.json', recursive=True):
    with open(os.path.join('/'.join(file.split('/')[:-1]), 'final.dump')) as f:
        read_atoms = True
        check_atoms = False
        for line in f:
            if check_atoms:
                if int(line) == 1:
                    read_atoms = False
                    check_atoms = False
                    continue
                else:
                    check_atoms = False
            if 'ITEM: NUMBER OF ATOMS' in line:
                check_atoms = True
        f.seek(0)
        if read_atoms:
            atoms = read_lammps_dump_text(fileobj=f)
            with open(file, 'r') as ff:
                data = json.load(ff)
            au_time.append(data['time'])
            au_energy.append(data['lammps'] / data['n_atoms'])
            au_n_atoms.append(data['n_atoms'])
            if data['n_atoms'] not in gold_order:
                gold_order[data['n_atoms']] = {'energy': [], 'structure': []}
            gold_order[data['n_atoms']]['energy'].append(float(data['lammps'] / data['n_atoms']))
            gold_order[data['n_atoms']]['structure'].append(atoms)


for n_atoms in platinum_order:
    pt_en, pt_st = (list(ls) for ls in zip(*sorted(zip(platinum_order[n_atoms]['energy'],
                                                       platinum_order[n_atoms]['structure']))))

    if len(platinum_order[n_atoms]['structure']) > 120:
        for en in platinum_order[n_atoms]['energy']:
            # print('dE: ', en - gold_order[n_atoms]['energy'][0])
            platinum_dist.append(en - platinum_order[n_atoms]['energy'][0])


base_dir = os.path.join(root_dir, 'energy_ordering')
for n_atoms in gold_order:
    os.chdir(base_dir)
    au_en, au_st = (list(ls) for ls in zip(*sorted(zip(gold_order[n_atoms]['energy'],
                                                       gold_order[n_atoms]['structure']))))

    if len(gold_order[n_atoms]['structure']) > 120:
        for en in gold_order[n_atoms]['energy']:
            # print('dE: ', en - gold_order[n_atoms]['energy'][0])
            gold_dist.append(en - gold_order[n_atoms]['energy'][0])

    # print('Clusters with {} atoms: {}'.format(n_atoms, len(gold_order[n_atoms]['structure'])))

    if os.path.isdir(str(n_atoms)):
        shutil.rmtree(str(n_atoms))
    os.mkdir(str(n_atoms))
    os.chdir(str(n_atoms))

    for si, st in enumerate(au_st):
        if os.path.isdir(str(si)):
            shutil.rmtree(str(si))
        os.mkdir(str(si))
        os.chdir(str(si))

        with open('energy', 'w') as f:
            f.write(str(au_en[si]))
        hyd_str = AseAtomsAdaptor().get_structure(au_st[si])
        new_crds = []
        for c in hyd_str.frac_coords:
            new_crds.append([float(x + 0.5) for x in c])
        Structure(lattice=hyd_str.lattice, species=['Au' for _ in range(len(hyd_str.species))],
                  coords=new_crds, coords_are_cartesian=False).to(filename='cluster.vasp', fmt='POSCAR')
        os.chdir('..')
        if si > 1:
            break


os.chdir(root_dir)

# plt.rc('text', usetex=True)
# plt.rc('font', family='sans-serif', serif='Palatino')
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.sans-serif'] = 'cm'
#
# plt.scatter(num_atoms, guess_energy, marker='.', color='g', label=r'Start points', s=0.5)
# plt.scatter(au_n_atoms, au_energy, marker='.', color='m', label=r'Relaxed', s=0.5)
#
# plt.title(r'Gold clusters')
# plt.xlabel(r'Number of atoms in cluster', fontsize=16, color='k')
# plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')
#
# plt.legend(loc='upper right')
#
# plt.tight_layout()
# plt.savefig('gold_clusters.png', dpi=400)
# plt.close()




for n_atoms in platinum_order:
    pt_en, pt_st = (list(ls) for ls in zip(*sorted(zip(platinum_order[n_atoms]['energy'],
                                                       platinum_order[n_atoms]['structure']))))

    if len(platinum_order[n_atoms]['structure']) > 120:
        for en in platinum_order[n_atoms]['energy']:
            # print('dE: ', en - gold_order[n_atoms]['energy'][0])
            platinum_dist.append(en - platinum_order[n_atoms]['energy'][0])


for n_atoms in gold_order:
    os.chdir(base_dir)
    au_en, au_st = (list(ls) for ls in zip(*sorted(zip(gold_order[n_atoms]['energy'],
                                                       gold_order[n_atoms]['structure']))))

    if len(gold_order[n_atoms]['structure']) > 120:
        for en in gold_order[n_atoms]['energy']:
            # print('dE: ', en - gold_order[n_atoms]['energy'][0])
            gold_dist.append(en - gold_order[n_atoms]['energy'][0])




plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', serif='Palatino')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'

plt.scatter(num_atoms, guess_energy, marker='.', color='g', label=r'Start points', s=0.5)
plt.scatter(pt_n_atoms, pt_energy, marker='.', color='m', label=r'Relaxed', s=0.5)

plt.title(r'Platinum clusters')
plt.xlabel(r'Number of atoms in cluster', fontsize=16, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')

plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('platinum_clusters.png', dpi=400)
plt.close()


plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', serif='Palatino')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'

half_list = [float(x/2) for x in pt_energy]
plt.scatter(au_n_atoms, au_energy, marker='.', color='orange', label=r'Gold', s=0.5)
plt.scatter(pt_n_atoms, half_list, marker='.', color='m', label=r'Platinum', s=0.5)

plt.title(r'Gold vs Platinum clusters')
plt.xlabel(r'Number of atoms in cluster', fontsize=16, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')

plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('cluster_compare.png', dpi=400)
plt.close()


plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', serif='Palatino')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'

plt.hist(au_n_atoms, bins=len(au_n_atoms))

plt.title(r'Clusters in database')
plt.xlabel(r'Number of atoms in cluster', fontsize=16, color='k')
plt.ylabel(r'Number of clusters in bin', fontsize=16, color='k')

plt.tight_layout()
plt.savefig('binned_clusters.png', dpi=400)
plt.close()


plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', serif='Palatino')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'

fig, ax = plt.subplots(3)
ax[0].plot(range(len(guess_dist)), guess_dist, '-', color='g', lw=1)
ax[0].text(50, 0.3, r'Initial Guess', color='g', fontsize=10)
# ax[0].set_ylabel(r'dE', fontsize=12, color='k')

ax[1].plot(range(len(gold_dist)), gold_dist, '-', color='orange', lw=1)
ax[1].text(45, 0.05, r'Gold', color='orange', fontsize=10)
ax[1].set_ylabel(r'dE [eV/atom]', fontsize=12, color='k')

ax[2].plot(range(len(platinum_dist)), platinum_dist, '-', color='m', lw=1)
ax[2].text(45, 0.2, r'Platinum', color='m', fontsize=10)
# ax[2].set_ylabel(r'dE', fontsize=12, color='k')

fig.suptitle(r'Distribution before and after relaxation (85 atom cluster)')
plt.xlabel(r'Number cluster', fontsize=16, color='k')

plt.tight_layout()
plt.savefig('relaxed_clusters.png', dpi=400)
plt.close()
