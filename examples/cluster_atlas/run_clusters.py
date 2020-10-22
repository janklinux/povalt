import io
import os
import json
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from ase.io import write
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from fireworks import LaunchPad
from povalt.firetasks.wf_generators import run_lammps


run_quip = False
make_wflows = False

def parse_quip(filename):
    start_parse = False
    with open(filename, 'r') as ff:
        intern_data = []
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
                    intern_data.append(tmp)
                start_parse = True
                tmp = list()
                predicted_energies.append(float(line.split('=')[1]))
            if start_parse:
                if line.startswith('AT'):
                    tmp.append(line[3:])
    dt = datetime.strptime(end_time, '%H:%M:%S') - datetime.strptime(start_time, '%H:%M:%S')
    return intern_data, predicted_energies, dt


with open('all_clusters.json', 'r') as f:
    all_clusters = json.load(f)

root_dir = os.getcwd()

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
    os.system('nice -n 10 quip atoms_filename=clusters.xyz param_filename=platinum.xml e > quip.result')
    os.chdir(root_dir)


xyz, energies, runtime = parse_quip('quip.result')


if make_wflows:
    lpad = LaunchPad().auto_load()
    lammps_params = {
        'lammps_settings': [
            'units metal', 'newton on', 'dimension   3', 'boundary    p p p', 'atom_style  atomic',
            'atom_modify map array', 'read_data   atom.pos', 'timestep    1',
            'thermo_style custom time pe ke temp', 'mass 1 195.084',
            'pair_style      quip',
            'pair_coeff * *  POT_FW_NAME "Potential xml_label=POT_FW_LABEL" 78',
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
            lmp_fw = run_lammps(lammps_params=lammps_params, structure=Structure.from_dict(s),
                                db_file='db.json', al_file=None)
            lpad.add_wf(lmp_fw)

    quit()


lmp_time = []
lmp_energy = []
lmp_n_atoms = []

for file in glob.glob('lammps/**/parsed.json', recursive=True):
    print(file)
    data = json.load(file)
    lmp_time.append(data['time'])
    lmp_energy.append(data['lammps'])
    lmp_n_atoms.append(data['n_atoms'])
    quit()


num_atoms = []
plot_energy = []
for i, (d, e) in enumerate(zip(xyz, energies)):
    num_atoms.append(int(d[0]))
    plot_energy.append(float(e))

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

plt.scatter(num_atoms, plot_energy, marker='.', color='g', label=None, s=0.5)
plt.scatter(lmp_n_atoms, lmp_energy, marker='.', color='m', label=None, s=0.5)
for i, (x, y, t) in enumerate(zip(lmp_n_atoms, lmp_energy, lmp_time)):
    plt.text(x+2, y, r'{}s'.format(round(t, 3)), fontsize=6)

plt.xlabel(r'Number of atoms in cluster', fontsize=16, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')

plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('clusters.png', dpi=600)
plt.close()
