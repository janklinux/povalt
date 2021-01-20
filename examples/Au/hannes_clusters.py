import os
import matplotlib.pyplot as plt
from ase.io import read, write


def parse_quip(filename):
    start_parse = False
    with open(filename, 'r') as ff:
        data = []
        predicted_energies = []
        for line in ff:
            if line.startswith('Energy='):
                if start_parse:
                    data.append(tmp)
                start_parse = True
                tmp = list()
                predicted_energies.append(float(line.split('=')[1]))
            if start_parse:
                if line.startswith('AT'):
                    tmp.append(line[3:])
    return predicted_energies  # dict({'data': data, 'predicted_energies': predicted_energies})


quip_data = parse_quip('quip.result')


if os.path.isfile('all_clusters.xyz'):
    os.unlink('all_clusters.xyz')

all_atoms = []
for c in range(1, 12):
    atoms = read('OUTCAR_S' + str(c))
    all_atoms.append(atoms)
    write(filename='all_clusters.xyz', images=atoms, format='xyz', append=True)


plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', serif='Palatino')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'

rmse = []
for i, at in enumerate(all_atoms):
    plt.scatter(at.get_potential_energy(force_consistent=True) / len(at.get_chemical_symbols()),
                quip_data[i] / len(at.get_chemical_symbols()),
                marker='.', color='navy', label=None, s=0.5)
    rmse.append(abs(abs(at.get_potential_energy(force_consistent=True) / len(at.get_chemical_symbols())) -
                abs(quip_data[i] / len(at.get_chemical_symbols()))))

plt.xlabel(r'Computed Energy [eV/atom]', fontsize=16, color='k')
plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')
plt.scatter(5, 5, marker='.', color='k', label=r'GAP vs DFT', facecolor='w', s=25)
plt.plot([-200, 0.5], [-200, 0.5], '-', color='k', linewidth=0.25)
plt.text(-2.7, -2.75, r'Max error: {} meV/atom'.format(round(max(rmse)*1000, 3)), fontsize=8)
# plt.text(-3.0, -0.9, r'Mean error: {} meV/atom'.format(round(avg_energy_error*1000, 1)), fontsize=8)
# plt.text(-3.0, -1.3, r'Mean force error: {} eV/\AA'.format(round(force_error, 3)), fontsize=8)
# plt.text(-1.0, -2.4, r'QUIP runtime: {}'.format(quip_time), fontsize=8)
# plt.text(-1.0, -3.1, get_command_line(gap_file), fontsize=4)
plt.legend(loc='upper left')
plt.xlim(-2.8, -2.6)
plt.ylim(-2.8, -2.6)
plt.tight_layout()
plt.savefig('GAP_vs_DFT.png', dpi=300)
plt.close()
