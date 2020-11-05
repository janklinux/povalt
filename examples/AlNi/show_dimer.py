import matplotlib.pyplot as plt


def parse_xyz(filename):
    with open(filename, 'r') as ff:
        atom_dist = []
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
                bits = line.split()
                for ii, bit in enumerate(bits):
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
                atom_dist.append(float(line.split()[3]))
                atoms = []
    return dict({'data': data, 'energy': reference_energies, 'dist': atom_dist})


result = parse_xyz('dimer.xyz')

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', serif='Palatino')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'
plt.plot(result['dist'], result['energy'], '.-', color='navy')
plt.xlabel(r'Radial Distance [\AA]', fontsize=22, color='k')
plt.ylabel(r'Free Energy [eV]', fontsize=22, color='k')
plt.title(r'Dimer', fontsize=16)
plt.show()
