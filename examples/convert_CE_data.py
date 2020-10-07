import os
import glob
import gzip
import numpy as np
from pymatgen import Structure
from monty.serialization import dumpfn


def parse_file(base_dir):
    scf = dict()
    i_step = 0

    parse_atoms = False
    parse_forces = False
    parse_stress = False

    with gzip.open(os.path.join(base_dir, 'run.gz'), 'rt') as f:
        for line in f:
            if '| Number of atoms' in line:
                n_atoms = int(line.split()[5])
            if 'Begin self-consistency loop:' in line:
                i_step += 1
                i_atom = 0
                i_force = 0
                scf[i_step] = {'atoms': [], 'forces': [], 'lattice': [], 'stress': [],
                               'species': [], 'energy': 0.0}
            if '| Total energy                  :' in line:
                scf[i_step]['energy'] = float(line.split()[6])
            if parse_forces:
                scf[i_step]['forces'].append([float(x) for x in line.split()[2:5]])
                i_force += 1
                if i_force == n_atoms:
                    parse_forces = False
            if 'Total atomic forces (unitary forces cleaned) [eV/Ang]:' in line:
                parse_forces = True
            if parse_atoms:
                if 'lattice' in line:
                    scf[i_step]['lattice'].append([float(x) for x in line.split()[1:4]])
                elif len(line.split()) == 0:
                    continue
                else:
                    scf[i_step]['atoms'].append([float(x) for x in line.split()[1:4]])
                    scf[i_step]['species'].append(line.split()[4])
                    i_atom += 1
                if i_atom == n_atoms:
                    parse_atoms = False
            if 'x [A]             y [A]             z [A]' in line:
                parse_atoms = True
            if parse_stress:
                if '|  x       ' in line:
                    scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                if '|  y       ' in line:
                    scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                if '|  z       ' in line:
                    scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                if '|  Pressure:' in line:
                    parse_stress = False
            if 'Analytical stress tensor - Symmetrized' in line:
                parse_stress = True

    with gzip.open(os.path.join(base_dir, 'parsed.xyz.gz'), 'wt') as f:
        for i in range(1, len(scf) + 1):
            lat = np.array([[scf[i]['lattice'][0][0], scf[i]['lattice'][0][1], scf[i]['lattice'][0][2]],
                            [scf[i]['lattice'][1][0], scf[i]['lattice'][1][1], scf[i]['lattice'][1][2]],
                            [scf[i]['lattice'][2][0], scf[i]['lattice'][2][1], scf[i]['lattice'][2][2]]])
            stress = np.array([[scf[i]['stress'][0][0], scf[i]['stress'][0][1], scf[i]['stress'][0][2]],
                               [scf[i]['stress'][1][0], scf[i]['stress'][1][1], scf[i]['stress'][1][2]],
                               [scf[i]['stress'][2][0], scf[i]['stress'][2][1], scf[i]['stress'][2][2]]])
            vol = np.dot(lat[0], np.cross(lat[1], lat[2]))
            virial = -np.dot(vol, stress)

            f.write('{}\n'.format(n_atoms))
            f.write('Lattice="{:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f}"'
                    ' Properties=species:S:1:pos:R:3:forces:R:3:force_mask:L:1'
                    ' stress="{:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f}"'
                    ' virial="{:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f}"'
                    ' free_energy={:6.6f} pbc="T T T"'
                    ' config_type=bulk\n'.format(
                        scf[i]['lattice'][0][0], scf[i]['lattice'][0][1], scf[i]['lattice'][0][2],
                        scf[i]['lattice'][1][0], scf[i]['lattice'][1][1], scf[i]['lattice'][1][2],
                        scf[i]['lattice'][2][0], scf[i]['lattice'][2][1], scf[i]['lattice'][2][2],
                        scf[i]['stress'][0][0], scf[i]['stress'][0][1], scf[i]['stress'][0][2],
                        scf[i]['stress'][1][0], scf[i]['stress'][1][1], scf[i]['stress'][1][2],
                        scf[i]['stress'][2][0], scf[i]['stress'][2][1], scf[i]['stress'][2][2],
                        virial[0][0], virial[0][1], virial[0][2],
                        virial[1][0], virial[1][1], virial[1][2],
                        virial[2][0], virial[2][1], virial[2][2],
                        scf[i]['energy']))

            for j in range(len(scf[i]['atoms'])):
                f.write('{} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} 0\n'.
                        format(scf[i]['species'][j],
                               scf[i]['atoms'][j][0], scf[i]['atoms'][j][1], scf[i]['atoms'][j][2],
                               scf[i]['forces'][j][0], scf[i]['forces'][j][1], scf[i]['forces'][j][2]))


complete_list = []
dir_list = glob.glob('block*/**/geometry.in.gz', recursive=True)

for i, d in enumerate(dir_list):
    store = dict()
    base = d[:-15]
    converged = False
    with gzip.open(os.path.join(base, 'run.gz'), 'rt') as f:
        for line in f:
            if 'Have a nice day.' in line:
                converged = True

    if not converged:
        # print('fizzled...')
        continue

    with open('/tmp/delme.in', 'w') as f:
        with gzip.open(d, 'rt') as fin:
            f.write(fin.read())

    struct = Structure.from_file('/tmp/delme.in')

    specs = []
    all_specs = []
    for s in struct.species:
        all_specs.append(str(s))
        if str(s) not in specs:
            specs.append(str(s))

    tmp = dict()
    for s in specs:
        tmp[s] = all_specs.count(s)/len(all_specs)

    parse_file(base_dir=base)
    with gzip.open(os.path.join(base, 'parsed.xyz.gz'), 'rt') as f:
        xyz = f.read()

    store['structure'] = struct.as_dict()
    store['composition'] = tmp
    store['xyz'] = xyz

    complete_list.append(store)

print('Parsed {} structure relaxations'.format(len(complete_list)))
dumpfn(obj=complete_list, fn='AlNi_complete.json')
