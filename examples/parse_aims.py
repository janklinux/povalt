import gzip
import numpy as np


parse_atoms = False
parse_forces = False
parse_stress = False

scf = dict()
i_step = 0
i_atom = 0
i_force = 0
n_atoms = None
md_lattice = []

with gzip.open('run.gz', 'rt') as f:
    content = f.readlines()

md_run = False
relaxation = False
for line in content:
    if 'MD_run' in line:
        md_run = True
    if 'relax_geometry' in line:
        if not line.strip().startswith('#'):
            relaxation = True

if md_run:
    for line in content:
        if '| Number of atoms' in line:
            n_atoms = int(line.split()[5])
        if 'lattice_vector' in line:
            md_lattice.append([float(x) for x in line.split()[1:4]])
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
            if len(line.split()) == 0:
                continue
            elif 'atom' in line:
                scf[i_step]['atoms'].append([float(x) for x in line.split()[1:4]])
                scf[i_step]['species'].append(line.split()[4])
                i_atom += 1
            if i_atom == n_atoms:
                parse_atoms = False
        if 'Atomic structure (and velocities) as used in the preceding time step:' in line:
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

elif relaxation:
    for line in content:
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

else:
    scf[1] = {'atoms': [], 'forces': [], 'lattice': [], 'stress': [],
              'species': [], 'energy': 0.0}
    for line in content:
        if '| Number of atoms' in line:
            n_atoms = int(line.split()[5])
        if 'lattice_vector' in line:
            scf[1]['lattice'].append([float(x) for x in line.split()[1:4]])
        if line.strip().startswith('atom') or line.strip().startswith('atom_frac'):
            scf[1]['atoms'].append([float(x) for x in line.split()[1:4]])
            scf[1]['species'].append(line.split()[4])
        if '| Total energy                  :' in line:
            scf[1]['energy'] = float(line.split()[6])
        if parse_forces:
            scf[1]['forces'].append([float(x) for x in line.split()[2:5]])
            i_force += 1
            if i_force == n_atoms:
                parse_forces = False
        if 'Total atomic forces (unitary forces cleaned) [eV/Ang]:' in line:
            parse_forces = True
        if parse_stress:
            if '|  x       ' in line:
                scf[1]['stress'].append([float(x) for x in line.split()[2:5]])
            if '|  y       ' in line:
                scf[1]['stress'].append([float(x) for x in line.split()[2:5]])
            if '|  z       ' in line:
                scf[1]['stress'].append([float(x) for x in line.split()[2:5]])
            if '|  Pressure:' in line:
                parse_stress = False
        if 'Analytical stress tensor - Symmetrized' in line:
            parse_stress = True
        if 'No geometry change - only restart of scf mixer after initial iterations.' in line:
            i_force = 0
            scf[1]['forces'] = []
            scf[1]['stress'] = []

with gzip.open('parsed.xyz.gz', 'wt') as f:
    for i in range(1, len(scf) + 1):
        if md_run:
            scf[i]['lattice'] = md_lattice
            lat = md_lattice
            scf[i]['stress'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            stress = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        elif relaxation:
            if len(scf[i]['stress']) > 0:
                lat = np.array([[scf[i]['lattice'][0][0], scf[i]['lattice'][0][1], scf[i]['lattice'][0][2]],
                                [scf[i]['lattice'][1][0], scf[i]['lattice'][1][1], scf[i]['lattice'][1][2]],
                                [scf[i]['lattice'][2][0], scf[i]['lattice'][2][1], scf[i]['lattice'][2][2]]])
                stress = np.array([[scf[i]['stress'][0][0], scf[i]['stress'][0][1], scf[i]['stress'][0][2]],
                                   [scf[i]['stress'][1][0], scf[i]['stress'][1][1], scf[i]['stress'][1][2]],
                                   [scf[i]['stress'][2][0], scf[i]['stress'][2][1], scf[i]['stress'][2][2]]])
            elif len(scf[i]['stress']) == 0 and len(scf[i]['lattice']) > 0:
                lat = np.array([[scf[i]['lattice'][0][0], scf[i]['lattice'][0][1], scf[i]['lattice'][0][2]],
                                [scf[i]['lattice'][1][0], scf[i]['lattice'][1][1], scf[i]['lattice'][1][2]],
                                [scf[i]['lattice'][2][0], scf[i]['lattice'][2][1], scf[i]['lattice'][2][2]]])
                scf[i]['stress'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                stress = scf[i]['stress']
            else:
                scf[i]['lattice'] = [[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]]
                lat = scf[i]['lattice']
                scf[i]['stress'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                stress = scf[i]['stress']
        else:
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
                ' config_type=CEbulk\n'.format(
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
