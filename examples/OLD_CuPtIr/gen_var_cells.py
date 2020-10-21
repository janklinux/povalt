import os
import time
import numpy as np
from pymatgen.core.structure import Structure, StructureError
from fireworks import Workflow, LaunchPad
from atomate.fhi_aims.fireworks.core import OptimizeFW


def get_optimize_wf(control_in, structure, struc_name='', name_in='Relax',
                    aims_cmd=None, tag=None, metadata=None):

    fws = [OptimizeFW(control=control_in, structure=structure, aims_cmd=aims_cmd, name="{} Relax".format(tag))]

    wfname = "{}:{}".format(struc_name, name_in)
    return Workflow(fws, name=wfname, metadata=metadata)


def get_strain_and_stress(old, new):
    eeps = np.empty((3, 3))
    for ii in range(3):
        for jj in range(3):
            eeps[ii, jj] = 1/2 * (
                (np.linalg.norm(new[ii]) - np.linalg.norm(old[jj])) / np.linalg.norm(old[jj]) +
                (np.linalg.norm(new[jj]) - np.linalg.norm(old[ii])) / np.linalg.norm(old[ii]))
    return eeps


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='tddft_fw', username='jank', password='b@sf_mongo',
                 ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)

atom = []
name = []
lattice = []

with open('base.in', 'r') as f:
    for line in f:
        if 'lattice_vector' in line:
            lattice.append([float(x) for x in line.split()[1:4]])
        if 'atom' in line:
            atom.append([float(x) for x in line.split()[1:4]])
            name.append(line.split()[-1])

atom = np.array(atom)
lattice = np.array(lattice)

np.random.seed(int(time.time()))
total_structures = 5000

i = 0
while i < total_structures:
    de = np.array([[(np.random.random()-0.5)*2.75, (np.random.random()-0.5)*0.85, (np.random.random()-0.5)*0.85],
                   [(np.random.random()-0.5)*0.85, (np.random.random()-0.5)*2.75, (np.random.random()-0.5)*0.85],
                   [(np.random.random()-0.5)*0.85, (np.random.random()-0.5)*0.85, (np.random.random()-0.5)*2.75]])

    new_lat = np.add(lattice, de)

    eps = get_strain_and_stress(old=lattice, new=new_lat)

    valid = True
    for ea in eps:
        for eb in ea:
            if eb > 0.15:
                valid = False

    if not valid:
        print('skipping stress component > 0.15...')
        continue

    new_crds = []
    new_species = []
    for s, c in zip(name, atom):
        new_species.append(s)
        new_crds.append(np.array([(np.random.random()-0.5)*0.15+c[0],
                                  (np.random.random()-0.5)*0.15+c[1],
                                  (np.random.random()-0.5)*0.15+c[2]]))

    try:
        new_cell = Structure(lattice=new_lat, species=new_species, coords=new_crds,
                             charge=None, validate_proximity=True, to_unit_cell=False,
                             coords_are_cartesian=True, site_properties=None)
    except StructureError:
        print('structure error, skipping')
        continue

    structure_name = '{} {} rnd distortion'.format(len(new_cell.sites), str(new_cell.symbol_set))
    meta = {'name': structure_name}

    with open('control.in', 'r') as f:
        control = f.readlines()

    run_wf = get_optimize_wf(control_in=control, structure=new_cell, struc_name=structure_name,
                             aims_cmd='srun aims', metadata=meta)

    lpad.add_wf(run_wf)

    i += 1
