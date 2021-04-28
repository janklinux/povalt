import os
import re
import glob
import time
import datetime
import numpy as np
from fireworks import LaunchPad
from pymatgen import Structure, Lattice
from pymatgen.io.vasp import Kpoints
from povalt.firetasks.wf_generators import aims_single_basis


np.random.seed(int(time.time()))

ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='tddft_fw', username='jank', password='b@sf_mongo', ssl=True,
                 ssl_ca_certs=ca_file, ssl_certfile=cl_file)


cars = glob.glob('relaxed_inputs_from_substitution/**/POSCAR', recursive=True)
all_structures = list()
for car in cars:
    all_structures.append(dict({'structure': Structure.from_file(car), 'mpid': car.split('/')[-2]}))
print('Structures total: {}'.format(len(all_structures)))

for element in all_structures:
    for _ in range(10):
        with open('control.in', 'r') as f:
            control = f.readlines()

        kpts = Kpoints.automatic_gamma_density(structure=element['structure'], kppa=1000).as_dict()
        aims_kpts = 'k_grid {} {} {}'.format(kpts['kpoints'][0][0], kpts['kpoints'][0][1], kpts['kpoints'][0][2])
        aims_shft = 'k_offset {} {} {}'.format(kpts['usershift'][0], kpts['usershift'][1], kpts['usershift'][2])
        processed_control = []
        for line in control:
            processed_control.append(re.sub('k_grid', aims_kpts, re.sub('k_offset', aims_shft, line)))

        new_lat = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                new_lat[i, j] = ((np.random.random()-0.5)*0.3+1)*element['structure'].lattice.matrix[i, j]

        new_crds = []
        new_species = []
        site_properties = dict({'initial_moment': []})
        element['structure'].lattice = Lattice(new_lat)
        for ic, (c, s) in enumerate(zip(element['structure'].cart_coords, element['structure'].species)):
            new_species.append(s)
            new_crds.append(np.array([(np.random.random()-0.5)*0.3+c[0],
                                      (np.random.random()-0.5)*0.3+c[1],
                                      (np.random.random()-0.5)*0.3+c[2]]))
            if s.name == 'Ir':
                site_properties['initial_moment'].append(-1.0*(-1.0)**(ic+1))
            else:
                site_properties['initial_moment'].append(-1.0**(ic+1))

        new_cell = Structure(lattice=new_lat, species=new_species, coords=new_crds,
                             charge=None, validate_proximity=True, to_unit_cell=False,
                             coords_are_cartesian=True, site_properties=site_properties)

        structure_name = '{} random distortion {}'.format(new_cell.composition.element_composition, element['mpid'])
        meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

        static_wf = aims_single_basis(aims_cmd='mpirun -n 2 aims', control=processed_control,
                                      structure=new_cell, basis_set='light',
                                      basis_dir='/home/jank/work/compile/FHIaims/species_defaults/defaults_2020',
                                      metadata=meta, name='single point light')
        lpad.add_wf(static_wf)
