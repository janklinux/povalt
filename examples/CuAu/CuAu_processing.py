import os
import time
import datetime
import numpy as np
from pymongo import MongoClient
from pymatgen.core.structure import Structure, StructureError
from fireworks import LaunchPad, Workflow
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.transformations.standard_transformations import SupercellTransformation
from atomate.vasp.fireworks.core import StaticFW
from atomate.vasp.powerups import add_modify_incar


def scale(idx):
    return np.array([[np.mod(idx, np.floor(idx/4)+1)+1, 0, 0],
                     [0, np.mod(idx, np.floor(idx/3)+1)+1, 0],
                     [0, 0, np.mod(idx, np.floor(idx/2)+1)+1]])


def get_static_wf(structure, struc_name='', name='Static_run', vasp_input_set=None,
                  vasp_cmd=None, db_file=None, user_kpoints_settings=None, tag=None, metadata=None):

    if vasp_input_set is None:
        raise ValueError('INPUTSET needs to be defined...')
    if user_kpoints_settings is None:
        raise ValueError('You have to specify the K-grid...')
    if vasp_cmd is None:
        raise ValueError('vasp_cmd needs to be set by user...')
    if tag is None:
        tag = datetime.datetime.now().strftime('%Y/%m/%d-%T')

    vis = vasp_input_set
    v = vis.as_dict()
    v.update({"user_kpoints_settings": user_kpoints_settings})
    vis_static = vis.__class__.from_dict(v)

    fws = [StaticFW(structure=structure, vasp_input_set=vis_static, vasp_cmd=vasp_cmd,
                    db_file=db_file, name="{} -- static".format(tag))]
    wfname = "{}: {}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
run_con = MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['CuAu']

lpad_cuau = LaunchPad(host='195.148.22.179', port=27017, name='cuau_fw', username='jank', password='mongo', ssl=False)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 8, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 200, 'LAECHG': 'False',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.'}

np.random.seed(int(time.time()))

for doc in data_coll.find({}):
    if 'CuAu CASM generated and relaxed structure for multiplication and rnd distortion' in doc['name']:
        bin_name = doc['name'].split('||')[1].split()[3]
        s = Structure.from_dict(doc['data']['final_structure'])
        cell = None
        kpt_set = None
        for i in range(1, 19):
            cell = SupercellTransformation(scaling_matrix=np.array(scale(i))).apply_transformation(s)
            if cell.num_sites > 120:
                kpt_set = Kpoints.automatic_gamma_density(structure=cell, kppa=1000).as_dict()
                break

        if cell is None or kpt_set is None:
            raise ValueError('Supercell trafo failed...')

        print('Number of atoms in supercell: {} || Composition: {}'
              .format(cell.num_sites, cell.composition.element_composition))

        for _ in range(5):  # 10 random structures per configuration
            # Create random distorted lattice
            added = False
            while not added:
                new_lat = np.empty((3, 3))
                for i in range(3):
                    for j in range(3):
                        new_lat[i, j] = (1 + (np.random.random() - 0.5) * 0.2) * cell.lattice.matrix[i, j]

                # Random distort the coordinates [and apply random magnetic moments]
                new_crds = []
                new_species = []
                site_properties = dict({'initial_moment': []})
                for s in cell.sites:
                    new_species.append(s.specie)
                    new_crds.append(np.array([(np.random.random() - 0.5) * 0.3 + s.coords[0],
                                              (np.random.random() - 0.5) * 0.3 + s.coords[1],
                                              (np.random.random() - 0.5) * 0.3 + s.coords[2]]))
                    if s.specie.name == 'Au':
                        site_properties['initial_moment'].append(1.0)
                    else:
                        site_properties['initial_moment'].append(-1.0)

                try:
                    new_cell = Structure(lattice=new_lat, species=new_species, coords=new_crds,
                                         charge=None, validate_proximity=True, to_unit_cell=False,
                                         coords_are_cartesian=True, site_properties=site_properties)
                except StructureError:
                    continue

                show_it = False
                if show_it:
                    new_cell.to(fmt='POSCAR', filename='/tmp/POSCAR')
                    os.system('VESTA /tmp/POSCAR')
                    quit()

                incar_set = MPStaticSet(new_cell)
                structure_name = str(new_cell.composition.element_composition)

                meta = {'name': structure_name,
                        'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

                static_wf = get_static_wf(structure=new_cell, struc_name=structure_name, vasp_input_set=incar_set,
                                          vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                                          user_kpoints_settings=kpt_set, metadata=meta)
                run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
                lpad_cuau.add_wf(run_wf)
                added = True
