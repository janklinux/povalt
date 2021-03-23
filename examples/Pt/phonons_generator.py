import os
import time
import random
import datetime
import numpy as np
from fireworks import Workflow, LaunchPad
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import StaticFW
from atomate.vasp.powerups import add_modify_incar


def get_epsilon(structure):
    eps = np.array([float(0.1*(x-0.5)) for x in np.random.random(6)])
    cij = np.array([[1+eps[0], eps[5]/2, eps[4]/2],
                    [eps[5]/2, 1+eps[1], eps[3]/2],
                    [eps[4]/2, eps[3]/2, 1+eps[2]]])
    print(structure.lattice.matrix)
    print(np.dot(cij, structure.lattice.matrix))


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


lpad = LaunchPad(host='195.148.22.179', port=27017, name='phonon_fw', username='jank', password='mongo', ssl=False)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 8, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 1,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 200, 'LAECHG': 'False',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.'}


random.seed(time.time())
total_structures = 50

phonon_inputs = [s for s in os.listdir('poscars') if s.startswith('POS')]

for ph_in in phonon_inputs:
    undistorted = Structure.from_file(os.path.join('poscars', ph_in))
    structure_name = '{} {} for phonons'.format(len(undistorted.sites), str(undistorted.symbol_set[0]))
    meta = {'name': structure_name,
            'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
    incar_set = MPStaticSet(undistorted)
    kpt_set = Kpoints.automatic_gamma_density(structure=undistorted, kppa=1200).as_dict()

    static_wf = get_static_wf(structure=undistorted, struc_name=structure_name, vasp_input_set=incar_set,
                              vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                              user_kpoints_settings=kpt_set, metadata=meta)
    run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)

    i = 0
    while i < total_structures:
        phonon_input = Structure.from_file(os.path.join('poscars', ph_in))

        max_dist = np.minimum(0.1 * float(ph_in.split('_')[-1]), 0.05)

        new_crds = []
        new_species = []
        for s in phonon_input.sites:
            new_species.append(s.specie)
            new_crds.append(np.array([(random.random() - 0.5) * max_dist + s.coords[0],
                                      (random.random() - 0.5) * max_dist + s.coords[1],
                                      (random.random() - 0.5) * max_dist + s.coords[2]]))

        new_cell = Structure(lattice=phonon_input.lattice, species=new_species, coords=new_crds,
                             charge=None, validate_proximity=True, to_unit_cell=False,
                             coords_are_cartesian=True, site_properties=None)

        incar_set = MPStaticSet(new_cell)
        structure_name = '{} {} for phonons'.format(len(new_cell.sites), str(new_cell.symbol_set[0]))
        meta = {'name': structure_name,
                'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
        kpt_set = Kpoints.automatic_gamma_density(structure=new_cell, kppa=1200).as_dict()

        static_wf = get_static_wf(structure=new_cell, struc_name=structure_name, vasp_input_set=incar_set,
                                  vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                                  user_kpoints_settings=kpt_set, metadata=meta)

        run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
        lpad.add_wf(run_wf)

        i += 1
