import os
import datetime
import numpy as np
from fireworks import Workflow, LaunchPad
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import StaticFW
from atomate.vasp.powerups import add_modify_incar


def get_epsilon(structure):
    eps = np.array([float(0.2*(x-0.5)) for x in np.random.random(6)])
    cij = np.array([[1+eps[0], eps[5]/2, eps[4]/2],
                    [eps[5]/2, 1+eps[1], eps[3]/2],
                    [eps[4]/2, eps[3]/2, 1+eps[2]]])
    return Lattice(np.transpose(np.dot(cij, np.transpose(structure.lattice.matrix))))


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

# ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
# cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
# lpad = LaunchPad(host='numphys.org', port=27017, name='tddft_fw', username='jank', password='b@sf_mongo', ssl=True,
#                  ssl_ca_certs=ca_file, ssl_certfile=cl_file)


crystal = os.getcwd().split('/')[-1]
if crystal not in ['bcc', 'fcc', 'hcp', 'sc']:
    raise ValueError('This directory is not conform with generator settings, please correct internals...')

if crystal == 'bcc':
    scale_mat = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
elif crystal == 'fcc':
    scale_mat = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
elif crystal == 'hcp':
    raise NotImplementedError('Not valid for HCP...')
    # scale_mat = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 4]])  # 108 atoms in hcp
elif crystal == 'sc':
    scale_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
else:
    raise ValueError('No elastics implemented for the chosen cell...')

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 2, 'ISMEAR': 0, 'ISPIN': 1,
             'ALGO': 'Normal', 'NELM': 60, 'LAECHG': '.FALSE.', 'LREAL': '.FALSE.',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.', 'LVHAR': '.FALSE.'}


use_epsilon = True
displacements = np.linspace(start=-0.2, stop=0.2, num=1000, endpoint=True)


for ds in displacements:
    prim = Structure.from_file('CONTCAR')
    # prim.make_supercell(scaling_matrix=[2, 2, 2], to_unit_cell=True)

    if use_epsilon:
        prim.lattice = get_epsilon(prim)
    else:
        prim.lattice = Lattice(prim.lattice.matrix + np.dot(scale_mat, ds))

    new_crds = []
    new_species = []
    for s in prim.sites:
        new_species.append(s.specie)
        new_crds.append(np.array([(np.random.random()-0.5)*0.02+s.coords[0],
                                  (np.random.random()-0.5)*0.02+s.coords[1],
                                  (np.random.random()-0.5)*0.02+s.coords[2]]))

    new_cell = Structure(lattice=prim.lattice, species=new_species, coords=new_crds,
                         charge=None, validate_proximity=True, to_unit_cell=False,
                         coords_are_cartesian=True, site_properties=None)

    incar_set = MPStaticSet(prim)
    structure_name = '{} {} for elastics'.format(len(prim.sites), str(prim.symbol_set[0]))
    meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
    kpt_set = Kpoints.automatic_density(structure=prim, kppa=1000).as_dict()

    # vasp_cmd = 'mpirun --bind-to package:report --map-by ppr:1:core:nooversubscribe -n 2 vasp_std',

    static_wf = get_static_wf(structure=prim, struc_name=structure_name, vasp_input_set=incar_set,
                              vasp_cmd='srun --nodes=1 --ntasks=4 --ntasks-per-node=4 '
                                       '--mem-per-cpu=1800 --exclusive vasp_std',
                              user_kpoints_settings=kpt_set, metadata=meta)
    run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)
