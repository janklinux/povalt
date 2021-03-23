import os
import glob
import datetime
from pymatgen import Structure
from fireworks import LaunchPad, Workflow
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.powerups import add_modify_incar


def get_relax_wf(structure, struc_name='', name='Static_run', vasp_input_set=None,
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

    fws = [OptimizeFW(structure=structure, vasp_input_set=vis_static, vasp_cmd=vasp_cmd,
                      db_file=db_file, name="{} -- static".format(tag))]
    wfname = "{}: {}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='basf_fw', username='jank', password='b@sf_mongo', ssl=True,
                 ssl_ca_certs=ca_file, ssl_certfile=cl_file)


crystal = os.getcwd().split('/')[-1].split('_')[-1]
if crystal not in ['bcc', 'fcc', 'hcp', 'sc']:
    raise ValueError('This directory is not conform with generator settings, please correct internals...')

all_files = []
for i in [2]:
    # hcp:
    # fcc: 1, 2
    # bcc:
    # sc:
    tmp = glob.glob('training_data/SCEL{:d}*/**/geometry.in'.format(i), recursive=True)
    all_files.extend(tmp)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 1, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 100, 'LAECHG': '.FALSE.', 'LREAL': 'AUTO',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.', 'ISIF': 7, 'IBRION': 2}

spin_key = {'Ni': 2, 'Cr': -2, 'Fe': 3, 'Nb': -1, 'Ta': -1, 'Mo': 1, 'Ti': 1, 'Al': -1}

for file in all_files:
    s = Structure.from_file(file)

    site_properties = {'initial_moment': []}
    for sn in s.species:
        site_properties['initial_moment'].append(spin_key[str(sn)])

    spin_structure = Structure(lattice=s.lattice, species=s.species, coords=s.frac_coords,
                               coords_are_cartesian=False, site_properties=site_properties)

    input_set = MPRelaxSet(spin_structure)
    kpts = Kpoints.automatic_gamma_density(structure=spin_structure, kppa=1200).as_dict()

    bin_name = crystal + '_'
    for ts in s.composition.element_composition:
        bin_name += ts.name

    scel = file.split('/')[1]
    structure_name = '{} {} {} {}'.format(scel, s.composition.reduced_formula, s.num_sites, bin_name)

    meta = {'name': structure_name,
            'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

    relax_wf = get_relax_wf(structure=spin_structure, struc_name=structure_name, name='Iconel CE Relaxation',
                            vasp_input_set=input_set, user_kpoints_settings=kpts, metadata=meta,
                            vasp_cmd='mpirun --bind-to package:report --map-by ppr:1:core:nooversubscribe '
                                     '-n 2 vasp_gpu')
    run_wf = add_modify_incar(relax_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)
