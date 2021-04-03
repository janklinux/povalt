import os
import sys
import datetime
from fireworks import LaunchPad, Workflow
from pymatgen import Structure
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

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 4, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 60, 'LAECHG': '.FALSE.', 'LREAL': '.FALSE.',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.'}

spin_key = {'Ni': 2, 'Cr': -2, 'Fe': 3, 'Nb': -1, 'Ta': -1, 'Mo': 1, 'Ti': 1, 'Al': -1}

for wfid in lpad.get_wf_ids({'state': 'FIZZLED'}):
    print('Processing WF {}...'.format(wfid))
    sys.stdout.flush()

    fw = lpad.get_fw_by_id(wfid)
    ldir = lpad.get_launchdir(fw_id=wfid)
    fw_dict = lpad.get_wf_by_fw_id(fw.fw_id).as_dict()

    trace = fw_dict['fws'][-1]['launches'][-1]['action']['stored_data']['_exception']['_stacktrace']
    print(trace)
    # if 'Max errors per job reached:' not in trace:
    #     raise NotImplementedError('This is not a max step error, check what happened with FW: {} in dir {}'
    #                               .format(wfid, ldir))

    if not os.path.isdir(ldir):
        raise FileNotFoundError('Are you on the right machine? '
                                'Workflow {} directory does not exist here... : {}'.format(wfid, ldir))

    s = Structure.from_file(os.path.join(ldir, 'CONTCAR.gz'))

    site_properties = {'initial_moment': []}
    for sn in s.species:
        site_properties['initial_moment'].append(spin_key[str(sn)])

    spin_structure = Structure(lattice=s.lattice, species=s.species, coords=s.frac_coords,
                               coords_are_cartesian=False, site_properties=site_properties)

    input_set = MPRelaxSet(spin_structure)
    kpts = Kpoints.automatic_gamma_density(structure=spin_structure, kppa=1200).as_dict()

    relax_wf = get_relax_wf(structure=spin_structure, struc_name=fw_dict['metadata']['name'], name='CE Relaxation',
                            vasp_input_set=input_set, user_kpoints_settings=kpts, metadata=fw_dict['metadata'],
                            vasp_cmd='srun --nodes=1 --ntasks=8 --ntasks-per-node=8 '
                                     '--mem-per-cpu=1800 --exclusive vasp_std')
    run_wf = add_modify_incar(relax_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)
    lpad.delete_wf(wfid, delete_launch_dirs=True)
