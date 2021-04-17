import datetime
from ase.io import read
from fireworks import LaunchPad, Workflow
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPStaticSet
from atomate.vasp.fireworks import StaticFW
from atomate.vasp.powerups import add_modify_incar


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


lpad = LaunchPad(host='195.148.22.179', port=27017, name='au_fw', username='jank', password='mongo', ssl=False)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 16, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 160, 'LAECHG': '.FALSE.',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.'}

cars = ['OUTCAR_S1', 'OUTCAR_S2', 'OUTCAR_S3', 'OUTCAR_S4', 'OUTCAR_S5', 'OUTCAR_S6',
        'OUTCAR_S7', 'OUTCAR_S8', 'OUTCAR_S9', 'OUTCAR_S10', 'OUTCAR_S11']

for car in cars:
    atoms = read(car, index=':')
    for at in atoms:
        cluster = AseAtomsAdaptor().get_structure(at)
        incar_set = MPStaticSet(cluster)
        structure_name = '{} {} Hannes cluster'.format(len(cluster.sites), str(cluster.symbol_set[0]))
        meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
        kpt_set = Kpoints.gamma_automatic([1, 1, 1], shift=[0, 0, 0])
        static_wf = get_static_wf(structure=cluster, struc_name=structure_name, vasp_input_set=incar_set,
                                  vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                                  user_kpoints_settings=kpt_set, metadata=meta)
        run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
        lpad.add_wf(run_wf)
