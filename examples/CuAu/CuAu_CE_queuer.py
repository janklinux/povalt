import glob
import datetime
from pymatgen import Structure
from fireworks import LaunchPad, Workflow
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.powerups import add_modify_incar


def get_optimize_wf(structure, struc_name='', name='Relax', vasp_input_set=None,
                    vasp_cmd=None, db_file=None, user_kpoints_settings=None, tag=None, metadata=None):

    if vasp_input_set is None:
        raise ValueError('INPUTSET needs to be defined...')
    if user_kpoints_settings is None:
        raise ValueError('You have to specify the K-grid...')
    if vasp_cmd is None:
        raise ValueError('vasp_cmd needs to be set by user...')
    if tag is None:
        tag = datetime.datetime.now().strftime('%Y/%m/%d-%T')

    vis_relax = vasp_input_set
    v = vis_relax.as_dict()
    v.update({"user_kpoints_settings": user_kpoints_settings})
    vis_relax = vis_relax.__class__.from_dict(v)

    fws = [OptimizeFW(structure=structure, vasp_input_set=vis_relax, vasp_cmd=vasp_cmd,
                      db_file=db_file, name="{} Relax".format(tag))]
    wfname = "{}:{}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


lpad = LaunchPad(host='195.148.22.179', port=27017, name='cuau_fw', username='jank', password='mongo', ssl=False)
lpad.reset('2021-01-20')

incar_mod = {'EDIFF': 1E-5, 'EDIFFG': 1E-3, 'NSW': 101, 'ISMEAR': 0, 'ISPIN': 1, 'ISYM': 0, 'NELM': 100,
             'ENCUT': 520, 'NCORE': 8, 'ALGO': 'Normal', 'AMIN': 0.1, 'LREAL': '.FALSE.'}

all_files = glob.glob('training_data/SCEL*/**/POSCAR', recursive=True)

for file in all_files:
    s = Structure.from_file(file)
    kpt_set = Kpoints.automatic_gamma_density(structure=s, kppa=1000).as_dict()
    print('Number of atoms in primitive cell: {} || Composition: {}'
          .format(s.num_sites, s.composition.element_composition))

    incar_set = MPRelaxSet(s)
    structure_name = str(s.composition.element_composition)

    meta = {'name': structure_name,
            'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}

    # vasp_cmd='mpirun --bind-to package:report --map-by ppr:1:core:nooversubscribe -n 2 vasp_std',
    # vasp_cmd='srun --ntasks=8 --mem-per-cpu=1800 --exclusive vasp_std',

    relax_wf = get_optimize_wf(structure=s, struc_name=structure_name, vasp_input_set=incar_set,
                               vasp_cmd='srun --nodes=1 --ntasks=16 --ntasks-per-node=16 '
                                        '--mem-per-cpu=1800 --exclusive vasp_std',
                               user_kpoints_settings=kpt_set, metadata=meta)
    run_wf = add_modify_incar(relax_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)
