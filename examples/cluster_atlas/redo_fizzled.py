import os
import shutil
from fireworks import LaunchPad
from pymatgen import Structure
from povalt.firetasks.wf_generators import AimsRelaxLightTight, AimsSingleBasis


base_dir = os.getcwd()
with open('control.in', 'r') as f:
    control = f.readlines()

lpad = LaunchPad(host='195.148.22.179', port=27017, name='train_fw', username='jank', password='mongo', ssl=False)

for fwid in lpad.get_wf_ids({'state': 'FIZZLED'}):
    os.chdir(base_dir)
    run_dir = lpad.get_launchdir(fw_id=fwid)
    print('redoing running: ', fwid, run_dir)
    if os.path.isdir(run_dir):
        fw = lpad.get_fw_by_id(fwid)
        fw_dict = lpad.get_wf_by_fw_id(fw.fw_id).as_dict()
        os.chdir(run_dir)
        with open('std_err.txt', 'r') as f:
            for line in f:
                if 'cgroup out-of-memory handler' in line:
                    print('OOM-kill, check settings for nodes')
                    if str(input('Continue? (Y/N)')) != 'Y':
                        quit()
        if os.path.isfile('geometry.in.next_step'):
            shutil.copy(src='geometry.in.next_step', dst='continue.in')
            cont_geo = Structure.from_file('continue.in')
        else:
            cont_geo = Structure.from_file('geometry.in')
        with open('control.in', 'r') as f:
            basis_set = None
            for line in f:
                if 'Suggested "light" defaults' in line:
                    basis_set = 'light'
                if 'Suggested "tight" defaults' in line:
                    basis_set = 'tight'

        if basis_set is None:
            raise ValueError('Error reading basis settings from control.in')

        if basis_set == 'light':
            rerun_fw = AimsRelaxLightTight(aims_cmd='srun --nodes=5 --ntasks=640 --ntasks-per-node=128',
                                           control=control, structure=cont_geo,
                                           basis_dir='/users/kloppej1/compile/FHIaims/species_defaults',
                                           metadata=fw_dict['metadata'],
                                           name='rerun-continue light -> tight')
        elif basis_set == 'tight':
            rerun_fw = AimsSingleBasis(aims_cmd='srun --nodes=10 --ntasks=1280 --ntasks-per-node=128',
                                       control=control, structure=cont_geo, basis_set='tight',
                                       basis_dir='/users/kloppej1/compile/FHIaims/species_defaults',
                                       metadata=fw_dict['metadata'],
                                       name='rerun-continue tight')
        else:
            raise ValueError('Error during basis set detection')
        lpad.add_wf(rerun_fw)
        lpad.delete_wf(fwid, delete_launch_dirs=True)


# for fwid in lpad.get_wf_ids({'state': 'RUNNING'}):
#     run_dir = lpad.get_launchdir(fw_id=fwid)
#     print('clearing: ', fwid, run_dir)
#     if os.path.isdir(run_dir):
#         lpad.rerun_fw(fwid)
#         shutil.rmtree(run_dir)
