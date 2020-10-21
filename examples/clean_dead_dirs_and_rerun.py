import os
import shutil
from fireworks import LaunchPad

lpad = LaunchPad.auto_load()

for fwid in lpad.get_wf_ids({'state': 'RUNNING'}):
    run_dir = lpad.get_launchdir(fw_id=fwid)
    print('clearing: ', fwid, run_dir)
    if os.path.isdir(run_dir):
        lpad.rerun_fw(fwid)
        shutil.rmtree(run_dir)


for fwid in lpad.get_wf_ids({'state': 'FIZZLED'}):
    run_dir = lpad.get_launchdir(fw_id=fwid)
    print('clearing: ', fwid, run_dir)
    if os.path.isdir(run_dir):
        lpad.rerun_fw(fwid)
        shutil.rmtree(run_dir)
