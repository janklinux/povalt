import os
import gzip
from fireworks import LaunchPad

lpad = LaunchPad.auto_load()

for wfid in lpad.get_fw_ids({'state': 'FIZZLED'}):
    auto_rerun = False
    print('WF: {}...'.format(wfid))

    fw = lpad.get_fw_by_id(wfid)
    ldir = lpad.get_launchdir(fw_id=wfid)
    fw_dict = lpad.get_wf_by_fw_id(fw.fw_id).as_dict()

    with gzip.open(os.path.join(ldir, 'std_err.txt.gz'), 'rt') as f:
        stderr = f.readlines()

    for line in stderr:
        if 'WARNING: There was an error initializing an OpenFabrics device.' in line:
            auto_rerun = True
        if 'Requested node configuration is not available' in line:
            auto_rerun = True

    if auto_rerun:
        lpad.rerun_fw(wfid)
    else:
        print(stderr)

#    trace = fw_dict['fws'][-1]['launches'][-1]['action']['stored_data']['_exception']['_stacktrace']
#    if 'Max errors' not in trace:
#        print(trace)
#        continue
        # raise NotImplementedError('This is not a max step error, check what happened with FW: {} in dir {}'
