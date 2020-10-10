import os
import io
import shutil
import pymongo
import numpy as np
from fireworks import LaunchPad
from ase.io import read as aseread
from ase.io import write as asewrite
from pymatgen.io.vasp import Vasprun, Outcar


ca_file = os.path.expanduser('~/ssl/numphys/ca.crt')
cl_file = os.path.expanduser('~/ssl/numphys/client.pem')
lpad = LaunchPad(host='numphys.org', port=27017, name='fw_run', username='jank', password='b@sf_mongo',
                 ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)

run_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True, ssl_ca_certs=ca_file, ssl_certfile=cl_file)
data_db = run_con.pot_train
data_db.authenticate('jank', 'b@sf_mongo')
data_coll = data_db['aurum']

for wfid in lpad.get_wf_ids({'state': 'COMPLETED'}):
    print('Processing WF {}...'.format(wfid))
    fw = lpad.get_fw_by_id(wfid)
    ldir = lpad.get_launchdir(fw_id=wfid)
    vrun = os.path.join(ldir, 'vasprun.xml.gz')
    orun = os.path.join(ldir, 'OUTCAR.gz')

    if not os.path.isdir(ldir):
        print('Are you on the right machine? Workflow {} directory does not exist here...'.format(wfid))

    run = Vasprun(vrun)
    runo = Outcar(orun)

    if not run.converged or not run.converged_electronic:
        raise ValueError('Run {} is NOT converged, something is very wrong here...'.format(wfid))

    atoms = aseread(vrun)

    for a in atoms:
        print(a)

    xyz = ''
    file = io.StringIO()
    asewrite(filename=file, images=atoms, format='xyz')
    file.seek(0)
    for f in file:
        xyz += f
    file.close()

    stress = atoms.get_stress()
    vol = atoms.get_volume()

    print(stress)
    print(vol)
    print(-np.dot(vol, stress))

    print('free EN from atoms:', atoms.get_potential_energy())
    print('free EN from outcar:', runo.final_energy)
    print('free EN from vasprun:', run.final_energy)

    quit()

    dft_data = dict()
    dft_data['xyz'] = xyz
    dft_data['PBE_54'] = run.potcar_symbols
    dft_data['parameters'] = run.parameters.as_dict()
    dft_data['free_energy'] = runo.final_energy  # this is the FREE energy, different from vasprun.xml
    dft_data['final_structure'] = run.final_structure.as_dict()

    if runo.final_energy > 0:
        print('Energy > 0, ignoring result...')
        shutil.rmtree(ldir)
        lpad.delete_wf(wfid)
        continue

    fw_dict = lpad.get_wf_by_fw_id(fw.fw_id).as_dict()
    data_name = 'Au random structure  ||  ' + fw_dict['metadata']['name'] + \
                '  ||  created ' + fw_dict['metadata']['date'] + '  ||  StaticFW'

    print(data_name)
    quit()

    data_coll.insert_one({'name': data_name, 'data': dft_data})
    shutil.rmtree(ldir)
    lpad.delete_wf(wfid)
