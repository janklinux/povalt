import re
import os
import datetime
import numpy as np
from fireworks import LaunchPad, Workflow
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.powerups import add_modify_incar
from atomate.vasp.fireworks import StaticFW


def get_static_wf(in_structure, struc_name='', name='', vasp_input_set=None,
                  vasp_cmd=None, user_kpoints_settings=None, tag=None, metadata=None):

    if vasp_input_set is None:
        raise ValueError('INPUTSET needs to be defined...')
    if user_kpoints_settings is None:
        raise ValueError('You have to specify the K-grid...')
    if vasp_cmd is None:
        raise ValueError('vasp_cmd needs to be set by user...')
    if tag is None:
        tag = datetime.datetime.now().strftime('%Y/%m/%d-%T')

    vis = vasp_input_set
    vs = vis.as_dict()
    vs.update({"user_kpoints_settings": user_kpoints_settings})
    vis_relax = vis.__class__.from_dict(vs)
    fws = [StaticFW(structure=in_structure, vasp_input_set=vis_relax, vasp_cmd=vasp_cmd,
                    name="{} -- relax".format(tag))]
    wfname = "{}: {}".format(struc_name, name)
    return Workflow(fws, name=wfname, metadata=metadata)


lpad = LaunchPad(host='195.148.22.179', port=27017, name='cu_fw', username='jank', password='mongo', ssl=False)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 16, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2, 'NELM': 200,
             'ALGO': 'Normal', 'LAECHG': '.FALSE.', 'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.', 'LVHAR': '.FALSE.'}


for si, structure in enumerate(usable_structures):
    os.chdir(base_dir)
    if os.path.isdir(str(si)):
        continue
    os.mkdir(str(si))
    os.chdir(str(si))

    # print('Running: {}'.format(structure.composition.element_composition))
    #
    # if len(structure.composition.chemical_system.split('-')) < 2:
    #     print('Skipped because of single-species')
    #     continue
    #
    # for file in os.listdir(pot_dir):
    #     if file.startswith('CuAu') or file.startswith('compre'):
    #         os.symlink(os.path.join(pot_dir, file), file)
    #
    # with open('atom.pos', 'w') as f:
    #     write_lammps_data(f, atoms=AseAtomsAdaptor().get_atoms(structure=structure), units='metal',
    #                       specorder=['Cu', 'Au'])
    #
    # with open('lammps.in', 'w') as f:
    #     for line in lammps_input:
    #         f.write(line + '\n')
    #
    # os.system('nice -n 15 lmp -in lammps.in')
    #
    # with open('final_positions.atom', 'r') as f:
    #     relaxed = AseAtomsAdaptor().get_structure(atoms=read_lammps_dump_text(f))
    #
    # spec_map = {'H': 'Cu', 'He': 'Au'}
    # relaxed.replace_species(species_mapping=spec_map)
    # relaxed.to(filename='final.vasp', fmt='POSCAR')

    relaxed = structure

    site_properties = {'initial_moment': []}
    for s in relaxed.sites:
        if s.specie.name == 'Cu':
            site_properties['initial_moment'].append(1.0)
    relaxed = Structure(lattice=relaxed.lattice, species=relaxed.species, coords=relaxed.frac_coords,
                        coords_are_cartesian=False, site_properties=site_properties)
    incar_set = MPStaticSet(relaxed)
    structure_name = re.sub(' ', '', str(relaxed.composition.element_composition)) + ' '
    structure_name += str(relaxed.num_sites) + ' in cluster'
    meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
    kpt_set = Kpoints.automatic_density(structure=relaxed, kppa=1000).as_dict()
    static_wf = get_static_wf(in_structure=relaxed, struc_name=structure_name, vasp_input_set=incar_set,
                              vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                              user_kpoints_settings=kpt_set, metadata=meta)
    run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)
