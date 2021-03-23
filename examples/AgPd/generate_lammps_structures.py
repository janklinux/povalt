import re
import os
import shutil
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
from atomate.vasp.fireworks.core import StaticFW
from atomate.vasp.powerups import add_modify_incar


def get_static_wf(in_structure, struc_name='', name='Static_run', vasp_input_set=None,
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

    fws = [StaticFW(structure=in_structure, vasp_input_set=vis_static, vasp_cmd=vasp_cmd,
                    db_file=db_file, name="{} -- static".format(tag))]

    wfname = "{}: {}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


make_ball = False
make_cyl = False

species = ['Pd', 'Ag']
systems = ['bcc', 'fcc', 'sc', 'hcp']

all_structures = []

for csys in systems:
    for spec in species:
        prim = Structure.from_file('_'.join(['POSCAR', csys, spec]))

        scale = {'bcc': np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
                 'fcc': np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
                 'hcp': np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
                 'sc':  np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])}

        cell = SupercellTransformation(scaling_matrix=scale[csys]).apply_transformation(prim)

        print(csys, spec, cell.num_sites)

        center = np.zeros(3)
        for lv in cell.lattice.matrix:
            center += lv / 2

        if spec == 'Ag':
            replace_species = 'Pd'
        else:
            replace_species = 'Ag'

        dist_list = []
        prev_num = 1
        for d in np.linspace(start=2, stop=6.5, num=20, endpoint=True):
            if len(cell.get_sites_in_sphere(center, d)) > prev_num:
                # print('found: ', len(cell.get_sites_in_sphere(center, d)), d)
                dist_list.append([d, len(cell.get_sites_in_sphere(center, d))])
                prev_num = len(cell.get_sites_in_sphere(center, d))
                for s in cell.get_sites_in_sphere(center, d):
                    for i, t in enumerate(cell.sites):
                        if s[0] == t:
                            cell.replace(i, species=replace_species)

                cell.sort()
                all_structures.append(cell)

                if make_ball:
                    cell.to(filename='{}_ball_embedded_in_{}_{}_{}.vasp'.format(replace_species, csys, spec, prev_num),
                            fmt='POSCAR')

        cell = SupercellTransformation(scaling_matrix=scale[csys]).apply_transformation(prim)

        for rad, atnum in dist_list:
            new_coords = []
            new_species = []
            for c, s in zip(cell.cart_coords, cell.species):
                if np.linalg.norm([c[0] - center[0], c[1] - center[1]]) < rad:
                    new_coords.append(c)
                    new_species.append(replace_species)
                else:
                    new_coords.append(c)
                    new_species.append(spec)

            new_cell = Structure(lattice=cell.lattice, species=new_species, coords=new_coords,
                                 coords_are_cartesian=True, to_unit_cell=False, site_properties=None)
            new_cell.sort()
            all_structures.append(new_cell)

            if make_cyl:
                new_cell.to(filename='{}_cyl_embedded_in_{}_{}_{}.vasp'.format(replace_species, csys, spec, atnum),
                            fmt='POSCAR')


derived_structures = []

for structure in all_structures:
    center = np.zeros(3)
    for lv in structure.lattice.matrix:
        center += lv / 2

    layer_dist = structure.get_distance(0, 1)

    for vac_dir in [0, 1, 2]:
        tmp = Structure(lattice=structure.lattice, species=structure.species, coords=structure.frac_coords,
                        coords_are_cartesian=False, to_unit_cell=False)
        rem_idx = []
        for si, s in enumerate(tmp.cart_coords):
            if np.abs(s[vac_dir] - center[vac_dir]) > layer_dist + 0.2:
                rem_idx.append(si)

        tmp.remove_sites(rem_idx)
        # tmp.to(filename='test.vasp', fmt='POSCAR')
        derived_structures.append(tmp)

    tmp = Structure(lattice=structure.lattice, species=structure.species, coords=structure.frac_coords,
                    coords_are_cartesian=False, to_unit_cell=False)
    rem_idx = []
    for si, s in enumerate(tmp.cart_coords):
        if np.abs(s[2] - center[2]) > 3.5:
            rem_idx.append(si)

    tmp.remove_sites(rem_idx)
    sux = Structure(lattice=[tmp.lattice.matrix[0], tmp.lattice.matrix[1], 2*tmp.lattice.matrix[2]],
                    species=tmp.species, coords=tmp.cart_coords, coords_are_cartesian=True, to_unit_cell=False)

    fuckarray = []
    for c in sux.cart_coords:
        fuckarray.append(c[2])

    middle = np.amax(fuckarray) - np.amin(fuckarray)
    move_dist = sux.lattice.matrix[2][2] / 2 - middle

    scrds = []
    for s in sux.sites:
        scrds.append(np.array([s.coords[0], s.coords[1], s.coords[2] + move_dist]))

    sux_move = Structure(lattice=[tmp.lattice.matrix[0], tmp.lattice.matrix[1], 2*tmp.lattice.matrix[2]],
                         species=tmp.species, coords=scrds, coords_are_cartesian=True, to_unit_cell=False)

    derived_structures.append(sux_move)

    # sux_move.to(filename='test.vasp', fmt='POSCAR')

all_structures.extend(derived_structures)

print('Total structures: {}'.format(len(all_structures)))

lammps_input = ['newton on', 'boundary p p p', 'units metal', 'atom_style atomic', 'read_data atom.pos',
                'mass 1 106.42', 'mass 2 107.8682', 'pair_style quip',
                'pair_coeff * * AgPd.xml "Potential xml_label=GAP_2021_2_23_120_20_30_23_270" 46 47',
                'thermo_style custom time pe ke temp', 'thermo 1', 'velocity all zero linear',
                'min_style cg', 'minimize 1e-10 1e-12 10000 100000',
                'write_dump all atom final_positions.atom']


lpad = LaunchPad(host='195.148.22.179', port=27017, name='agpd_fw', username='jank', password='mongo', ssl=False)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 16, 'ISMEAR': 0, 'ISYM': 0, 'ISPIN': 2,
             'ALGO': 'Normal', 'AMIN': 0.01, 'NELM': 200, 'LAECHG': '.FALSE.',
             'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.'}

base_dir = os.getcwd()
for si, structure in enumerate(all_structures):
    if os.path.isdir(str(si)):
        shutil.rmtree(str(si))
    os.mkdir(str(si))
    os.chdir(str(si))

    print('Running: {}'.format(structure.composition.element_composition))

    for file in os.listdir(base_dir):
        if file.startswith('AgPd') or file.startswith('compre'):
            os.symlink(os.path.join(base_dir, file), file)

    with open('atom.pos', 'w') as f:
        write_lammps_data(f, atoms=AseAtomsAdaptor().get_atoms(structure=structure), units='metal',
                          specorder=['Pd', 'Ag'])

    with open('lammps.in', 'w') as f:
        for line in lammps_input:
            f.write(line + '\n')

    os.system('nice -n 15 mpirun -n 2 lmp -in lammps.in')

    with open('final_positions.atom', 'r') as f:
        relaxed = AseAtomsAdaptor().get_structure(atoms=read_lammps_dump_text(f))

    spec_map = {'H': 'Pd', 'He': 'Ag'}
    relaxed.replace_species(species_mapping=spec_map)
    relaxed.to(filename='final.vasp', fmt='POSCAR')

    site_properties = {'initial_moment': []}
    for s in relaxed.sites:
        if s.specie.name == 'Ag':
            site_properties['initial_moment'].append(1.0)
        else:
            site_properties['initial_moment'].append(-1.0)

    relaxed.site_properties['initial_moment'] = site_properties['initial_moment']

    relaxed = Structure(lattice=relaxed.lattice, species=relaxed.species, coords=relaxed.frac_coords,
                        coords_are_cartesian=False, site_properties=site_properties)

    incar_set = MPStaticSet(relaxed)
    structure_name = re.sub(' ', '', str(relaxed.composition.element_composition)) + \
                     ' ' + str(relaxed.num_sites) + ' in man_gen'

    meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
    kpt_set = Kpoints.automatic_gamma_density(structure=relaxed, kppa=1000).as_dict()

    static_wf = get_static_wf(in_structure=relaxed, struc_name=structure_name, vasp_input_set=incar_set,
                              vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                              user_kpoints_settings=kpt_set, metadata=meta)

    run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)

    os.chdir(base_dir)
