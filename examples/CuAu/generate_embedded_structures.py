import re
import os
import time
import json
import shutil
import random
import datetime
import subprocess
import numpy as np
from mpi4py import MPI
from fireworks import LaunchPad, Workflow
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure, Lattice
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate.vasp.powerups import add_modify_incar
from atomate.vasp.fireworks.core import StaticFW


def get_epsilon(in_lattice):
    eps = np.array([float(0.2*(x-0.5)) for x in np.random.random(6)])
    cij = np.array([[1+eps[0], eps[5]/2, eps[4]/2],
                    [eps[5]/2, 1+eps[1], eps[3]/2],
                    [eps[4]/2, eps[3]/2, 1+eps[2]]])
    return Lattice(np.transpose(np.dot(cij, np.transpose(in_lattice)))), Lattice(in_lattice)


def is_frozen_job(jobdir):
    most_recent = None
    for ff in os.listdir(jobdir):
        tt = time.time() - os.path.getmtime(os.path.join(jobdir, ff))
        if most_recent is None:
            most_recent = tt
        elif tt < most_recent:
            most_recent = tt
    return False if most_recent < 600 else True


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
                    db_file='', name="{} -- static".format(tag))]
    wfname = "{}: {}".format(struc_name, name)
    return Workflow(fws, name=wfname, metadata=metadata)


mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

random.seed(int(time.time()))

show_ball = False
show_cyl = False

species = ['Cu', 'Au']
systems = ['bcc', 'fcc', 'sc', 'hcp']

run_mode = 'gen_embedded'  # gen_embedded, gen_liquid

lammps_mode = 'relax'  # MD, relax

run_name = '4th_{}_{}'.format(run_mode.split('_')[1], lammps_mode)

do_elastic = True


lpad = LaunchPad(host='195.148.22.179', port=27017, name='test_fw', username='jank', password='mongo', ssl=False)

incar_mod = {'EDIFF': 1E-5, 'ENCUT': 520, 'NCORE': 16, 'ISMEAR': 0,
             'ISYM': 0, 'ISPIN': 2, 'NELM': 150, 'ALGO': 'Normal',
             'LAECHG': '.FALSE.', 'LCHARG': '.FALSE.', 'LVTOT': '.FALSE.', 'LVHAR': '.FALSE.'}


if run_mode not in ['gen_embedded', 'gen_liquid']:
    raise SyntaxError('Error in run_mode')


if run_mode == 'gen_embedded':
    if mpi_rank == 0:
        all_structures = []
        usable_structures = []
        for csys in systems:
            for spec in species:
                prim = Structure.from_file('_'.join(['POSCAR', csys, spec]))

                scale = {'bcc': np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
                         'fcc': np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
                         'hcp': np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
                         'sc':  np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])}

                cell = SupercellTransformation(scaling_matrix=scale[csys]).apply_transformation(prim)

                print(csys, spec, cell.num_sites)

                center = np.zeros(3)
                for lv in cell.lattice.matrix:
                    center += lv / 2

                if spec == 'Cu':
                    replace_species = 'Au'
                else:
                    replace_species = 'Cu'

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

                        if show_ball:
                            cell.to(filename='{}_ball_embedded_in_{}_{}_{}.vasp'
                                    .format(replace_species, csys, spec, prev_num),
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

                    if show_cyl:
                        new_cell.to(filename='{}_cyl_embedded_in_{}_{}_{}.vasp'
                                    .format(replace_species, csys, spec, atnum),
                                    fmt='POSCAR')

        omits = 0
        derived_structures = []

        for structure in all_structures:
            for lat_mul in [2, 3]:
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
                sux = Structure(lattice=[tmp.lattice.matrix[0], tmp.lattice.matrix[1], lat_mul * tmp.lattice.matrix[2]],
                                species=tmp.species, coords=tmp.cart_coords,
                                coords_are_cartesian=True, to_unit_cell=False)

                helparray = []
                for c in sux.cart_coords:
                    helparray.append(c[2])

                middle = np.amax(helparray) - np.amin(helparray)
                move_dist = sux.lattice.matrix[2][2] / 2 - middle

                scrds = []
                for s in sux.sites:
                    scrds.append(np.array([s.coords[0], s.coords[1], s.coords[2] + move_dist]))

                sux_move = Structure(lattice=[tmp.lattice.matrix[0],
                                              tmp.lattice.matrix[1],
                                              lat_mul * tmp.lattice.matrix[2]],
                                     species=tmp.species, coords=scrds, coords_are_cartesian=True, to_unit_cell=False)

                derived_structures.append(sux_move)

                # if lat_mul == 3:
                #     print('make additions array to index the dipole corrected ones here...')
                #     quit()

                # sux_move.to(filename='test.vasp', fmt='POSCAR')

        all_structures.extend(derived_structures)

        hkl_interfaces = []
        hkl_list = []
        for ih in range(0, 4):
            for ik in range(0, 4):
                for il in range(0, 4):
                    hkl = np.array([ih, ik, il])
                    if np.sum(hkl) <= 1:
                        continue
                    else:
                        hkl_list.append(hkl)

        print('Number of hkl planes: {}'.format(len(hkl_list)))

        for csys in systems:
            s = Structure.from_file('POSCAR_' + csys + '_Au')

            tmp_crds = []
            for ii in range(-20, 20):
                for jj in range(-20, 20):
                    for kk in range(-20, 20):
                        for c in s.cart_coords:
                            tmp_crds.append(c + np.dot(np.array([ii, jj, kk]), s.lattice.matrix))

            for hkl in hkl_list:
                plane = np.zeros((3, 3))
                plane[0] = hkl
                plane[1] = np.array([-1, 0, hkl[0]/hkl[2]])
                plane[2] = np.cross(plane[0], plane[1])

                theta = np.zeros(3)

                for ia, (v, w) in enumerate(zip(plane, s.lattice.matrix)):
                    theta[ia] = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))

                rotx = [[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])], [0, np.sin(theta[0]), np.cos(theta[0])]]
                roty = [[np.cos(theta[1]), 0, np.sin(theta[1])], [0, 1, 0], [-np.sin(theta[1]), 0, np.cos(theta[1])]]
                rotz = [[np.cos(theta[2]), -np.sin(theta[2]), 0], [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]]

                rotation = np.dot(rotx, np.dot(roty, rotz))

                if csys == 'sc':
                    target = np.dot(rotation, np.dot(s.lattice.matrix, np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])))
                else:
                    target = np.dot(rotation, np.dot(s.lattice.matrix, np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])))

                inv_target = np.linalg.inv(target)

                new_crds = []
                new_specs = []
                for c in tmp_crds:
                    tmp = np.dot(c, inv_target)
                    if 0 <= tmp[0] < 1 and 0 <= tmp[1] < 1 and 0 <= tmp[2] < 1:
                        new_crds.append(c)

                news = Structure(lattice=target, species=['Au' for _ in range(len(new_crds))],
                                 coords=new_crds, coords_are_cartesian=True)

                remove_list = []
                for ii in range(len(news.cart_coords)):
                    for jj in range(ii + 1, len(news.cart_coords)):
                        if news.get_distance(ii, jj) < 1:
                            remove_list.append(jj)
                news.remove_sites(remove_list)

                for ic, c in enumerate(news.frac_coords):
                    if c[1] < 0.5:
                        news.replace(ic, 'Cu')
                news.sort()

                if news not in hkl_interfaces:
                    if 10 >= news.num_sites <= 120:
                        continue
                    hkl_interfaces.append(news)
                    print('csys: {}  -  hkl: {}  - natoms: {}'.format(csys, hkl, news.num_sites))

        all_structures.extend(hkl_interfaces)
        for structure in all_structures:
            if not len(structure.composition.elements) == 2:
                omits += 1
                continue
            else:
                usable_structures.append(structure.as_dict())

        with open('all_structures.json', 'w') as file:
            json.dump(obj=usable_structures, fp=file)

        print('\nTotal structures written: {}  -- omitted: {}\n'.format(len(usable_structures), omits))


elif run_mode == 'gen_liquid':
    if mpi_rank == 0:
        omits = 0
        all_structures = []
        usable_structures = []
        for csys in systems:
            for spec in species:
                prim = Structure.from_file('_'.join(['POSCAR', csys, spec]))

                scale = {'bcc': np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
                         'fcc': np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
                         'hcp': np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
                         'sc':  np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])}

                print(csys, spec)

                if spec == 'Au':
                    replace_species = 'Cu'
                else:
                    replace_species = 'Au'

                for _ in range(10):
                    cell = SupercellTransformation(scaling_matrix=scale[csys]).apply_transformation(prim)
                    for idx in random.sample(range(len(cell.sites)), random.randint(1, int(len(cell.sites)*0.95))):
                        cell.replace(idx, replace_species)
                    cell.sort()
                    all_structures.append(cell)

        for structure in all_structures:
            if len(structure.composition.elements) == 2:
                usable_structures.append(structure.as_dict())

        with open('all_structures.json', 'w') as file:
            json.dump(obj=usable_structures, fp=file)

        print('\nTotal structures written: {}  -- omitted: {}\n'.format(len(usable_structures), omits))


mpi_comm.barrier()
with open('all_structures.json') as file:
    usable_structures = json.load(fp=file)

local_index = []
local_structures = []
for si, st in enumerate(usable_structures):
    if si % mpi_size == mpi_rank:
        local_index.append(si)
        local_structures.append(Structure.from_dict(st))
del usable_structures

pot_dir = '/home/jank/work/Aalto/GAP_data/CuAu/validation'
gap_label = [str(s) for s in os.listdir(pot_dir) if s.startswith('CuAu.xml.sparseX')][0][17:-1]

if lammps_mode == 'relax':
    lammps_input = ['newton on', 'boundary p p p', 'units metal', 'atom_style atomic', 'read_data atom.pos',
                    'mass 1 63.546', 'mass 2 196.967', 'pair_style quip', 'neigh_modify every 1 delay 0 check yes',
                    'pair_coeff * * CuAu.xml "Potential xml_label={}" 29 79'.format(gap_label),
                    'thermo_style custom time pe ke temp', 'thermo 1', 'velocity all zero linear',
                    'min_style cg', 'minimize 1e-10 1e-12 10000 100000',
                    'write_dump all atom final_positions.atom']

elif lammps_mode == 'MD':
    if mpi_size >= 10:
        temp_range = np.linspace(start=500, stop=5000, num=mpi_size, endpoint=True)
        press_range = np.linspace(start=1, stop=20000, num=mpi_size, endpoint=True)
        init_temp = temp_range[mpi_rank]
        run_temp = init_temp
        init_press = press_range[mpi_rank]
        run_press = init_press
    else:
        init_temp = (mpi_rank+1) * 500
        run_temp = init_temp
        init_press = 3**mpi_rank
        run_press = init_press

    lammps_input = ['newton on', 'boundary p p p', 'units metal', 'atom_style atomic', 'read_data atom.pos',
                    'mass 1 63.546', 'mass 2 196.967', 'pair_style quip', 'neigh_modify every 1 delay 0 check yes',
                    'pair_coeff * * CuAu.xml "Potential xml_label={}" 29 79'.format(gap_label),
                    'thermo_style custom time pe ke temp press', 'thermo 10', 'velocity all zero linear',
                    'timestep 0.002',
                    'fix 1 all nvt temp {} {} 0.1'.format(init_temp, run_temp),
                    'run 100',
                    'unfix 1',
                    'fix 1 all nvt temp {} {} 0.1'.format(init_temp, run_temp),
                    'fix 2 all press/berendsen iso {} {} 1.0 modulus 2.5e6'.format(init_press, run_press),
                    'run 1000',
                    'write_dump all atom final_positions.atom']

else:
    raise NotImplementedError('Invalid lammps_mode')

total_killed = 0
base_dir = os.path.join(os.getcwd(), run_name)
for si, structure in zip(local_index, local_structures):
    os.chdir(base_dir)
    if os.path.isdir(str(si)):
        shutil.rmtree(str(si))
    os.mkdir(str(si))
    os.chdir(str(si))

    print('CPU{} running job {}: {}'.format(mpi_rank, si, structure.composition.element_composition))

    for file in os.listdir(pot_dir):
        if file.startswith('CuAu') or file.startswith('compre'):
            os.symlink(os.path.join(pot_dir, file), file)
    structure.to(filename='initial.vasp', fmt='POSCAR')
    with open('atom.pos', 'w') as f:
        write_lammps_data(f, atoms=AseAtomsAdaptor().get_atoms(structure=structure), units='metal',
                          specorder=['Cu', 'Au'])
    with open('lammps.in', 'w') as f:
        for line in lammps_input:
            f.write(line + '\n')

    command = 'lmp -in lammps.in'  # -k on t 1 g 1 -sf kk'

    gave_up = False
    rerun = 0
    successful_lammps = False
    while not successful_lammps:
        sout = open('/dev/zero', 'w')
        serr = open('std_err', 'w')
        p = subprocess.Popen(command.split(), stdout=sout, stderr=serr)

        err = None
        job_killed = False
        last_check = time.time()
        while p.poll() is None:
            time.sleep(1)
            if time.time() - last_check > 30:
                last_check = time.time()
                if is_frozen_job('.'):
                    p.kill()
                    print('CPU{} KILLED job {}'.format(mpi_rank, si))
                    job_killed = True

        sout.close()
        serr.close()

        if job_killed:
            total_killed += 1

        if os.path.isfile('final_positions.atom'):
            successful_lammps = True
        else:
            rerun += 1
            if rerun == 3:
                print('giving up on job {}'.format(si))
                gave_up = True
                break
            print('rerunning job {}'.format(si))

    if gave_up:
        continue

    with open('final_positions.atom', 'r') as f:
        relaxed = AseAtomsAdaptor().get_structure(atoms=read_lammps_dump_text(f))
    spec_map = {'H': 'Cu', 'He': 'Au'}
    relaxed.replace_species(species_mapping=spec_map)
    relaxed.to(filename='final.vasp', fmt='POSCAR')
    site_properties = {'initial_moment': []}
    for s in relaxed.sites:
        if s.specie.name == 'Cu':
            site_properties['initial_moment'].append(1.0)
        else:
            site_properties['initial_moment'].append(-1.0)
    relaxed = Structure(lattice=relaxed.lattice, species=relaxed.species, coords=relaxed.frac_coords,
                        coords_are_cartesian=False, site_properties=site_properties)
    incar_set = MPStaticSet(relaxed)
    structure_name = re.sub(' ', '', str(relaxed.composition.element_composition)) + ' '
    structure_name += str(relaxed.num_sites) + ' lammps_{} {}'.format(run_name, run_mode.split('_')[1])
    meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
    kpt_set = Kpoints.automatic_density(structure=relaxed, kppa=1000).as_dict()
    static_wf = get_static_wf(in_structure=relaxed, struc_name=structure_name, vasp_input_set=incar_set,
                              vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                              user_kpoints_settings=kpt_set, metadata=meta)
    run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
    lpad.add_wf(run_wf)

    if do_elastic:
        relaxed.lattice, orig_lattice = get_epsilon(in_lattice=relaxed.lattice.matrix)
        incar_set = MPStaticSet(relaxed)
        structure_name = re.sub(' ', '', str(relaxed.composition.element_composition)) + ' '
        structure_name += str(relaxed.num_sites) + ' lammps_{} epsilon'.format(run_name)
        meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
        kpt_set = Kpoints.automatic_density(structure=relaxed, kppa=1000).as_dict()
        static_wf = get_static_wf(in_structure=relaxed, struc_name=structure_name, vasp_input_set=incar_set,
                                  vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                                  user_kpoints_settings=kpt_set, metadata=meta)
        run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
        lpad.add_wf(run_wf)

        for displace in np.linspace(start=-0.1, stop=0.1, num=20, endpoint=True):
            relaxed.lattice = Lattice(np.dot(displace, np.identity(3)) + orig_lattice.matrix)
            incar_set = MPStaticSet(relaxed)
            structure_name = re.sub(' ', '', str(relaxed.composition.element_composition)) + ' '
            structure_name += str(relaxed.num_sites) + ' lammps_{} isotropic'.format(run_name)
            meta = {'name': structure_name, 'date': datetime.datetime.now().strftime('%Y/%m/%d-%T')}
            kpt_set = Kpoints.automatic_density(structure=relaxed, kppa=1000).as_dict()
            static_wf = get_static_wf(in_structure=relaxed, struc_name=structure_name, vasp_input_set=incar_set,
                                      vasp_cmd='srun --nodes=1 --ntasks=128 --ntasks-per-node=128 vasp_std',
                                      user_kpoints_settings=kpt_set, metadata=meta)
            run_wf = add_modify_incar(static_wf, modify_incar_params={'incar_update': incar_mod})
            lpad.add_wf(run_wf)

MPI.Finalize()
