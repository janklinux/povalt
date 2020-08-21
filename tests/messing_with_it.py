from ase.io import read
from ase.io.lammpsrun import read_lammps_dump_text
from povalt.helpers import remove_residual_files
from povalt.lammps.lammps import Lammps
from povalt.training.training import TrainPotential
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure, Element


train_pot = False
relax = True

remove_residual_files('.')

if train_pot:
    print('training...')
    tr_pot = TrainPotential(atoms_filename='complete.xyz', order=2, compact_clusters='T', nb_cutoff=5.0, n_sparse=20,
                            nb_covariance_type='ard_se', nb_delta=0.5, theta_uniform=1.0, nb_sparse_method='uniform',
                            l_max=8, n_max=8, atom_sigma=0.5, zeta=4, soap_cutoff=5.0, central_weight=1.0,
                            config_type_n_sparse='{fcc:500:bcc:500:hcp:500:sc:500}', soap_delta=0.1, f0=0.0,
                            soap_covariance_type='dot_product', soap_sparse_method='cur_points',
                            default_sigma='{0.002 0.2 0.2 0.2}',
                            config_type_sigma='{dimer:0.002:0.2:0.2:0.2:fcc:0.002:0.2:0.2:0.2:bcc:0.002:0.2:0.2:0.2:'
                                              'hcp:0.002:0.2:0.2:0.2:sc:0.002:0.2:0.2:0.2}',
                            energy_parameter_name='free_energy',
                            force_parameter_name='forces',
                            force_mask_parameter_name='force_mask',
                            sparse_jitter=1E-8, do_copy_at_file='F', sparse_separate_file='T', gp_file='Pt_test.xml')

    pot_base_name = tr_pot.train(mpi_cmd=None, mpi_procs=None, omp_threads=6)
else:
    pot_base_name = 'GAP_2020_8_20_180_12_8_35_224'


print('running...')
if relax:
    #  Relaxation
    lammps_settings = ['units metal', 'newton on', 'dimension 3', 'atom_style atomic', 'boundary p p p',
                       'read_data atom.pos', 'mass * 195.084', 'pair_style quip',
                       'pair_coeff * * Pt_test.xml "Potential xml_label=' + str(pot_base_name) + '" 78',
                       'compute energy all pe',
                       'group bulk type 1',
                       'fix 1 bulk move linear 1.0 1.0 1.0',
                       'min_style cg', 'min_modify dmax 0.001', 'minimize 1.0E-8 1.0E-9 1000 10000',
                       'write_dump all atom final_positions.atom']
else:
    # MD
    lammps_settings = ['variable x index 1', 'variable y index 1', 'variable z index 1', 'variable t index 2000',
                       'newton on',
                       'boundary p p p', 'units metal', 'atom_style atomic', 'read_data atom.pos', 'mass * 195.084',
                       'pair_style quip',
                       'pair_coeff * * Pt_test.xml "Potential xml_label=' + str(pot_base_name) + '" 78',
                       'compute energy all pe',
                       'neighbor 2.0 bin', 'thermo 100', 'timestep 0.001',
                       'fix 1 all npt temp 400 400 0.01 iso 1000.0 1000.0 1.0',
                       'run $t',
                       'write_dump all atom final_positions.atom']

pt_bulk = read('../geo.vasp')
lmp = Lammps(pt_bulk)
lmp.write_md(atoms_file='atom.pos', lammps_settings=lammps_settings, units='metal')
lmp.run(binary='lmp', cmd_params='-k on t 1 g 1 -sf kk',
        output_filename='LOG', mpi_cmd='mpirun', mpi_procs=4)


print('result:')
with open('final_positions.atom', 'r') as f:
    final_atoms = read_lammps_dump_text(fileobj=f, index=-1)

lammps_result = AseAtomsAdaptor.get_structure(final_atoms)
lattice = lammps_result.lattice
species = [Element('Pt') for i in range(len(lammps_result.species))]
coords = lammps_result.frac_coords

rerun_structure = Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=False)

print(rerun_structure)
rerun_structure.to(fmt='POSCAR', filename='geo.vasp')
