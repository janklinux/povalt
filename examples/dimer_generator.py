from povalt.generators.generators import Dimer

d = Dimer(species=['Cu', 'Au'], lattice=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
          min_dist=0.7, max_dist=7.5, show_spin_curves=False, show_result=True, cores=4, mpi_cmd='mpirun')
d.run_dimer_vasp()

# d = Dimer(species=['Al', 'Ni'], lattice=None, min_dist=0.25, max_dist=6.5,
#           show_spin_curves=False, show_result=True, cores=2)
# d.run_dimer_aims()
