from povalt.generators.generators import Dimer

# d = Dimer(species=['Ir', 'O'], lattice=[[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]],
#           min_dist=0.7, max_dist=8.0, show_curve=True, cores=6)
# d.run_dimer_vasp()

d = Dimer(species=['Al', 'Ni'], lattice=None, min_dist=0.25, max_dist=6.5,
          show_spin_curves=False, show_result=True, cores=2)
d.run_dimer_aims()
