from povalt.generators.generators import Dimer

d = Dimer(species=['Ni', 'Al'], lattice=None, min_dist=0.5, max_dist=10.0, n_points=35)

d.run_dimer_aims()
