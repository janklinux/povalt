from povalt.generators.generators import Dimer

d = Dimer(species=['Cu', 'Pt', 'Ir'], lattice=None, min_dist=0.5, max_dist=7.0, n_points=25)

d.run_dimer_aims()
