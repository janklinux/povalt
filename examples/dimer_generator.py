from povalt.generators.generators import Dimer

# d = Dimer(species=['Cu', 'Pt', 'Ir'], lattice=None, min_dist=0.3, max_dist=8.0, show_curve=True)

d = Dimer(species=['Pt'], lattice=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 15.0]],
          min_dist=0.7, max_dist=8.0, show_curve=True)

d.run_dimer_vasp()