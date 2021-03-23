from ase.io import write
from ase.lattice.cubic import Diamond
from ase.visualize import view
from ase.build import add_vacuum, bulk, surface
import numpy as np


counter = 1
for a in [3.8]:
        pt_atoms = bulk('Pt', 'fcc', a = a, cubic=True)
        au_atoms = bulk('Au', 'fcc', a = a, cubic=True)
        slab = surface(pt_atoms, (2, 1, 3), 3)
        slab.center(vacuum=15., axis=2)
        slab2 = surface(au_atoms, (2, 1, 3), 3)
        slab2.center(vacuum=15., axis=2)
        slab2.positions += [a, a, 0]
        step = fuck
        write(filename='test.vasp', images=step, format='vasp')
