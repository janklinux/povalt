import os
import shutil
import warnings
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Kpoints, Vasprun
from pymatgen.io.vasp.inputs import UnknownPotcarWarning


warnings.simplefilter('ignore', category=UnknownPotcarWarning)

write_poscars = True

factors = [2, 3, 4, 5, 6, 7]

# cu_en = Vasprun('vasprun.xml').final_energy
# ni_en = Vasprun('Ni/pure/vasprun.xml').final_energy
# co_en = Vasprun('Co/pure/vasprun.xml').final_energy
#
# ni_cell = []
# co_cell = []
# for f in factors:
#     ni_run = Vasprun('Ni/{}/vasprun.xml'.format(f))
#     co_run = Vasprun('Co/{}/vasprun.xml'.format(f))
#
#     n_ni = int(str(ni_run.final_structure.species).count('Ni'))
#     n_cu = int(str(ni_run.final_structure.species).count('Cu'))
#     n_co = int(str(co_run.final_structure.species).count('Co'))
#
#     # print(ni_run.final_energy)
#     print(n_cu, cu_en, n_ni, ni_en, n_co, co_en)
#
#     print('Nickel: {}: {}'.format(f, float(ni_run.final_energy) - (n_cu*cu_en)-(n_ni*ni_en)))
#     print('Cobalt: {}: {}'.format(f, float(co_run.final_energy) - (n_cu*cu_en)-(n_co*co_en)))
#
#     # co_cell.append(Vasprun('Co/{}/vasprun.xml'.format(f)).final_energy-((cu_en*f**3-1)+co_en))
#
#
# # print(ni_cell)
# # print(co_cell)


if write_poscars:
    for f in factors:
        if os.path.isdir(str(f)):
            shutil.rmtree(str(f))
        os.mkdir(str(f))
        s = Structure.from_file('CONTCAR')
        s.make_supercell([f, f, f])
        # Kpoints.automatic_density(structure=s, kppa=1000).write_file(os.path.join(str(f), 'KPOINTS'))
        s.replace(0, 'Ni')
        s.sort()
        os.chdir(str(f))
        s.to(filename='geometry.in', fmt='aims')
        os.chdir('..')
