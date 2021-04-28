import os
import json
from pymatgen import Structure


with open('all_clusters.json', 'r') as f:
    clusters = json.load(f)

selected = []
for c in clusters:
    for i, d in enumerate(clusters[c]['energies']['dft']):
        if c == str(60):
            selected.append(Structure.from_dict(clusters[c]['structures'][i]))

base_dir = os.getcwd()
for idx, sel in enumerate(selected):
    sel.replace_species({'Pt': 'Cu'})
    site_properties = {'initial_moment': []}
    for s in sel.sites:
        if s.specie.name == 'Cu':
            site_properties['initial_moment'].append(1.0)
        else:
            site_properties['initial_moment'].append(-1.0)

    relaxed = Structure(lattice=sel.lattice, species=sel.species, coords=sel.frac_coords,
                        coords_are_cartesian=False, site_properties=site_properties)

    os.mkdir(os.path.join('clusters', str(idx)))
    os.chdir(os.path.join('clusters', str(idx)))
    relaxed.to(filename='geometry.in', fmt='aims')
    os.chdir(base_dir)
