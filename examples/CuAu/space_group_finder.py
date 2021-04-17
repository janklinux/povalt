import numpy as np
from pymatgen import Structure, Lattice


a = 1.98
c = 3.14

fcc_pos = np.array([0, 0, 0])
hcp_pos = np.array([[0, 0, 0], [1/2, 1/3, 2/3], [1/2, 2/3, 1/3]])

fcc_lat = Lattice([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
bcc_lat = Lattice([[-a/2, a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, -a/2]])
sc_lat = Lattice([[a, 0, 0], [0, a, 0], [0, 0, a]])
hcp_lat = Lattice([[a/2, -a*np.sqrt(3)/2, 0], [a/2, a*np.sqrt(3)/2, 0], [0, 0, c]])

structures = []
structures.append(Structure(lattice=fcc_lat, species=['Pt'], coords=[fcc_pos]))
structures.append(Structure(lattice=bcc_lat, species=['Pt'], coords=[fcc_pos]))
structures.append(Structure(lattice=sc_lat, species=['Pt'], coords=[fcc_pos]))
structures.append(Structure(lattice=hcp_lat, species=['Pt', 'Pt', 'Pt'], coords=hcp_pos))

for structure in structures:
    print(structure.get_space_group_info())
