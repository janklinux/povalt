import os
import numpy as np
from ase.io import read, write


vasp_cij = []
parse = False
with open('../vasp/OUTCAR', 'r') as f:
    content = f.readlines()

for il, line in enumerate(content):
    if 'TOTAL ELASTIC MODULI (kBar)' in line:
        for ik in range(3, 9):
            vasp_cij.append([np.round(float(x), 2) for x in content[il+ik].split()[1:]])


quip_cij = []
atoms = read('../vasp/POSCAR')
write(filename='input.xyz', images=atoms, format='xyz')
os.system('quip atoms_filename=input.xyz param_filename=platinum.xml cij > quip.result')
with open('quip.result', 'r') as f:
    for line in f:
        if line.startswith('CIJ'):
            quip_cij.append([np.round(10 * float(x), 2) for x in line.split()[1:]])

for ev, eq in zip(np.abs(vasp_cij), np.abs(quip_cij)):
    print('{:4.2f}  {:4.2f}  {:4.2f}  {:4.2f}  {:4.2f}  {:4.2f}'.format(np.abs(ev[0]-eq[0]), np.abs(ev[1]-eq[1]),
                                                                        np.abs(ev[2]-eq[2]), np.abs(ev[3]-eq[3]),
                                                                        np.abs(ev[4]-eq[4]), np.abs(ev[5]-eq[5])))
