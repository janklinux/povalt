import sys
import json
import numpy as np
import tensorflow as tf
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from quippy.convert import ase_to_quip
from quippy.descriptors import Descriptor


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, enable=True)

with open('../all_clusters.json', 'r') as f:
    all_clusters = json.load(f)

all_q = []
print('working on group: ', end='')
sys.stdout.flush()
for c in all_clusters:
    print('{} '.format(int(c)), end='')
    sys.stdout.flush()
    for s in all_clusters[c]['structures']:
        quip_atoms = ase_to_quip(AseAtomsAdaptor().get_atoms(Structure.from_dict(s)))
        desc = Descriptor('soap_turbo l_max=8 alpha_max={8} atom_sigma_r={0.5} atom_sigma_t={0.5} '
                          'atom_sigma_r_scaling={0.0} atom_sigma_t_scaling={0.0} zeta=6 rcut_hard=4.7 '
                          'rcut_soft=4.2 basis="poly3gauss" scaling_mode="polynomial" amplitude_scaling={1.0} '
                          'n_species=1 species_Z=78 radial_enhancement={1} central_weight={1.0} delta=0.1 '
                          'f0=0.0 covariance_type=dot_product sparse_method=cur_points')
        q = desc.calc_descriptor(quip_atoms)
        all_q.append(tf.constant(np.array(q), dtype=tf.float16, shape=q.shape))
    if int(c) > 149:
        break
print('')
sys.stdout.flush()

correlation_limit = 0.65
correlated_list = []
low_bound = 0.85
print('Number of descriptor matrices: {}, number of matrix.matrix operations: {}'.format(len(all_q), len(all_q)**2))
print('Starting GPU comparison of all matrices, working... ', end='')
sys.stdout.flush()
with tf.device('GPU:0'):
    for i, p in enumerate(all_q):
        if i % int(len(all_q)/10) == 0:
            print('{:2.2f}% '.format(i/len(all_q)*100), end='')
            sys.stdout.flush()
        for j, q in enumerate(all_q):
            matrix = tf.matmul(p, tf.transpose(q))
            if np.any(matrix < correlation_limit):
                correlated_list.append([i, j])
            if np.any(matrix < low_bound):
                # print('old: {} -- new: {}'.format(low_bound, tf.reduce_min(matrix)))
                low_bound = tf.reduce_min(matrix)
                # if np.linalg.norm(matrix) < p.shape[0]/2:
                # print('Norm: {:3.3f} -- Shapes  p: {} | q: {}'.format(np.linalg.norm(matrix), p.shape, q.shape))
                # print('I: {}, J: {}'.format(i, j))
                # for row in matrix:
                #     for el in row:
                #         print('{:3.3f}  '.format(np.round(el, 2)), end='')
                #     print('')
                # print('  =========================')
                # sys.stdout.flush()
print('')
print('Overall absolute min: {}'.format(low_bound))

best_list = []
for p, q in correlated_list:
    if p not in best_list:
        best_list.append(p)
    if q not in best_list:
        best_list.append(q)

print('best list len: {}'.format(len(best_list)))

with open('best_list.json', 'w') as f:
    json.dump(obj=best_list, fp=f)
