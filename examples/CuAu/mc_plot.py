import os
import json
import matplotlib.pyplot as plt


data = dict()
cols = ['<comp(a)>', 'param_chem_pot(a)', '<potential_energy>', '<atom_frac(Au)>', '<atom_frac(Cu)>']
with open('results.json', 'r') as f:
    res = json.load(f)

plt.plot(res['param_chem_pot(a)'], res['<atom_frac(Au)>'], label=r'frac(Au)')
plt.plot(res['param_chem_pot(a)'], res['<atom_frac(Cu)>'], label=r'frac(Cu)')

plt.title(r'AuCu in {}'.format(os.getcwd().split('/')[-2]))
plt.xlabel(r'parametric chem pot')
plt.ylabel(r'comp(a)')

plt.ylim(-0.1, 1.1)

plt.legend(loc='center right')
plt.savefig('res.png', dpi=300)

quit()

data['mc'] = []
data['mc'].append(res[cols])

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'

print(data)
