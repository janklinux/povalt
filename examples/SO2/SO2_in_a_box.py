import os
import time
import shutil
import numpy as np


def write_geo(in_lat, in_geo, in_name):
    with open('geometry.in', 'w') as f:
        for il in in_lat:
            f.write('lattice_vector {:6.6f} {:6.6f} {:6.6f}\n'.format(il[0], il[1], il[2]))
        for ig, cg in enumerate(in_geo):
            f.write('atom {:6.6f} {:6.6f} {:6.6f} {}\n'.format(cg[0], cg[1], cg[2], in_name[ig]))


so2_geo = [[0.0, 0.0, 0.0],
           [-1.26498443, -0.74450652, 0.0],
           [1.26498444, -0.74450651, 0.0]]
so2_name = ['S', 'O', 'O']

# 1 g/cm^3 * 1/18 mol/g * 6e23 molecules/mol * 1/(10^8)^3 cm^3/AA^3 for water
# 0.0026288 g/cm^3 * 1/64 mol/g * 6e23 molecules/mol * 1/(10^8)^3 cm^3/AA^3 for SO2 Gas
# 1.4611 g/cm^3 * 1/64 mol/g * 6e23 molecules/mol * 1/(10^8)^3 cm^3/AA^3 for SO2 Liquid
# 0.0026288 g/cm^3 * 1/64 mol/g * 6e23 molecules/mol * 1/(10^8)^3 cm^3/AA^3 for SO2 Gas

so2_per_a3_gas = 0.0026288 * 1.0 / 18.0 * 6.022E23 * 1.0 / 1.0E8**3
so2_per_a3_liq = 1.4661 * 1.0 / 18.0 * 6.022E23 * 1.0 / 1.0E8**3
so2_per_a3_sol = 2 * 1.0 / 18.0 * 6.022E23 * 1.0 / 1.0E8**3

box_lens = np.linspace(start=so2_per_a3_gas, stop=so2_per_a3_liq, num=30, endpoint=True)
np.append(box_lens, np.linspace(start=so2_per_a3_gas, stop=so2_per_a3_liq, num=20, endpoint=True))

base_dir = os.getcwd()
for ib, blen in enumerate(box_lens):
    os.chdir(base_dir)
    if os.path.isdir(str(ib)):
        shutil.rmtree(str(ib))
    os.mkdir(str(ib))
    os.chdir(str(ib))

    a0 = np.cbrt(12 / blen)
    lat = [[a0, 0, 0], [0, a0, 0], [0, 0, a0]]

    print(a0)

    np.random.seed(int(time.time()))

    new_geo = []
    w_name = []

    for i in range(2):
        for j in range(3):
            for k in range(2):
                ra = np.random.random([3])*2*np.pi
                rpos = np.random.random([3]) / 2
                rotx = [[1, 0, 0], [0, np.cos(ra[0]), -np.sin(ra[0])], [0, np.sin(ra[0]), np.cos(ra[0])]]
                roty = [[np.cos(ra[1]), 0, np.sin(ra[1])], [0, 1, 0], [-np.sin(ra[1]), 0, np.cos(ra[1])]]
                rotz = [[np.cos(ra[2]), -np.sin(ra[2]), 0], [np.sin(ra[2]), np.cos(ra[2]), 0], [0, 0, 1]]

                tmp = []
                for v in so2_geo:
                    tmp.append(np.dot(rotz, np.dot(roty, np.dot(rotx, v))))

                tmp[0][0] += a0 / 2 * i + rpos[0]
                tmp[1][0] += a0 / 2 * i + rpos[0]
                tmp[2][0] += a0 / 2 * i + rpos[0]

                tmp[0][1] += a0 / 3 * j + rpos[1]
                tmp[1][1] += a0 / 3 * j + rpos[1]
                tmp[2][1] += a0 / 3 * j + rpos[1]

                tmp[0][2] += a0 / 2 * k + rpos[2] * 4
                tmp[1][2] += a0 / 2 * k + rpos[2] * 4
                tmp[2][2] += a0 / 2 * k + rpos[2] * 4

                new_geo.append(tmp)

                for n in range(3):
                    w_name.append(so2_name[n])

    wgeo = []
    for g in new_geo:
        for lg in g:
            wgeo.append(list(lg))

    write_geo(in_lat=lat, in_geo=wgeo, in_name=w_name)
