import matplotlib.pyplot as plt


cores = [4, 8, 16, 32, 64, 128]
times = []

for c in cores:
    with open(str(c)+'/OUTCAR', 'r') as f:
        for line in f:
            if 'Total CPU time used (sec):' in line:
                times.append(float(line.split()[5]))

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', serif='Palatino')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3

plt.plot(cores, times, '.-', linewidth=2, color='navy')

plt.xlabel(r'Number of CPU cores', fontsize=16, color='k')
plt.ylabel(r'Total runtime [sec]', fontsize=16, color='k')

plt.xticks(cores, cores)
# plt.legend(loc='upper right', fontsize=8)
# plt.xlim(-7, 1)
# plt.ylim(-7, 1)

plt.tight_layout()
plt.savefig('GC_scaling.png', dpi=600)
plt.close()
