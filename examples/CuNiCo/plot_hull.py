from casm.project import Project, Selection, write_eci
from casm.learn import open_halloffame, open_input, checkhull, to_json
import pandas
import matplotlib.pyplot as plt


proj = Project()
sel = Selection(proj, 'CALCULATED', all=False)

comp = 'comp(a)'
Ef = 'formation_energy'
hull_dist = 'hull_dist(CALCULATED,atom_frac)'
sel.query([comp, Ef, hull_dist])

df = sel.data.sort_values([comp])
hull_tol = 1e-6
df_hull = df[df[hull_dist] < hull_tol]

casm_learn_input = 'ga.fit'
selection = 'train'

fit_input = open_input(casm_learn_input)

hall = open_halloffame(fit_input["halloffame_filename"])

# select ECI to use, get convex hull configurations and add pandas.DataFrame
# attributes to hall[indiv_i]:
#
#    "dft_gs" : DFT calculated ground states
#    "clex_gs" : predicted ground states
#    "gs_missing" : DFT ground states that are not predicted ground states
#    "gs_spurious" : Predicted ground states that are not DFT ground states
#    "uncalculated" : Predicted ground states and near ground states that have not been calculated
#    "below_hull" : All configurations predicted below the prediction of the DFT hull
#    "ranged_rms": root-mean-square error calculated for the subset of configurations
#      whose DFT formation energy lies within some range of the DFT convex hull.
#      Currently calculated for ranges 0.001, 0.005, 0.01, 0.05, 0.1, 0.5 eV/unit cell
#

dft_only = True
for indiv_i in range(1):
    checkhull(fit_input, hall, indices=[indiv_i])

    indiv = hall[indiv_i]
    write_eci(proj, hall[0].eci, to_json(indiv_i, hall[indiv_i]))

    sel = Selection(proj, selection, all=False)

    comp_a = 'comp(a)'
    is_calculated = 'is_calculated'
    dft_Ef = 'formation_energy'
    clex_Ef = 'clex(formation_energy)'
    sel.query([comp_a, is_calculated, dft_Ef, clex_Ef])

    df = sel.data

    df_calc = df[df.loc[:, is_calculated] == 1].apply(pandas.to_numeric, errors='ignore')

    # plt.rc('text', usetex=True)
    # plt.rc('font', family="sans-serif", serif="Palatino")
    # plt.rcParams['text.latex.preamble'] = [
    #        r'\usepackage{amssymb}'
    #        r'\usepackage{siunitx}',
    #        r'\sisetup{detect-all}',
    #        r'\usepackage[cm]{sfmath}']

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = 'cm'
    # plt.rcParams['xtick.major.size'] = 8
    # plt.rcParams['xtick.major.width'] = 3
    # plt.rcParams['xtick.minor.size'] = 4
    # plt.rcParams['xtick.minor.width'] = 3
    # plt.rcParams['xtick.labelsize'] = 18
    # plt.rcParams['ytick.major.size'] = 8
    # plt.rcParams['ytick.major.width'] = 3
    # plt.rcParams['ytick.minor.size'] = 4
    # plt.rcParams['ytick.minor.width'] = 3
    # plt.rcParams['ytick.labelsize'] = 18
    # plt.rcParams['axes.linewidth'] = 3

    plt.xlabel(r'Parametric $\mu$', fontsize=14)
    plt.ylabel(r'E$_f$ (eV/prim)', fontsize=14)

    # to plot dft formation energy & hull
    plt.scatter(df_calc[comp_a], df_calc[dft_Ef], facecolors='none', edgecolors='b', label=r'dft')
    plt.plot(df_hull[comp], df_hull[Ef], 'b.-')
    # plt.plot(indiv.dft_gs[comp_a], indiv.dft_gs[dft_Ef], 'bo-', label='_nolegend_')

    if dft_only:
        plt.xlim([-0.02, 1.02])
        plt.legend(loc='best')
        plt.tight_layout()
        fname = 'dft_input.png'
        plt.savefig(fname, dpi=170)
        dft_only = False
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.sans-serif'] = 'cm'
        plt.xlabel(r'Parametric $\mu$', fontsize=14)
        plt.ylabel(r'E$_f$ (eV/prim)', fontsize=14)

    # to plot calculated configurations predicted Ef & predicted hull
    plt.scatter(df_calc[comp_a], df_calc[clex_Ef], color='r', marker='.', label=r'clex')
    plt.plot(indiv.clex_gs[comp_a], indiv.clex_gs[clex_Ef], 'r.-', label='_nolegend_')

    plt.xlim([-0.02, 1.02])

    plt.legend(loc='best')
    plt.tight_layout()
    fname = 'hull_' + str(indiv_i) + '.png'
    plt.savefig(fname, dpi=170)
    plt.close()
