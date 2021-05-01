import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from datetime import datetime


def parse_quip(filename):
    start_parse = False
    with open(filename, 'r') as ff:
        data = []
        predicted_energies = []
        first_line_parsed = False
        for line in ff:
            if not first_line_parsed:
                start_time = line.split()[3]
                first_line_parsed = True
            if line.startswith('libAtoms::Finalise:'):
                if len(line.split()) == 3:
                    end_time = line.split()[2]
            if line.startswith('Energy='):
                predicted_energies.append(float(line.split('=')[1]))
                if start_parse:
                    data.append(tmp)
                start_parse = True
                tmp = list()
            if start_parse:
                if line.startswith('AT'):
                    tmp.append(line[3:])

    data.append(tmp)

    dt = datetime.strptime(end_time, '%H:%M:%S') - datetime.strptime(start_time, '%H:%M:%S')
    return dict({'data': data, 'predicted_energies': predicted_energies}), dt


def parse_xyz(filename):
    with open(filename, 'r') as ff:
        data = []
        reference_energies = []
        atoms = []
        line_count = 0
        for line in ff:
            line_count += 1
            if line_count == 1:
                n_atoms = int(line)
            if line_count == 2:
                info = line
                bits = line.split()
                for ii, bit in enumerate(bits):
                    if 'free_energy' in bit:
                        reference_energies.append(float(bit.split('=')[1]))
            if 2 < line_count <= n_atoms + 2:
                atoms.append(line)
            if line_count == n_atoms + 2:
                line_count = 0
                tmp = list()
                tmp.append('{}\n'.format(n_atoms))
                tmp.append(info)
                for a in atoms:
                    tmp.append(a)
                data.append(tmp)
                atoms = []
    return dict({'data': data, 'reference_energies': reference_energies})


def get_differences(result, reference):
    result_data = []
    reference_data = []

    for ip, res in enumerate(result['data']):
        coord = []
        virial = []
        forces = []
        n_atoms = 0
        line_count = 0
        for line in res:
            line_count += 1
            if line_count == 1:
                n_atoms = int(line)
            # if line_count == 2:
            #     bits = line.split()
            #     virial.append(float(bits[0].split('=')[1][1:]))
            #     for v in bits[1:8]:
            #         virial.append(float(v))
            #     virial.append(float(bits[8][:-1]))
            if 2 < line_count <= n_atoms + 2:
                coord.append([float(x) for x in line.split()[1:4]])
                forces.append([float(x) for x in line.split()[-3:]])
            if line_count == n_atoms + 2:
                line_count = 0
                tmp = dict()
                tmp['free_energy'] = result['predicted_energies'][ip]
                tmp['virial'] = virial
                tmp['coords'] = coord
                tmp['forces'] = forces
                result_data.append(tmp)
                virial = []
                coord = []
                forces = []

    for ref in reference['data']:
        energy = 0
        coord = []
        virial = []
        forces = []
        config_type = None
        n_atoms = 0
        line_count = 0
        for line in ref:
            line_count += 1
            if line_count == 1:
                n_atoms = int(line)
            if line_count == 2:
                bits = line.split()
                for ii, bit in enumerate(bits):
                    if 'free_energy' in bit:
                        energy = float(bit.split('=')[1])
                    if 'virial' in bit:
                        virial.append(float(bits[ii].split('=')[1][1:]))
                        for v in bits[ii+1:ii+8]:
                            virial.append(float(v))
                        virial.append(float(bits[ii+8][:-1]))
                    if 'config_type' in bit:
                        config_type = bit.split('=')[1]
            if 2 < line_count <= n_atoms + 2:
                coord.append([float(x) for x in line.split()[1:4]])
                forces.append([float(x) for x in line.split()[4:7]])
            if line_count == n_atoms + 2:
                line_count = 0
                tmp = dict()
                tmp['free_energy'] = energy
                tmp['virial'] = virial
                tmp['coords'] = coord
                tmp['forces'] = forces
                tmp['config_type'] = config_type
                reference_data.append(tmp)
                energy = 0
                virial = []
                coord = []
                forces = []

    energy_diff = []
    force_diff = []
    reference_per_atom = []
    prediction_per_atom = []
    system_config = []

    print('parsed sizes: data: {}, ref: {}'.format(len(result_data), len(reference_data)))

    for i in range(len(result_data)):
        for fa, fb in zip(np.array(result_data[i]['coords']), np.array(reference_data[i]['coords'])):
            if not np.all(fa == fb):
                print('not the same: ', fa, fb)
                quit()

        reference_per_atom.append(reference_data[i]['free_energy'] / len(result_data[i]['coords']))
        prediction_per_atom.append(result_data[i]['free_energy'] / len(result_data[i]['coords']))
        system_config.append(reference_data[i]['config_type'])

        energy_diff.append(np.abs(np.abs(reference_data[i]['free_energy'] / len(result_data[i]['coords'])) -
                                  np.abs(result_data[i]['free_energy'] / len(result_data[i]['coords']))))

        df = 0
        for fa, fb in zip(np.array(result_data[i]['forces']), np.array(reference_data[i]['forces'])):
            df += np.linalg.norm(fa - fb) / 3
        force_diff.append(df / len(result_data[i]['forces']))

        # df = 0
        # for fa, fb in zip(np.array(result_data[i]['virial']), np.array(reference_data[i]['virial'])):
        #     df += np.linalg.norm(fa - fb) / 3
        # virial_diff.append(df/len(result_data[i]['virial']))

    return reference_per_atom, prediction_per_atom, energy_diff, force_diff, system_config


def scatterplot(result_energy, reference_energy, system_type, quip_time, max_energy_error,
                avg_energy_error, force_error, gap_name):

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', serif='Palatino')
    # plt.rcParams['font.family'] = 'DejaVu Sans'
    # plt.rcParams['font.sans-serif'] = 'cm'
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

    plt_color = {'fcc': 'y', 'bcc': 'navy', 'hcp': 'g', 'sc': 'm', 'slab': 'r', 'phonons': 'b',
                 'addition': 'brown', 'cluster': 'green', 'trimer': 'lightgreen', 'elastics': 'lightblue'}

    for ip, tp in enumerate(['fcc', 'bcc', 'hcp', 'sc', 'slab']):  # , 'phonons',
                             # 'addition', 'cluster', 'trimer', 'elastics']):
        plt_x = []
        plt_y = []
        for cnt, (res, ref) in enumerate(zip(result_energy, reference_energy)):
            if system_type[cnt] == tp:
                plt_x.append(ref)
                plt_y.append(res)

        plt.text(-3.7, -0.5 - (ip * 0.15), r'{}: {}'.format(tp, len(plt_x)), color=plt_color[tp], fontsize=6)
        plt.scatter(plt_x, plt_y, marker='.', color=plt_color[tp], label=None, s=0.5)

    # fcc_dft = -24.39050152 / 4
    # fcc_gap = -24.404683 / 4
    # plt.plot(fcc_dft, fcc_gap, marker='x', color='k')
    # plt.annotate(r'fcc', xy=(fcc_dft, fcc_gap), xytext=(fcc_dft - 0.1, fcc_gap - 0.5), color='y', fontsize=6,
    #              arrowprops=dict(facecolor='k', edgecolor='k', width=0.1,
    #                              headwidth=2.0, headlength=4.0, shrink=0.05))
    #
    # bcc_dft = -11.97199694 / 2
    # bcc_gap = -11.989746 / 2
    # plt.plot(bcc_dft, bcc_gap, marker='x', color='k')
    # plt.annotate(r'bcc', xy=(bcc_dft, bcc_gap), xytext=(bcc_dft - 0.5, bcc_gap + 0.5), color='navy', fontsize=6,
    #              arrowprops=dict(facecolor='k', edgecolor='k', width=0.1,
    #                              headwidth=2.0, headlength=4.0, shrink=0.05))
    #
    # hcp_dft = -17.78139257 / 3
    # hcp_gap = -17.785577 / 3
    # plt.plot(hcp_dft, hcp_gap, marker='x', color='k')
    # plt.annotate(r'hcp', xy=(hcp_dft, hcp_gap), xytext=(hcp_dft + 0.5, hcp_gap - 0.5), color='g', fontsize=6,
    #              arrowprops=dict(facecolor='k', edgecolor='k', width=0.1,
    #                              headwidth=2.0, headlength=4.0, shrink=0.05))
    #
    # sc_dft = -5.57232362
    # sc_gap = -5.6127757  # -5.6727757
    # plt.plot(sc_dft, sc_gap, marker='x', color='k')
    # plt.annotate(r'sc', xy=(sc_dft, sc_gap), xytext=(sc_dft - 0.5, sc_gap + 0.5), color='m', fontsize=6,
    #              arrowprops=dict(facecolor='k', edgecolor='k', width=0.1,
    #                              headwidth=2.0, headlength=4.0, shrink=0.05))

    plt.xlabel(r'Computed Energy [eV/atom]', fontsize=16, color='k')
    plt.ylabel(r'Predicted Energy [eV/atom]', fontsize=16, color='k')

    plt.scatter(5, 5, marker='.', color='k', label=r'GAP vs DFT', facecolor='w', s=25)

    plt.plot([-3.85, 0.5], [-3.85, 0.5], '-', color='k', linewidth=0.25)

    plt.text(-3, -0.6, r'Max error: {} eV/atom'.format(round(max_energy_error, 3)), fontsize=8)
    plt.text(-3, -0.75, r'Mean error: {} meV/atom'.format(round(avg_energy_error*1000, 1)), fontsize=8)
    plt.text(-3, -0.9, r'Mean force error: {} eV/\AA'.format(round(force_error, 3)), fontsize=8)

    print('QUIP runtime: {}'.format(quip_time))
    # plt.text(-3.0, -4.4, r'QUIP runtime: {}'.format(quip_time), fontsize=8)

    plt.legend(loc='upper left')

    plt.xlim(-4, 1)
    plt.ylim(-4, 1)

    plt.tight_layout()
    fname = 'GAP_vs_DFT-' + gap_name + '.png'
    plt.savefig(fname, dpi=300)
    plt.close()


reference = parse_xyz('test.xyz')

result, runtime = parse_quip('quip.result')
eref, epred, de, df, sys_conf = get_differences(result=result, reference=reference)

for ie, e in enumerate(de):
    if e > 0.1:
        print(ie, e)
        with open('error_{}.xyz'.format(ie), 'w') as f:
            for line in reference['data'][ie]:
                f.write(str(line))
        atoms = read('error_{}.xyz'.format(ie))
        write(filename='error_{}.vasp'.format(ie), images=atoms, format='vasp')
        os.unlink('error_{}.xyz'.format(ie))

scatterplot(result_energy=epred, reference_energy=eref, system_type=sys_conf,
            quip_time=runtime, max_energy_error=np.amax(de), avg_energy_error=np.sum(de) / len(de),
            force_error=np.sum(df) / len(df), gap_name='2b+SOAP')