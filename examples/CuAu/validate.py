import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_command_line(filename):
    with open(filename, 'r') as f:
        complete_line = ''
        for line in f:
            if 'command_line>' in line:
                complete_line += line.strip() + ' '

    return_line = ''
    for ib, char in enumerate(re.sub('_', '-', complete_line[24:-19])):
        return_line += char
        if ib % 115 == 0 and ib != 0:
            return_line += '\n'
    return_line = re.sub('{', '', return_line)
    return_line = re.sub('}', '', return_line)
    return return_line


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
            if line_count == 2:
                bits = line.split()
                for ib, bit in enumerate(bits):
                    if 'virial' in bit:
                        virial.append(float(bits[ib].split('=')[1][1:]))
                        for v in bits[ib+1:ib+8]:
                            virial.append(float(v))
                        virial.append(float(bits[ib+8][:-1]))
            if 2 < line_count <= n_atoms + 2:
                coord.append([float(x) for x in line.split()[1:4]])
                forces.append([float(x) for x in line.split()[12:15]])
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

    return reference_per_atom, prediction_per_atom, energy_diff, force_diff, system_config


def scatterplot(result_energy, reference_energy, system_type, force_diffs, quip_time):
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', serif='Palatino')

    plt_color = {'fcc_Au': 'y', 'bcc_Au': 'navy', 'hcp_Au': 'g', 'sc_Au': 'm', 'slab_Au': 'r',
                 'fcc_Cu': 'y', 'bcc_Cu': 'navy', 'hcp_Cu': 'g', 'sc_Cu': 'm', 'slab_Cu': 'r',
                 'fcc_AuCu': 'y', 'bcc_AuCu': 'navy', 'hcp_AuCu': 'g', 'sc_AuCu': 'm'}

    all_list = [['fcc_Au', 'bcc_Au', 'hcp_Au', 'sc_Au', 'slab_Au'],
                ['fcc_Cu', 'bcc_Cu', 'hcp_Cu', 'sc_Cu', 'slab_Cu'],
                ['fcc_AuCu', 'bcc_AuCu', 'hcp_AuCu', 'sc_AuCu']]

    ax = []
    for ispec, sub_list in enumerate(all_list):
        plt_string = int('31'+str(ispec+1))
        ax.append(plt.subplot(plt_string))
        e_errors = []
        f_errors = []
        for ip, tp in enumerate(sub_list):
            plt_x = []
            plt_y = []
            for cnt, (res, ref, df) in enumerate(zip(result_energy, reference_energy, force_diffs)):
                if system_type[cnt] == tp:
                    plt_x.append(ref)
                    plt_y.append(res)
                    e_errors.append(np.abs(res - ref))
                    f_errors.append(df)

            ax[ispec].text(-3.9, -0.5 - (ip * 0.3), r'{}'.format(re.sub('_', '\_', tp)),
                           color=plt_color[tp], fontsize=5)
            ax[ispec].scatter(plt_x, plt_y, marker='.', color=plt_color[tp], label=None, s=0.5)

        ax[ispec].plot([-4.25, 0.5], [-4.25, 0.5], ':', color='k', linewidth=0.25)
        ax[ispec].text(-3.5, -0.6, r'Max: {} eV/atom'.format(round(np.amax(e_errors), 3)), fontsize=5)
        ax[ispec].text(-3.5, -0.9, r'Mean: {} meV/atom'.format(round(np.sum(e_errors) / len(e_errors) * 1000, 1)),
                       fontsize=5)
        ax[ispec].text(-3.5, -1.2, r'Mean force: {} eV/\AA'.format(round(np.sum(f_errors)/len(f_errors), 3)),
                       fontsize=5)

        ax[ispec].text(-1.2, -3.0, r'structures: {}'.format(len(e_errors)), fontsize=8)

        ax[ispec].set_xlim(-4.5, 1)
        ax[ispec].set_ylim(-4.5, 1)

        if ispec < 2:
            ax[ispec].axes.xaxis.set_visible(False)

    ax[2].set_xlabel(r'Computed [eV/atom]', fontsize=8, color='k')
    ax[1].set_ylabel(r'Predicted [eV/atom]', fontsize=8, color='k')

    # plt.text(-2.5, -3.6, r'runtime: {}'.format(quip_time), fontsize=8)

    plt.tight_layout()
    plt.savefig('Multi_GAP.png', dpi=300)
    plt.close()


reference = parse_xyz('test.xyz')

result, runtime = parse_quip('quip.result')
eref, epred, de, df, sys_conf = get_differences(result=result, reference=reference)

# for ie, e in enumerate(de):
#     if e > 0.1:
#         print(ie, e)
#         with open('/tmp/error_{}.xyz'.format(ie), 'w') as f:
#             for line in reference['data'][ie]:
#                 f.write(str(line))
#         atoms = read('/tmp/error_{}.xyz'.format(ie))
#         write(filename='error_{}_{}eV.vasp'.format(ie, np.round(e, 3)), images=atoms, format='vasp')
#         os.unlink('/tmp/error_{}.xyz'.format(ie))

scatterplot(result_energy=epred, reference_energy=eref, system_type=sys_conf, force_diffs=df, quip_time=runtime)
