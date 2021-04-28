import os
import json
import numpy as np


def modify_mu(settings, max_mu_in, min_mu_in, mu_in, d_mu_in,
              max_temp_in, min_temp_in, temp_in, temp_inc_in, mode_in=None):
    if mode_in is None or (mode_in != 'heat' and mode_in != 'cool' and
                           mode_in != 'up' and mode_in != 'down'):
        raise AttributeError('mode must be heat | cool | up | down')

    if mode_in == 'cool':
        settings['driver']['initial_conditions']['temperature'] = max_temp_in
        settings['driver']['final_conditions']['temperature'] = min_temp_in
        settings['driver']['incremental_conditions']['temperature'] = -temp_inc_in

        settings['driver']['initial_conditions']['param_chem_pot'] = {'a': mu_in}
        settings['driver']['final_conditions']['param_chem_pot'] = {'a': mu_in}
        settings['driver']['incremental_conditions']['param_chem_pot'] = {'a': 0.0}

    if mode_in == 'heat':
        settings['driver']['initial_conditions']['temperature'] = min_temp_in
        settings['driver']['final_conditions']['temperature'] = max_temp_in
        settings['driver']['incremental_conditions']['temperature'] = temp_inc_in

        settings['driver']['initial_conditions']['param_chem_pot'] = {'a': mu_in}
        settings['driver']['final_conditions']['param_chem_pot'] = {'a': mu_in}
        settings['driver']['incremental_conditions']['param_chem_pot'] = {'a': 0.0}

    if mode_in == 'up':
        settings['driver']['initial_conditions']['temperature'] = temp_in
        settings['driver']['final_conditions']['temperature'] = temp_in
        settings['driver']['incremental_conditions']['temperature'] = 0.0

        settings['driver']['initial_conditions']['param_chem_pot'] = {'a': min_mu_in}
        settings['driver']['final_conditions']['param_chem_pot'] = {'a': max_mu_in}
        settings['driver']['incremental_conditions']['param_chem_pot'] = {'a': d_mu_in}

    if mode_in == 'down':
        settings['driver']['initial_conditions']['temperature'] = temp_in
        settings['driver']['final_conditions']['temperature'] = temp_in
        settings['driver']['incremental_conditions']['temperature'] = 0.0

        settings['driver']['initial_conditions']['param_chem_pot'] = {'a': max_mu_in}
        settings['driver']['final_conditions']['param_chem_pot'] = {'a': min_mu_in}
        settings['driver']['incremental_conditions']['param_chem_pot'] = {'a': -d_mu_in}


with open('base_mc.json') as f:
    base_settings = json.load(f)

dmu = 0.1
max_mu = 5
min_mu = -4
temp_min = 10
temp_max = 2000
temp_inc = 20

mu_grid = np.linspace(start=min_mu, stop=max_mu, num=int((max_mu-min_mu)/dmu), endpoint=True)
temp_grid = np.linspace(start=temp_min, stop=temp_max, num=int((temp_max-temp_min)/temp_inc), endpoint=True)

base_dir = os.getcwd()

for mode in ['heat', 'cool']:
    os.mkdir(mode)
    for i, mu in enumerate(mu_grid):
        os.mkdir(os.path.join(mode, str(i)))

        modify_mu(base_settings, max_mu_in=max_mu, min_mu_in=min_mu, mu_in=mu, d_mu_in=dmu,
                  max_temp_in=temp_max, min_temp_in=temp_min, temp_in=0, temp_inc_in=temp_inc, mode_in=mode)

        with open(os.path.join(base_dir, mode, str(i), 'gc_mc.json'), 'w') as f:
            json.dump(base_settings, f)

        with open(os.path.join(base_dir, mode, str(i), 'job_script'), 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('#SBATCH -n 1\n')
            f.write('#SBATCH -N 1\n')
            f.write('#SBATCH --mem-per-cpu=3900mb\n')
            f.write('#SBATCH -t 5-00:00:00\n')
            f.write('#SBATCH --partition=Def\n')
            f.write('#SBATCH --constraint=E5649\n')
            f.write('#SBATCH --job-name=%s\n' % (str(mode) + '_' + str(i)))
            f.write('#SBATCH -o out\n')
            f.write('#SBATCH -e err\n')
            f.write('#SBATCH --mail-type=FAIL\n')
            f.write('#SBATCH --mail-user=jank@numphys.org\n')
            f.write('ulimit -s unlimited\n')
            f.write('cd %s\n' % os.path.join(base_dir, mode, str(i)))
            f.write('casm monte -s gc_mc.json\n')

        os.chdir(os.path.join(base_dir, mode, str(i)))
        os.system('sbatch job_script')
        os.chdir(base_dir)

for mode in ['up', 'down']:
    os.mkdir(mode)
    for i, temp in enumerate(temp_grid):
        os.mkdir(os.path.join(mode, str(i)))

        modify_mu(base_settings, max_mu_in=max_mu, min_mu_in=min_mu, mu_in=0, d_mu_in=dmu,
                  max_temp_in=temp_max, min_temp_in=temp_min, temp_in=temp, temp_inc_in=temp_inc, mode_in=mode)

        with open(os.path.join(base_dir, mode, str(i), 'gc_mc.json'), 'w') as f:
            json.dump(base_settings, f)

        with open(os.path.join(base_dir, mode, str(i), 'job_script'), 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('#SBATCH -n 1\n')
            f.write('#SBATCH -N 1\n')
            f.write('#SBATCH --mem-per-cpu=2900mb\n')
            f.write('#SBATCH -t 5-00:00:00\n')
            f.write('#SBATCH --partition=Def\n')
            f.write('#SBATCH --job-name=%s\n' % (str(mode) + '_' + str(i)))
            f.write('#SBATCH -o out\n')
            f.write('#SBATCH -e err\n')
            f.write('#SBATCH --mail-type=END\n')
            f.write('#SBATCH --mail-user=jank@numphys.org\n')
            f.write('ulimit -s unlimited\n')
            f.write('cd %s\n' % os.path.join(base_dir, mode, str(i)))
            f.write('casm monte -s gc_mc.json\n')

        os.chdir(os.path.join(base_dir, mode, str(i)))
        os.system('sbatch job_script')
        os.chdir(base_dir)
