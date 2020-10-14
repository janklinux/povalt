"""
Python package for training, validation and refinement of machine learned potentials

Copyright (C) 2020, Jan Kloppenburg

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import json
from fireworks import Firework, Workflow, ScriptTask
from povalt.firetasks.base import Lammps, PotentialTraining, Aims


def train_potential(train_params, for_validation, db_file):
    """
    Trains a potential with given parameters and stores it in the db

    Args:
        train_params: parameters for gap_fit
        for_validation: is the potential for validation (boolean)
        db_file: database info for storing the potential

    Returns:
        the workflow to add into Launchpad
    """

    db_info, al_info = read_info(db_file=db_file, al_file=None)

    fw_train = Firework([PotentialTraining(train_params=train_params, for_validation=for_validation,
                                           al_info=None, db_info=db_info)], parents=None, name='TrainTask')
    return Workflow([fw_train], name='TrainFlow')


def run_lammps(lammps_params, structures, db_file, al_file):
    """
    Runs LAMMPS with the supplied structures
    Use when a trained potential exists and we need more LAMMPS runs with it

    Args:
        lammps_params:  parameters for the MD in LAMMPS
        structures: list of pymatgen structures
        db_file: file containing the db information
        al_file: auto-launcher settings

    Returns:
        the workflow for Launchpad
    """

    db_info, al_info = read_info(db_file=db_file, al_file=al_file)

    lmp_fws = []
    for s in structures:
        params = lammps_params.copy()
        params['structure'] = s.as_dict()
        lmp_fws.append(Firework([Lammps(lammps_params=params, db_info=db_info)], name='LAMMPS CG'))

    return Workflow(lmp_fws, name='multi_LAMMPS')


def train_and_run_multiple_lammps(train_params, lammps_params, structures, db_file, is_slab, al_file=None):
    """
    Trains a potential and then runs LAMMPS with it, single machine workflow meaning training and validation runs
    on the same cluster

    Args:
        train_params: parameters for the potential training
        lammps_params:  parameters for LAMMPS
        structures: list of structures to run
        db_file: json formatted file containing the db info
        al_file: json formatted file containing the auto-launch settings
        is_slab: True is structure is slab, for VASP runs

    Returns:
        the workflow for Launchpad
    """

    if al_file is None:
        raise FileNotFoundError('al_file can not be missing when using auto-launch')

    db_info, al_info = read_info(db_file=db_file, al_file=al_file)

    all_fws = []
    dep_fws = []

    train_fw = Firework([PotentialTraining(train_params=train_params, for_validation=False,
                                           db_info=db_info, al_info=al_info)], parents=None, name='TrainTask')
    all_fws.append(train_fw)

    launch_fw = Firework([ScriptTask.from_str('cd {}; qlaunch -q {} rapidfire --nlaunches {}'
                                              .format(al_info['base_dir'],
                                                      os.path.join(al_info['base_dir'], al_info['queue_file']),
                                                      str(al_info['num_launches'])))],
                         parents=train_fw, name='AutoLauncher')
    all_fws.append(launch_fw)

    for s in structures:
        params = lammps_params.copy()
        params['structure'] = s.as_dict()
        dep_fws.append(Firework([Lammps(lammps_params=params, db_info=db_info, is_slab=is_slab)],
                                parents=train_fw, name='LAMMPS CG'))

    all_fws.extend(dep_fws)

    return Workflow(all_fws, {train_fw: dep_fws}, name='train_multiLammps_autolaunch')


def aims_relax(aims_cmd, control, structure, basis_dir, metadata, name):
    """
    Performs a light and tight relaxation for the given structure object

    Args:
        aims_cmd: command to run aims i.e. srun aims
        control: control.in as list of lines
        structure: pymatgen structure to relax
        basis_dir: directory where the basis is located (light / tight directories)
        metadata: metadata to pass into both runs for identification
        name: name of the workflow

    Returns:
        Workflow to insert into LaunchPad
    """

    run_fw = Firework([Aims(aims_cmd=aims_cmd, control=control, structure=structure,
                            basis_dir=basis_dir, basis_set='light', rerun_metadata=metadata)])

    return Workflow([run_fw], name=name, metadata=metadata)


def read_info(db_file, al_file):
    """
    simple read in of the json files
    Args:
        db_file: database info
        al_file: auto-launch info

    Returns:
        dictionaries db_info and al_info
    """

    if al_file is not None:
        try:
            with open(al_file, 'r') as f:
                al_info = json.load(f)
        except FileNotFoundError:
            al_info = None
    else:
        al_info = None

    try:
        with open(db_file, 'r') as f:
            db_info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('db_file has to exist')

    return db_info, al_info
