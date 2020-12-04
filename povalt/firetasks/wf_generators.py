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
import datetime
from fireworks import Firework, Workflow, ScriptTask
from povalt.firetasks.base import Lammps, LammpsCG, PotentialTraining, Aims
from atomate.vasp.fireworks.core import StaticFW
from fireworks.utilities.fw_utilities import explicit_serialize


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


def run_lammps_auto_dft(lammps_params, structures, db_file, al_file):
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


def run_lammps(lammps_params, structure, db_file, al_file):
    """
    Runs LAMMPS with the supplied structures
    Use when a trained potential exists and we need more LAMMPS runs with it

    Args:
        lammps_params:  parameters for the MD in LAMMPS
        structure: pymatgen structure
        db_file: file containing the db information
        al_file: auto-launcher settings

    Returns:
        the workflow for Launchpad
    """

    db_info, al_info = read_info(db_file=db_file, al_file=al_file)

    params = lammps_params.copy()
    params['structure'] = structure.as_dict()

    return Workflow([Firework([LammpsCG(lammps_params=params, db_info=db_info)], name='LAMMPS CG')], name='CG_LAMMPS')


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


def vasp_static_wf(structure, struc_name='', name='Static_run', vasp_input_set=None,
                   vasp_cmd=None, db_file=None, user_kpoints_settings=None, tag=None, metadata=None):
    """
    Static VASP workflow, to generate single point DFT training data
    Args:
        structure: pymatgen structure object
        struc_name: name of the structure
        name: name of the workflow
        vasp_input_set: materials project input set
        vasp_cmd: command to run
        db_file: file containing db related infos
        user_kpoints_settings: kpoints object for k-grid
        tag: tag for the wokflow
        metadata: additional data for the workflow

    Returns:
        the workflow add into LaunchPad
    """
    if vasp_input_set is None:
        raise ValueError('INPUTSET needs to be defined...')
    if user_kpoints_settings is None:
        raise ValueError('You have to specify the K-grid...')
    if vasp_cmd is None:
        raise ValueError('vasp_cmd needs to be set by user...')
    if tag is None:
        tag = datetime.datetime.now().strftime('%Y/%m/%d-%T')

    vis = vasp_input_set
    v = vis.as_dict()
    v.update({"user_kpoints_settings": user_kpoints_settings})
    vis_static = vis.__class__.from_dict(v)

    fws = [StaticFW(structure=structure, vasp_input_set=vis_static, vasp_cmd=vasp_cmd,
                    db_file=db_file, name="{} -- static".format(tag))]

    wfname = "{}: {}".format(struc_name, name)
    return Workflow(fws, name=wfname, metadata=metadata)


def aims_single_basis(aims_cmd, control, structure, basis_set, basis_dir, metadata, name, parents=None):
    """
    Workflow to run FHIaims with the specified basis set
    Args:
        aims_cmd: command to run FHIaims
        control: control.in file
        structure: pymatgen structure object
        basis_set: light or tight
        basis_dir: directory where folder for basis_set is located
        metadata: additional data
        name: name of the WF
        parents: do we have some?

    Returns:
        The workflow for LaunchPad
    """
    fws = Firework([AimsSingleBasis(aims_cmd=aims_cmd, control=control, structure=structure, basis_set=basis_set,
                                    basis_dir=basis_dir, metadata=metadata, name=name, parents=parents)])

    wfname = "{}: {}".format('Aims single basis ', name)
    return Workflow(fws, name=wfname, metadata=metadata)


def aims_double_basis(aims_cmd, control, structure, basis_dir, metadata, name):
    """
    Workflow to run FHIaims first light then tight basis sets automatically
    Args:
        aims_cmd: command to run FHIaims
        control: control.in file
        structure: pymatgen structure object
        basis_dir: directory where folder for basis_set is located
        metadata: additional data
        name: name of the WF

    Returns:
        The workflow for LaunchPad

    """
    fws = Firework([Aims(aims_cmd=aims_cmd, control=control, structure=structure, basis_set='light',
                   basis_dir=basis_dir, single_basis=False, rerun_metadata=metadata)])
    wfname = "{}: {}".format('Aims light start for final tight ', name)
    return Workflow([fws], name=wfname, metadata=metadata)


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
