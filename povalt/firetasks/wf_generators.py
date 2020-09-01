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

from fireworks import Firework, Workflow
from povalt.firetasks.training import LammpsMD
from povalt.firetasks.training import PotentialTraining


def potential_trainer(train_params):
    """
    Trains a potential with given parameters

    Args:
        train_params: parameters for gap_fit

    Returns:
        the workflow to add into Launchpad
    """

    if not train_params or len(train_params) != 33:
        raise ValueError('Training parameters have to be defined, abort.')

    fw_train = Firework([PotentialTraining(train_params=train_params)], parents=None, name='TrainTask')
    return Workflow([fw_train], name='TrainFlow')


def train_and_run_single_lammps(train_params, lammps_params):
    """
    Trains a potential and then runs LAMMPS MD with it

    Args:
        train_params: parameters for the potential training
        lammps_params:  parameters for the MD in LAMMPS

    Returns:
        the workflow for Launchpad
    """

    if not train_params or len(train_params) != 33:
        raise ValueError('Training parameters have to be defined, abort.')
    if not lammps_params or len(lammps_params) != 9:
        raise ValueError('LAMMPS parameters have to be defined, abort.')

    train_fw = Firework([PotentialTraining(train_params=train_params)], parents=None, name='TrainTask')
    md_run = Firework([LammpsMD(lammps_params=lammps_params)], parents=train_fw, name='Lammps_MD')

    return Workflow([train_fw, md_run], {train_fw: [md_run]}, name='train_and_MD')


def train_and_run_multiple_lammps(train_params, lammps_params, num_lammps):
    """
    Trains a potential and then runs LAMMPS MD with it

    Args:
        train_params: parameters for the potential training
        lammps_params:  parameters for the MD in LAMMPS
        num_lammps: number of LAMMPS MDs to run

    Returns:
        the workflow for Launchpad
    """

    if not train_params or len(train_params) != 33:
        raise ValueError('Training parameters have to be defined, abort.')
    if not lammps_params or len(lammps_params) != 9:
        raise ValueError('LAMMPS parameters have to be defined, abort.')

    all_fws = []
    dep_fws = []

    train_fw = Firework([PotentialTraining(train_params=train_params)], parents=None, name='TrainTask')

    all_fws.append(train_fw)

    for i in range(num_lammps):
        dep_fws.append(Firework([LammpsMD(lammps_params=lammps_params,
                                          db_file='/home/jank/work/test/fireworks/db.json')],
                                parents=train_fw, name='Lammps_MD'))

    all_fws.extend(dep_fws)

    return Workflow(all_fws, {train_fw: dep_fws}, name='train_and_multi_MD')
