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
from povalt.firetasks.training import Lammps_MD
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


def train_and_run_lammps(train_params, lammps_params):
    """
    Trains a potential and the nruns LAMMPS MD with it

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

    fw_train = Firework([PotentialTraining(train_params=train_params)], parents=None, name='TrainTask')
    md_run = Firework([Lammps_MD(lammps_params=lammps_params)], parents=fw_train, name='Lammps_MD')
    return Workflow([fw_train, md_run], {fw_train: [md_run]}, name='train_and_MD')
