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
from povalt.firetasks.training import PotentialTraining


def potential_trainer(some_args=None):
    if some_args is None:
        all_args = ['asdf', 'asdf', 'ret']
    else:
        all_args = some_args

    fw_train = Firework([PotentialTraining(my_params=all_args)], parents=None, name='TrainTask')

    return Workflow([fw_train], name='TrainFlow')
