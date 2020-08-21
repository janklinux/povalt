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
import subprocess


def remove_residual_files(work_dir):
    """
    Removes all files that are named like the ones we will generate during the procedure
    for training, LAMMPS MD and validation
    Args:
        work_dir:

    Returns:
        nothing
    """

    file_list = ['final_positions.atom', 'fit_output', 'fit_error', 'geo.in',
                 'lammps_error', 'lammps.in', 'LOG', 'log.lammps']
    for file in os.listdir(work_dir):
        if file in file_list:
            os.unlink(file)


def find_binary(binary):
    """
    Finds an executable in bash shells

    Args:
        binary: name of the binary

    Returns:
        full path of the binary if it exists, None otherwise
    """

    try:
        result = subprocess.check_output(['which', str(binary)], encoding='utf8')
    except subprocess.CalledProcessError:
        result = None
    if result is None:
        raise FileNotFoundError('program >{}< not found in path'.format(binary))
    return result
