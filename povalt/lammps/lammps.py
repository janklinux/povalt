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

import tempfile
from ase.atoms import Atoms
from ase.io import read as ase_read
from ase.io.lammpsdata import write_lammps_data
from pymatgen.core.structure import Structure


class Lammps:
    """
    General class for all LAMMPS related jobs like generating and running structures
    for validation and potential refinement
    """
    def __init__(self, structure):
        """
        init parameters and find out what object we have
        Args:
            structure: the structure object we deal with
        """

        if isinstance(structure, Structure):
            file = tempfile.mkstemp()[1]
            structure.to(fmt='POSCAR', filename=file)
            self.structure = ase_read(file, format='vasp')
        elif isinstance(structure, Atoms):
            self.structure = structure
        else:
            raise ValueError('Structure object has to be pymatgen structure or ase atoms object')

    def write_MD(self, output_file, structure, lammps_settings):
        """
        Writes LAMMPS input files to run MD
        Args:
            output_file: file to write to
            structure: the structure to write
            lammps_settings: settings string to write as lammps input

        Returns:
            nothing, writes content to files
        """

        write_lammps_data(fileobj=output_file, atoms=structure)
        with open('lammps.in', 'w') as f:
            for line in lammps_settings:
                f.write(line)
