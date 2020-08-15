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

from povalt.firetasks.vasp import VaspTasks


class FccGenerator:
    """
    Basic class to generate FCC structures with random displacements
    """
    def __init__(self):
        """
        init
        """
        pass

    def generate_fcc_cell(self, atom_type, ncore):
        """
        generates a FCC cell with the specified atom as only occupant
        input:
            atom_type: type of atom to put into the cell
            ncore: VASP ncore setting
        returns:
            a relaxation workflow for vasp
        """
        metadata = {'name': 'cell generation',
                    'task': 'relaxation',
                    'cell': 'fcc'}
        return VaspTasks.get_relax_wf(structure='fcc', structure_name='FCC input cell', atom_type=atom_type,
                                      vasp_cmd='srun --nodes 1 vasp_std', ncore=ncore, metadata=metadata)


class BccGenerator:
    """
    Basic class to generate BCC structures with random displacements
    """
    def __init__(self):
        """
        init
        """
        pass

    def generate_bcc_cell(self, atom_type):
        """
        generates a BCC cell with the specified atom as only occupant
        """
        pass


class ScGenerator:
    """
    Basic class to generate SC structures with random displacements
    """
    def __init__(self):
        """
        init
        """
        pass

    def generate_sc_cell(self, atom_type):
        """
        generates a SC cell with the specified atom as only occupant
        """
        pass


class HcpGenerator:
    """
    Basic class to generate HCP structures with random displacements
    """
    def __init__(self):
        """
        init
        """
        pass

    def generate_hcp_cell(self, atom_type):
        """
        generates a HCP cell with the specified atom as only occupant
        """
        pass
