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
import gzip
import glob
import subprocess
import numpy as np
from pymatgen import Structure
from custodian import Custodian
from custodian.custodian import Job
from custodian.fhi_aims.handlers import AimsRelaxHandler, FrozenJobErrorHandler
from custodian.fhi_aims.validators import AimsConvergedValidator
from fireworks import FiretaskBase, explicit_serialize, Firework


class AimsJob(Job):
    """
    A basic job. Just runs whatever is in the directory. But conceivably
    can be a complex processing of inputs etc. with initialization.
    """

    def __init__(self, aims_cmd, control, structure, basis_set, basis_dir, metadata, single_basis,
                 output_file='run', stderr_file='std_err.txt'):
        """
        This constructor is necessarily complex due to the need for
        flexibility. For standard kinds of runs, it's often better to use one
        of the static constructors. The defaults are usually fine too.

        Args:
            aims_cmd (str): Command to run aims as a list of args. For example,
                if you are using mpirun, it can be something like
                ["mpirun", "aims"]
            control: control.in as list of lines
            structure: pymatgen structure object
            basis_set: light or tight
            basis_dir: directory containing the light/tight directory of basis files
            metadata: metadata to add into wf
            single_basis: True for a single basis set run (no auto-add), False for light/tight relax
            output_file (str): Name of file to direct standard out to.
                Defaults to "vasp.out".
            stderr_file (str): Name of file to direct standard error to.
                Defaults to "std_err.txt".
        """
        self.aims_cmd = aims_cmd
        self.output_file = output_file
        self.stderr_file = stderr_file
        self.run_dir = os.getcwd()
        self.control = control
        self.structure = structure
        if basis_set.lower() not in ['light', 'tight']:
            raise ValueError('basis set can be only light or tight for now...')
        self.basis_set = basis_set
        self.basis_dir = basis_dir
        self.metadata = metadata
        if not isinstance(single_basis, bool):
            raise ValueError('single_basis variable has to be boolean')
        self.single_basis = single_basis

    def setup(self):
        """
        Performs initial setup for AimsJob
        """

        base = os.path.join(self.basis_dir, self.basis_set)
        basis = []
        elements = []
        for s in self.structure.types_of_species:
            if str(s) not in elements:
                elements.append(str(s))

        for el in elements:
            basis.append(glob.glob(base+'/*'+el+'*')[0])

        with open('control.in', 'wt') as f:
            for line in self.control:
                f.write(line)
            for fname in basis:
                with open(fname, 'r') as fin:
                    f.write(fin.read())

        self.structure.to(filename='geometry.in', fmt='aims')

    def run(self):
        """
        Performs the actual run

        Returns:
            subprocess.Popen for monitoring
        """

        with open(self.output_file, 'w') as f_std, open(self.stderr_file, "w", buffering=1) as f_err:
            p = subprocess.Popen(self.aims_cmd.split(), stdout=f_std, stderr=f_err)
        return p

    def postprocess(self):
        """
        Postprocessing is adding tight run if previous was light and gzip for now
        """
        parse_atoms = False
        parse_forces = False
        parse_stress = False

        scf = dict()
        i_step = 0
        i_atom = 0
        i_force = 0
        n_atoms = None
        md_lattice = []

        with open(self.output_file, 'r') as f:
            content = f.readlines()

        md_run = False
        relaxation = False
        for line in content:
            if 'MD_run' in line:
                md_run = True
            if 'relax_geometry' in line:
                if not line.strip().startswith('#'):
                    relaxation = True

        if md_run:
            for line in content:
                if '| Number of atoms' in line:
                    n_atoms = int(line.split()[5])
                if 'lattice_vector' in line:
                    md_lattice.append([float(x) for x in line.split()[1:4]])
                if 'Begin self-consistency loop:' in line:
                    i_step += 1
                    i_atom = 0
                    i_force = 0
                    scf[i_step] = {'atoms': [], 'forces': [], 'lattice': [], 'stress': [],
                                   'species': [], 'energy': 0.0}
                if '| Total energy                  :' in line:
                    scf[i_step]['energy'] = float(line.split()[6])
                if parse_forces:
                    scf[i_step]['forces'].append([float(x) for x in line.split()[2:5]])
                    i_force += 1
                    if i_force == n_atoms:
                        parse_forces = False
                if 'Total atomic forces (unitary forces cleaned) [eV/Ang]:' in line:
                    parse_forces = True
                if parse_atoms:
                    if len(line.split()) == 0:
                        continue
                    elif 'atom' in line:
                        scf[i_step]['atoms'].append([float(x) for x in line.split()[1:4]])
                        scf[i_step]['species'].append(line.split()[4])
                        i_atom += 1
                    if i_atom == n_atoms:
                        parse_atoms = False
                if 'Atomic structure (and velocities) as used in the preceding time step:' in line:
                    parse_atoms = True
                if parse_stress:
                    if '|  x       ' in line:
                        scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  y       ' in line:
                        scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  z       ' in line:
                        scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  Pressure:' in line:
                        parse_stress = False
                if 'Analytical stress tensor - Symmetrized' in line:
                    parse_stress = True

        elif relaxation:
            for line in content:
                if '| Number of atoms' in line:
                    n_atoms = int(line.split()[5])
                if 'Begin self-consistency loop:' in line:
                    i_step += 1
                    i_atom = 0
                    i_force = 0
                    scf[i_step] = {'atoms': [], 'forces': [], 'lattice': [], 'stress': [],
                                   'species': [], 'energy': 0.0}
                if '| Total energy                  :' in line:
                    scf[i_step]['energy'] = float(line.split()[6])
                if parse_forces:
                    scf[i_step]['forces'].append([float(x) for x in line.split()[2:5]])
                    i_force += 1
                    if i_force == n_atoms:
                        parse_forces = False
                if 'Total atomic forces (unitary forces cleaned) [eV/Ang]:' in line:
                    parse_forces = True
                if parse_atoms:
                    if 'lattice' in line:
                        scf[i_step]['lattice'].append([float(x) for x in line.split()[1:4]])
                    elif len(line.split()) == 0:
                        continue
                    else:
                        scf[i_step]['atoms'].append([float(x) for x in line.split()[1:4]])
                        scf[i_step]['species'].append(line.split()[4])
                        i_atom += 1
                    if i_atom == n_atoms:
                        parse_atoms = False
                if 'x [A]             y [A]             z [A]' in line:
                    parse_atoms = True
                if parse_stress:
                    if '|  x       ' in line:
                        scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  y       ' in line:
                        scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  z       ' in line:
                        scf[i_step]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  Pressure:' in line:
                        parse_stress = False
                if 'Analytical stress tensor - Symmetrized' in line:
                    parse_stress = True

        else:
            scf[1] = {'atoms': [], 'forces': [], 'lattice': [], 'stress': [],
                      'species': [], 'energy': 0.0}
            for line in content:
                if '| Number of atoms' in line:
                    n_atoms = int(line.split()[5])
                if 'lattice_vector' in line:
                    scf[1]['lattice'].append([float(x) for x in line.split()[1:4]])
                if line.strip().startswith('atom') or line.strip().startswith('atom_frac'):
                    scf[1]['atoms'].append([float(x) for x in line.split()[1:4]])
                    scf[1]['species'].append(line.split()[4])
                if '| Total energy                  :' in line:
                    scf[1]['energy'] = float(line.split()[6])
                if parse_forces:
                    scf[1]['forces'].append([float(x) for x in line.split()[2:5]])
                    i_force += 1
                    if i_force == n_atoms:
                        parse_forces = False
                if 'Total atomic forces (unitary forces cleaned) [eV/Ang]:' in line:
                    parse_forces = True
                if parse_stress:
                    if '|  x       ' in line:
                        scf[1]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  y       ' in line:
                        scf[1]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  z       ' in line:
                        scf[1]['stress'].append([float(x) for x in line.split()[2:5]])
                    if '|  Pressure:' in line:
                        parse_stress = False
                if 'Analytical stress tensor - Symmetrized' in line:
                    parse_stress = True
                if 'No geometry change - only restart of scf mixer after initial iterations.' in line:
                    i_force = 0
                    scf[1]['forces'] = []
                    scf[1]['stress'] = []

        with open('parsed.xyz', 'w') as f:
            for i in range(1, len(scf) + 1):
                if md_run:
                    scf[i]['lattice'] = md_lattice
                    lat = md_lattice
                    scf[i]['stress'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                    stress = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                elif relaxation:
                    if len(scf[i]['stress']) > 0:
                        lat = np.array([[scf[i]['lattice'][0][0], scf[i]['lattice'][0][1], scf[i]['lattice'][0][2]],
                                        [scf[i]['lattice'][1][0], scf[i]['lattice'][1][1], scf[i]['lattice'][1][2]],
                                        [scf[i]['lattice'][2][0], scf[i]['lattice'][2][1], scf[i]['lattice'][2][2]]])
                        stress = np.array([[scf[i]['stress'][0][0], scf[i]['stress'][0][1], scf[i]['stress'][0][2]],
                                           [scf[i]['stress'][1][0], scf[i]['stress'][1][1], scf[i]['stress'][1][2]],
                                           [scf[i]['stress'][2][0], scf[i]['stress'][2][1], scf[i]['stress'][2][2]]])
                    elif len(scf[i]['stress']) == 0 and len(scf[i]['lattice']) > 0:
                        lat = np.array([[scf[i]['lattice'][0][0], scf[i]['lattice'][0][1], scf[i]['lattice'][0][2]],
                                        [scf[i]['lattice'][1][0], scf[i]['lattice'][1][1], scf[i]['lattice'][1][2]],
                                        [scf[i]['lattice'][2][0], scf[i]['lattice'][2][1], scf[i]['lattice'][2][2]]])
                        scf[i]['stress'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                        stress = scf[i]['stress']
                    else:
                        scf[i]['lattice'] = [[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]]
                        lat = scf[i]['lattice']
                        scf[i]['stress'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                        stress = scf[i]['stress']
                else:
                    lat = np.array([[scf[i]['lattice'][0][0], scf[i]['lattice'][0][1], scf[i]['lattice'][0][2]],
                                    [scf[i]['lattice'][1][0], scf[i]['lattice'][1][1], scf[i]['lattice'][1][2]],
                                    [scf[i]['lattice'][2][0], scf[i]['lattice'][2][1], scf[i]['lattice'][2][2]]])
                    stress = np.array([[scf[i]['stress'][0][0], scf[i]['stress'][0][1], scf[i]['stress'][0][2]],
                                       [scf[i]['stress'][1][0], scf[i]['stress'][1][1], scf[i]['stress'][1][2]],
                                       [scf[i]['stress'][2][0], scf[i]['stress'][2][1], scf[i]['stress'][2][2]]])

                vol = np.dot(lat[0], np.cross(lat[1], lat[2]))
                virial = -np.dot(vol, stress)

                f.write('{}\n'.format(n_atoms))
                f.write('Lattice="{:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f}"'
                        ' Properties=species:S:1:pos:R:3:forces:R:3:force_mask:L:1'
                        ' stress="{:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f}"'
                        ' virial="{:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f}"'
                        ' free_energy={:6.6f} pbc="T T T"'
                        ' config_type=CEbulk\n'.format(
                            scf[i]['lattice'][0][0], scf[i]['lattice'][0][1], scf[i]['lattice'][0][2],
                            scf[i]['lattice'][1][0], scf[i]['lattice'][1][1], scf[i]['lattice'][1][2],
                            scf[i]['lattice'][2][0], scf[i]['lattice'][2][1], scf[i]['lattice'][2][2],
                            scf[i]['stress'][0][0], scf[i]['stress'][0][1], scf[i]['stress'][0][2],
                            scf[i]['stress'][1][0], scf[i]['stress'][1][1], scf[i]['stress'][1][2],
                            scf[i]['stress'][2][0], scf[i]['stress'][2][1], scf[i]['stress'][2][2],
                            virial[0][0], virial[0][1], virial[0][2],
                            virial[1][0], virial[1][1], virial[1][2],
                            virial[2][0], virial[2][1], virial[2][2],
                            scf[i]['energy']))

                for j in range(len(scf[i]['atoms'])):
                    f.write('{} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} {:6.6f} 0\n'.
                            format(scf[i]['species'][j],
                                   scf[i]['atoms'][j][0], scf[i]['atoms'][j][1], scf[i]['atoms'][j][2],
                                   scf[i]['forces'][j][0], scf[i]['forces'][j][1], scf[i]['forces'][j][2]))

        for file in os.listdir(self.run_dir):
            self.compress(file)
            os.unlink(file)

    def get_relaxed_structure(self):
        """
        Returns the relaxed structure for the tight run
        Returns:
            the relaxed structure object
        """

        print('GETTING REL STRU')

        if self.basis_set == 'tight':
            return None, dict()

        if self.single_basis:
            return None, dict()

        os.chdir(self.run_dir)
        if os.path.isfile('geometry.in.next_step.gz'):
            self.decompress('geometry.in.next_step')
            structure = Structure.from_file('geometry.in.next_step')
            os.unlink('geometry.in.next_step')
        else:
            self.decompress('geometry.in')
            structure = Structure.from_file('geometry.in')
            os.unlink('geometry.in')

        param_dict = {'aims_cmd': self.aims_cmd,
                      'output_file': 'run.tight',
                      'control': self.control,
                      'basis_dir': self.basis_dir,
                      'metadata': self.metadata}

        return structure, param_dict

    @staticmethod
    def decompress(filename):
        with open(filename, 'wb') as f_out:
            with gzip.open(filename+'.gz', 'rb') as f_in:
                f_out.write(f_in.read())

    @staticmethod
    def compress(filename):
        with gzip.open(filename+'.gz', 'wb') as f_out:
            with open(filename, 'rb') as f_in:
                f_out.write(f_in.read())


@explicit_serialize
class RunAimsCustodian(FiretaskBase):
    """
    Run FHIaims

    Required params:
        aims_cmd (str): the name of the full executable for running FHIaims

    Optional params:

    """
    required_params = ['aims_cmd', 'control', 'structure', 'basis_set', 'basis_dir', 'single_basis']
    optional_params = ['aims_output', 'rerun_metadata']

    def run_task(self, fw_spec):
        job = AimsJob(aims_cmd=self['aims_cmd'], control=self['control'], structure=self['structure'],
                      basis_set=self['basis_set'], single_basis=self['single_basis'], basis_dir=self['basis_dir'],
                      output_file=self['aims_output'], metadata=self['rerun_metadata'])
        validators = [AimsConvergedValidator()]
        handlers = [AimsRelaxHandler(), FrozenJobErrorHandler()]
        c = Custodian(handlers=handlers, jobs=[job], validators=validators, max_errors=3)
        c.run()


@explicit_serialize
class AimsSingleBasis(Firework):
    def __init__(self, aims_cmd, control, structure, basis_set, basis_dir, metadata, name, parents=None):

        """
        Performs a single basis DFT run with the specified basis set (no auto additions)

        Args:
            aims_cmd: command to run aims i.e. srun aims
            control: control.in as list of lines
            structure: pymatgen structure to relax
            basis_set: basis set to use, directory name within basis_dir
            basis_dir: directory where the basis is located (light / tight directories)
            metadata: metadata to pass into both runs for identification
            name: name of the workflow
            parents: parents of this WF

        Returns:
            Workflow to insert into LaunchPad
        """
        t = list()
        t.append(RunAimsCustodian(aims_cmd=aims_cmd, control=control, structure=structure,
                                  basis_set=basis_set, basis_dir=basis_dir, single_basis=True,
                                  aims_output='run', rerun_metadata=metadata))

        super(AimsSingleBasis, self).__init__(t, parents=parents, name='{}-{}'.
                                              format(structure.composition.reduced_formula, name))


@explicit_serialize
class AimsRelaxLightTight(Firework):
    def __init__(self, aims_cmd, control, structure, basis_dir, metadata, name, parents=None):
        """
        Performs a light and tight relaxation for the given structure object

        Args:
            aims_cmd: command to run aims i.e. srun aims
            control: control.in as list of lines
            structure: pymatgen structure to relax
            basis_dir: directory where the basis is located (light / tight directories)
            metadata: metadata to pass into both runs for identification
            name: name of the workflow
            parents: parents of the WF

        Returns:
            Workflow to insert into LaunchPad
        """
        t = list()
        t.append(RunAimsCustodian(aims_cmd=aims_cmd, control=control, structure=structure,
                                  basis_set='light', basis_dir=basis_dir, single_basis=False,
                                  aims_output='run', rerun_metadata=metadata))

        super(AimsRelaxLightTight, self).__init__(t, parents=parents, name="{}-{}".
                                                  format('{} - {}'.join(structure.composition.reduced_formula), name))
