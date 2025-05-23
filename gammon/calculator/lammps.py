import subprocess
import shutil
from pathlib import Path
import numpy as np

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_masses, chemical_symbols
from ase.io.lammpsdata import write_lammps_data


class LammpsCalc(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self,
                 pair_style,
                 pair_coeff,
                 files,
                 command="lmp_serial",
                 specorder=None, **kwargs):
        """
        specorder : str or list
          The elements in order of apparition in the potential file
        """
        super().__init__(**kwargs)
        if isinstance(pair_coeff, str):
            pair_coeff = [pair_coeff]
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
        self.files = files
        self.command = command

        self.lmp_folder = Path("LAMMPS")
        if self.lmp_folder.exists():
            shutil.rmtree(self.lmp_folder)
        self.lmp_folder.mkdir()
        for file in files:
            dest = self.lmp_folder / file.name
            if not dest.exists():
                if file.is_dir():
                    shutil.copytree(file, dest)
                else:
                    shutil.copy(file, dest)
        if specorder is not None and isinstance(specorder, str):
            specorder = specorder.split()
        self.specorder = specorder
        if self.specorder is None:
            for line in self.pair_coeff:
                if "mliap" in line:
                    self.specorder = line.split()[4:]
                    break

        if self.specorder is None:
            raise NotImplementedError("self.specorder is None")

    def create_atoms(self, crystal, h_atoms):
        """
        Create an ase.Atoms containing the cristalline structure from crystal
        and the hydrogen atoms from h_atoms

        crystal: ase.atoms
         The cristalline structure to work with

        h_atoms: The list inside Structure
         Contains the positions of the atoms
        """
        nH = len(h_atoms[1])
        if nH == 0:
            return crystal
        ase_h = Atoms(symbols='H' * nH,
                      scaled_positions=h_atoms[0],
                      cell=crystal.cell,
                      pbc=crystal.pbc)
        atoms = crystal+ase_h
        return atoms

    def get_potential_energy(self,
                             crystal: Atoms,
                             mu: float = 0,
                             proc_id: int = 0,
                             force_consistent=True):
        """
        Such that I can use get_potential_energy from ASE and other utility
        """
        E, _ = self.calculate(atoms=crystal,
                              properties=['energy'],
                              proc_id=proc_id)
        return E

    def calculate(self,
                  atoms: Atoms,
                  properties: list = ['energy', 'forces'],
                  system_changes: list = None,
                  proc_id: int = 0):
        # System changes is there for ase
        super().calculate(atoms, properties)
        input_fn = self.lmp_folder.absolute() / f"lammps_input{proc_id}.in"
        self.write_lammps_input(atoms,
                                input_fn,
                                properties)
        self._run_lammps(input_fn)
        e, f = self._read_lammps_output(input_fn,
                                        properties=properties)
        self._clean(input_fn)
        self.results['energy'] = e
        self.results['forces'] = f
        return e, f

    def get_forces(self, atoms: Atoms):
        e, f = self.calculate(atoms, properties=['energy', 'forces'])
        return f

    def write_lammps_input(self, atoms, input_fn, properties=['energy']):
        # Define the atoms data file path
        atoms_fn = input_fn.with_suffix('.atoms')
        log_fn = input_fn.with_suffix('.log')

        ismace = "mace" in self.pair_style
        smass, ssymbol = "", ""
        for i, symbol in enumerate(self.specorder):
            idx = chemical_symbols.index(symbol)
            smass += f"mass {i+1} {atomic_masses[idx]}\n"
            ssymbol += f"group {symbol} type {i+1}\n"

        write_lammps_data(atoms_fn,
                          atoms,
                          specorder=self.specorder)

        # Construct the initialization section
        init_section = [
            "# Lammps input created by mock-GCMC",
            "##########################",
            "#     Initialization     #",
            "##########################",
            "units metal"]
        if ismace:
            init_section += [
                "newton on\n",
                "atom_modify map yes\n",
            ]

        init_section += [
            "boundary p p p",
            "atom_style atomic",
            f"read_data {Path(atoms_fn).name}"
        ]
        init_section += [smass, ssymbol]
        if ismace:
            # Else GCMC may exceed the max number of neighbor
            init_section += ["neigh_modify one 4000 page 200000\n"]
        init_section = "\n".join(init_section)

        # Construct the interaction section
        interaction_section = [
            "#######################",
            "#     Interaction     #",
            "#######################",
            f"{self.pair_style}"
        ]
        interaction_section += self.pair_coeff
        interaction_section = "\n".join(interaction_section)

        # Construct the thermostat section
        thermostat_section = [
            "######################",
            "#     Thermostat     #",
            "######################",
            "timestep 0.001",
            "fix ff all store/force",
            "fix f1 all nvt temp 300 300 $(100*dt)",
            "velocity        all create 300.0 1234",
            "fix fcm all recenter INIT INIT INIT\n",
        ]
        if "forces" in properties:
            conf_fn = input_fn.with_suffix('.forces')
            thermostat_section += [f"dump last all custom 1 {conf_fn} " +
                                   "id type xu yu zu vx vy vz " +
                                   "fx fy fz element"]
        thermostat_section += [f"log {log_fn}", "run 0"]
        thermostat_section = "\n".join(thermostat_section)
        input_content = (init_section + "\n\n" + interaction_section +
                         "\n\n" + thermostat_section)
        with open(input_fn, 'w') as f:
            f.write(input_content)
        return input_fn

    def _run_lammps(self, input_fn):
        cmd = f"{self.command} -in {str(input_fn.absolute())}"
        with open(input_fn.with_suffix('.out'), 'w') as f:
            subprocess.run(cmd,
                           stdout=f,
                           stderr=f,
                           cwd=str(self.lmp_folder),
                           shell=True)

    def _read_lammps_output(self,
                            input_fn,
                            properties=["energy", "forces"]):
        log_fn = input_fn.with_suffix('.log')
        conf_fn = input_fn.with_suffix('.forces')
        energy = None
        forces = None

        # Read Energy
        with open(log_fn, 'r') as f:
            for line in f:
                if "E_pair" in line:
                    index = line.split().index("E_pair")
                    next_line = next(f).strip()
                    energy = float(next_line.split()[index])

        if "forces" in properties:
            parse = False
            forces = []
            fx_idx, fy_idx, fz_idx = 0, 0, 0
            with open(conf_fn) as f:
                for line in f:
                    if parse:
                        parts = line.split()
                        fx = float(parts[fx_idx])
                        fy = float(parts[fy_idx])
                        fz = float(parts[fz_idx])
                        forces.append([fx, fy, fz])

                    if line.startswith("ITEM: ATOMS"):
                        parts = line.split()
                        fx_idx = parts.index("fx")-2
                        fy_idx = parts.index("fy")-2
                        fz_idx = parts.index("fz")-2
                        parse = True
            forces = np.array(forces)
        return energy, forces

    def _clean(self, input_fn):
        log_fn = input_fn.with_suffix('.log')
        conf_fn = input_fn.with_suffix('.forces')
        Path(log_fn).unlink()
        if Path(conf_fn).exists():
            Path(conf_fn).unlink()

    def _parse_forces(self, file):
        # Parse forces from the log file
        forces = []
        for line in file:
            if line.strip() == '':
                continue
            forces.append([float(x) for x in line.split()])
        return forces
