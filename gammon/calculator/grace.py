from pathlib import Path
import shutil
import subprocess

from ase.atoms import Atoms

from .lammps import LammpsCalc


class GraceCalc(LammpsCalc):
    def __init__(self,
                 model_fn: Path,
                 specorder: str,
                 complexity: str,
                 lmp_cmd: Path = Path("lmp")):
        """
        Calculator using grace/fs model calling Lammps

        model_fn : Path
          The absolute path to the mace_model

        specorder : str
          The order and type of atom that should appear e.g. "Nd Mg Ni H"

        complexity : str
          Complexity of the grace model : "FS", "1L" or "2L"
        """
        self.complexity = complexity

        if complexity == "FS":
            pair_style = "pair_style grace/fs\n"
        else:
            pair_style = "pair_style grace"

        pair_coeff = f"pair_coeff * * {model_fn} {specorder}"
        super().__init__(pair_style, pair_coeff,
                         files=[model_fn],
                         specorder=specorder,
                         command=lmp_cmd)

        self.model_fn = model_fn
        self.lmp_cmd = lmp_cmd
        self.lmp_folder = Path('LAMMPS').absolute()
        if self.lmp_folder.exists():
            shutil.rmtree(self.lmp_folder)
        self.lmp_folder.mkdir()

    def prepare_calc(self, struct):
        """
        """
        pass

    # @timing_decorator
    def get_potential_energy(self, crystal, h_atoms,
                             mu=0, proc_id=0):
        """
        Return the energy of this structure in eV

        crystal: ase.Atoms
         The absorbant without hydrogen

        h_atoms: The list inside Structure
         Contains the positions of the atoms (xred, idx)

        mu: float
        The chemical potential of H_2/2 (eV/at)
        """
        to_evaluate = self.create_atoms(crystal, h_atoms)
        E, _ = self.calculate(atoms=to_evaluate,
                              properties=['energy'],
                              proc_id=proc_id)
        return E - mu * len(h_atoms[1])

    def create_atoms(self, crystal, h_atoms):
        """
        Create an ase.Atoms containing the cristalline structure from crystal
        and the hydrogen atoms from h_atoms

        crystal: ase.atoms
         The cristalline structure to work with

        h_atoms: list
         The info of the h_atoms stored in Structure (xred, idx)
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

    def get_interaction_energy_LAMMPS(self, atoms, proc_id=0):
        """
        Evaluate the energy of this atoms in eV

        atoms: ase.atoms
         The atoms object to evaluate
        """
        input_fn = self.lmp_folder / f"lammps_input{proc_id}.in"
        self.write_lammps_input(atoms, input_fn)
        self.launch_calc(input_fn)
        return self.read_output(input_fn.with_suffix('.log'))

    def launch_calc(self, input_fn):
        cmd = [self.lmp_cmd, "-in", input_fn]
        with open("/dev/null") as f:
            process = subprocess.Popen(cmd, stdout=f, cwd=self.lmp_folder)
            process = process.wait()

    def read_output(self, log_fn):
        """
        log_fn: Path
            The filename of the LAMMPS log file.
        """
        with open(log_fn, 'r') as log_file:
            for line in log_file:
                if "TotEng" in line:
                    # Read the next line which contains the actual value
                    next_line = next(log_file)
                    return float(next_line.split()[2])
        raise ValueError(f"TotEng not found in {log_fn}")

    def __str__(self):
        s = "Grace Calculator\n"
        s += f"Model file : {self.model_fn}\n"
        return s
