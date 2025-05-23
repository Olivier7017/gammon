from pathlib import Path
import numpy as np

from ase.atoms import Atoms

from .lammps import LammpsCalc


class SnapCalc(LammpsCalc):
    def __init__(self,
                 model_fn: Path,
                 descriptor_fn: Path,
                 pair_style: str,
                 pair_coeff: list,
                 lmp_cmd: Path = Path("lmp")):
        """
        model_fn : Path
          The absolute path to the snap *.model file
        """
        super().__init__(pair_style, pair_coeff,
                         files=[model_fn, descriptor_fn])
        self.lmp_cmd = lmp_cmd
        self.lmp_folder = Path('LAMMPS').absolute()

        self.model_fn = model_fn
        self.descriptor_fn = descriptor_fn
        self.crystal_E = None

    def prepare_calc(self, struct):
        """
        """
        pass

    # @timing_decorator
    def get_potential_energy(self,
                             crystal: Atoms,
                             h_atoms: np.array,
                             mu: float = 0,
                             proc_id: int = 0):
        """
        Return the energy of this structure in eV

        crystal: ase.Atoms
         The absorbant without hydrogen

        h_atoms: list
         The info of the h_atoms stored in Structure

        mu: float
        The chemical potential of H_2/2 (eV/at)
        """
        to_evaluate = self.create_atoms(crystal, h_atoms)
        E, _ = self.calculate(atoms=to_evaluate,
                              properties=['energy'],
                              proc_id=proc_id)
        if self.crystal_E is None:
            self.crystal_E, _ = self.calculate(atoms=crystal,
                                               properties=['energy'],
                                               proc_id=proc_id)
        return E - mu * len(h_atoms[1]) - self.crystal_E

    def __str__(self):
        s = "Snap Calculator\n"
        s += f"Model file : {self.model_fn}\n"
        s += f"Descriptor file : {self.descriptor_fn}\n"
        return s
