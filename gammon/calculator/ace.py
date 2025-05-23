import os
from pathlib import Path
import shutil
import warnings

from .lammps import LammpsCalc

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove GPU warning for tf
import tensorflow as tf  # noqa


try:
    import pyace  # noqa
    from pyace.generalfit import GeneralACEFit  # noqa
    from pyace import create_multispecies_basis_config  # noqa
    from pyace.metrics_aggregator import MetricsAggregator  # noqa
except ImportError:
    pass


class AceCalc(LammpsCalc):
    def __init__(self,
                 model_fn: Path,
                 lmp_cmd: Path = Path("lmp")):
        """
        model_fn : Path
          The absolute path to the ace_model (*.yace)
        """
        if not str(model_fn).endswith("yace"):
            e = "You need a yace file : Maybe pace_yaml2yace ?"
            raise ValueError(e)

        with open(model_fn, "r") as f:
            line = f.readline()
            typat = line.split("[")[1].split("]")[0].replace(",", "").strip()

        pair_style = "pair_style pace"
        pair_coeff = f"pair_coeff * * {model_fn} {typat}"

        super().__init__(pair_style, pair_coeff,
                         files=[model_fn],
                         specorder=typat)

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

        h_atoms: list
         The info of the h_atoms stored in Structure

        mu: float
        The chemical potential of H_2/2 (eV/at)
        """
        to_evaluate = self.create_atoms(crystal, h_atoms)
        E, _ = self.calculate(atoms=to_evaluate,
                              properties=['energy'],
                              proc_id=proc_id)
        return E - mu * len(h_atoms[1])

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
        s = "Ace Calculator\n"
        s += f"Model file : {self.model_fn}\n"
        return s
