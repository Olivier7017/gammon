from abc import ABC, abstractmethod


class Calculator(ABC):
    @abstractmethod
    def get_potential_energy(self, crystal, h_atoms,
                             mu=0, proc_id=0) -> float:
        pass
