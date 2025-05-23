import pickle
import numpy as np

from ase import Atoms
from ase.io import read

from .constants import kB_eV, h_eV, mH_eV
from .constants import rng_seed

# move, swap, add Hydrogen, delete hydrogen, volume increase, volume decrease
from .constants import MOV, SWP, ADD, DEL, VOP, VOM


class OneState:
    """
    Deals with one state of GCMC
    """
    def __init__(self, proc_id, struct, calc, write_queue,
                 mu, T, buffer_size, result_fn, prob):
        # Passed object
        self.proc_id = proc_id
        self.mu = mu
        self.T = T
        self.calc = calc
        self.write_queue = write_queue
        self.struct = struct
        self.buffer_size = buffer_size
        self.result_fn = result_fn
        # Move, Swap Site, Add, Remove, Volume increase, Volume decrease
        self.prob = np.array(prob)

        # Quantity that will be useful
        self.beta = 1/(T * kB_eV)
        # Thermal DeBroglie Wavelength cubed in ang : (h^2 beta / 2 pi m)^3/2
        self.tdbw3 = ((h_eV**2 * self.beta) / (np.pi * 2 * mH_eV))**(3/2)

        # Simulation information
        self.tried = np.zeros(6)
        self.accepted = np.zeros(6)
        self.write_buffer = [[], []]
        self.step = 0

        # Prepare RNG
        self.rng = np.random.default_rng(rng_seed + self.proc_id)
        self.struct.set_rng(self.rng)

        # Preparation
        self.calc.prepare_calc(struct)

    def run(self, nstep):
        """
        Do nstep of GCMC
        """
        self.prepare_onestate()
        with np.errstate(over='ignore'):
            for _ in range(int(nstep)):
                self.step += 1
                op, accepted = self.one_step()
                self.sample(op, accepted)

        print(f"Finish from state{self.proc_id}")

    def prepare_onestate(self):
        """
        Prepare OneState before going into the simulation loop
        """
        # 1. Calculate the starting energy
        E = self.calc.get_potential_energy(crystal=self.struct.atoms,
                                           h_atoms=self.struct.h_atoms,
                                           mu=self.mu,
                                           proc_id=self.proc_id)
        self.struct.curr_E = E

    def one_step(self):
        """
        Try a GCMC move
        """
        r = self.rng.random()
        prob = self.calc_prob()
        for op in [MOV, SWP, ADD, DEL, VOP, VOM]:
            if r < prob[op]:
                break

        self.tried[op] += 1
        new_at, new_H = self.struct.try_operation(op)
        is_accepted = 0
        if new_H is not None:
            new_E = self.calc.get_potential_energy(crystal=new_at,
                                                   h_atoms=new_H,
                                                   mu=self.mu,
                                                   proc_id=self.proc_id)
            prefactor = self.calc_prefactor(op)
            is_accepted = self.metropolis(curr_E=self.struct.curr_E,
                                          new_E=new_E,
                                          prefactor=prefactor)
            if is_accepted:
                self.struct.atoms = new_at
                self.struct.h_atoms = new_H
                self.struct.curr_E = new_E
            self.accepted[op] += is_accepted
        return op, is_accepted

    def metropolis(self, curr_E, new_E, prefactor):
        """
        Use Metropolis algorithm to determine if a move is accepted
        P_acc = min[1, 1/L^3 * V/(N+1) exp(-beta (dE - mu))]
        P_del = min[1, L^3*N/V exp(-beta (dE - mu))]
        P_mov = min[1, exp(beta*dE)]
        P_swp = min[1, exp(beta*dE)]
        L is the thermal DeBroglie wavelength

        Input : Energy (eV) of the initial structure and the tried one
                Prefactor is either : L^3*N/V, 1/L^3*V/(N+1) or 1
        Output : If the move is accepted

        Warning : I'm doing min[1, (prefactor * np.exp)] which is not equal
                  to min[1, np.exp]*prefactor for np.exp > 1.
        """
        e1 = prefactor * np.exp(self.beta*(curr_E - new_E))
        if e1 > self.rng.random():
            return 1
        return 0

    def calc_prefactor(self, op):
        """
        Calculate the prefactor for the metropolis algorithm according to
        Add : 1/L^3 * V/(N+1) or 1
        Del : L^3*N/V
        Mov/Swp : 1

        Note : There is another factor corresponding to the add which is only
               done in an absorption site. It is calculated in calc_prob
        """
        if op == MOV or op == SWP or op == VOP or op == VOM:
            return 1

        V = self.struct.get_volume()
        N = self.struct.nH
        if op == DEL:
            return self.tdbw3 * N / V
        if op == ADD:
            return V / (self.tdbw3 * (N + 1))

    def calc_prob(self):
        """
        Compute the add operation probability based on available absorption
        volume and full structure volume. Also renormalize prob.
        """
        prob = self.prob.copy()
        abs_V = self.struct.get_absorption_volume()
        V = self.struct.get_volume()
        if self.struct.only_add_empty:  # Only insert in empty sites
            nsites = len(self.struct.abs_sites)
            abs_V = abs_V * ((nsites - self.struct.nH) / nsites)
        prob[2] = prob[2] * (abs_V/V)
        return np.cumsum(prob / np.sum(prob))

    def sample(self, operation, isaccepted):
        """
        Sample the system
        int : The id of the state that is sampled
        """
        # Write Step / Operation / isAccepted / curr_E / nH
        nH = self.struct.nH

        curr_E = self.struct.curr_E
        curr_E = f"{curr_E:16.12f}".rjust(16)
        dict_op = {MOV: "MOV", SWP: "SWP", ADD: "ADD",
                   DEL: "DEL", VOP: "VOP", VOM: "VOM"}
        s_op = dict_op.get(operation)

        line = (f"{self.step:<8} {self.proc_id:<5} {s_op:<4}" +
                f"{bool(isaccepted):<10} {nH:<5} {curr_E}\n")
        self.write_buffer[0].append(line)

        # self.write_buffer[1].append([struct.atoms, struct.h_atoms])

        if not self.step % self.buffer_size:
            at = self.struct.atoms
            if len(self.struct.h_atoms[0]) > 0:
                at_H = Atoms('H' * len(self.struct.h_atoms[0]),
                             cell=at.cell,
                             pbc=at.pbc,
                             scaled_positions=self.struct.h_atoms[0])
                at = at + at_H
            self.write_data(at)

    def write_data(self, at):
        """
        1. Create the dictionnary to write
        2. Send the dict to the write queue
        3. Clean up write_buffer
        """
        i = self.proc_id
        # 1. Prepare the dictionary for the data to write
        item = {
            "fn_out": self.result_fn / "GCMC.out",
            "fn_traj": self.result_fn / f"state{i}.traj",
            "fn_rng": self.result_fn / f"restart_state{i}.pkl",
            "outdata": self.write_buffer[0],
            "trajdata": at,
            "restart_data": {
                "h_state": self.struct.h_atoms,
                "rng_state": self.rng.bit_generator.state
            }
        }

        # 2. Send the dictionary to the writer thread
        self.write_queue.put(item)

        # 3. Clean up the write_buffer for the current state
        self.write_buffer = [[], []]

    def restart_simulation(self, pkl_file, traj_file,
                           tried, accepted, curr_E):
        """
        Put this state as if it never stopped
        1. Pop the H atoms from the last traj and set it as struct.atoms
        2. Get struct.h_atoms and rng_state from the pkl file
        3. Set
        """
        # 1. Set struct.atoms
        last = read(traj_file, index="-1")
        self.struct.atoms = last[[atom.index for atom in
                                  last if atom.symbol != 'H']]

        # 2. Read pkl
        with open(pkl_file, 'rb') as f:
            restart = pickle.load(f)
        self.struct.h_atoms = restart['h_state']

        self.rng.bit_generator.state = restart['rng_state']
        self.struct.set_rng(self.rng)

        # 3. Set simulation data
        self.tried = tried
        self.accepted = accepted
        self.step = int(sum(self.tried))

        # 4. Set curr_E
        E = self.calc.get_potential_energy(
                    crystal=self.struct.atoms,
                    h_atoms=self.struct.h_atoms,
                    mu=self.mu,
                    proc_id=self.proc_id)
        if abs(E - curr_E) > 1e-8:
            print("Energy discrepancy during the restart" +
                  f"{self.struct.curr_E}, {E}")
        self.struct.curr_E = E
