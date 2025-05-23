from typing import Optional
from pathlib import Path
import os
import shutil
import copy
import numpy as np
import multiprocessing

from .constants import rng_seed

# move, swap, add Hydrogen, delete hydrogen, volume increase, volume decrease
from .constants import MOV, SWP, ADD, DEL, VOP, VOM
from .structure import Structure
from .calculator import HarmonicCalc, Calculator
from .onestate import OneState
from .writer import Writer


def run_one_state(state, step):
    """
    The function that is called by multiprocessing.Process
    """
    state.run(step)


class GCMC:
    """
    Main class of the GCMC program
    """
    def __init__(self,
                 struct: Structure,
                 nstate: int,
                 calc: Calculator,
                 prob: list[int],
                 T: float,
                 mu=0,
                 result_fn: Path = Path("Results"),
                 buffer_size: int = 500):
        """
        struct : MockGCMC Structure
        The initial structure to do the simulation on

        nstate : int
        Number of simulation to run in parallel

        calc : Calculator
        Object to calculate the potential energy of the system

        prob : list of 6 float
        Probability to do in order MOV, SWP, ADD, DEL, VOP, VOM

        mu : float or list where len(mu)=nstate
        The chemical potential used for the structs

        buffer_size : int
        The number of step to skip between each writing operation
        This is useful even with writer because everytime I write to GCMC.out,
        I also write 1 rng_state and 1 atoms

        debug_mode : bool
        If we have debug logging
        """
        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)
        if len(prob) == 4:
            prob = np.concatenate([prob, [0., 0.]])
        assert abs(sum(prob) - 1) < 1e-8 and len(prob) == 6, f"prob={prob}"

        if isinstance(calc, HarmonicCalc):
            if sum(prob[-2:]) > 1e-8:
                raise NotImplementedError("No volume change with HarmonicCalc")
            if struct.max_crystal_mov > 0.:
                raise ValueError("HarmonicCalc cannot move crystal atom")
        self.nstate = nstate

        if isinstance(mu, float) or isinstance(mu, int):
            mu = [mu for _ in range(nstate)]
        else:
            if len(mu) != nstate:
                raise ValueError("len(mu) != nstate")
        self.mu = list(mu)  # If np.array, the print break PostProcess
        self.result_fn = result_fn
        self.writer = Writer(result_fn, struct, nstate)

        self.states = []
        for state in range(nstate):
            self.states.append(OneState(proc_id=state,
                                        struct=copy.deepcopy(struct),
                                        calc=copy.deepcopy(calc),
                                        write_queue=self.writer.write_queue,
                                        mu=mu[state],
                                        T=T,
                                        buffer_size=buffer_size,
                                        result_fn=result_fn,
                                        prob=prob))

    def restart_simulation(self):
        """
        Put the simulation in the state as if it never stopped
        1. Read the GCMC.out to get tried, accepted, curr_E
        2. Get the filename of traj and pkl
        3. Give these info to OneState and let it restart
        """
        # 1. Set Step, tried and accepted
        restart_tried = [np.zeros(6) for _ in range(self.nstate)]
        restart_accepted = [np.zeros(6) for _ in range(self.nstate)]
        restart_curr_E = [0 for _ in range(self.nstate)]
        op_mapping = {"MOV": MOV, "SWP": SWP, "ADD": ADD,
                      "DEL": DEL, "VOP": VOP, "VOM": VOM}
        with open(self.result_fn / "GCMC.out", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip().startswith("-") or len(line.strip()) == 0:
                continue

            # Read a line from GCMC.out
            # Step / State / Op / Accepted / nH / curr_E
            toread = line.strip().split()
            tmp_state, tmp_op = int(toread[1]), toread[2]
            tmp_accepted, _ = int(toread[3]), int(toread[4])
            tmp_curr_E = float(toread[5])

            # Add the data to the tables
            restart_tried[tmp_state][op_mapping[tmp_op]] += 1
            restart_accepted[tmp_state][op_mapping[tmp_op]] += tmp_accepted
            restart_curr_E[tmp_state] = tmp_curr_E

        # Restart every OneState
        for i in range(self.nstate):
            traj_file = self.result_fn / f"state{i}.traj"
            pkl_file = self.result_fn / f"restart_state{i}.pkl"
            self.states[i].restart_simulation(pkl_file=pkl_file,
                                              traj_file=traj_file,
                                              tried=restart_tried[i],
                                              accepted=restart_accepted[i],
                                              curr_E=restart_curr_E[i])

    def run(self,
            nstep: int,
            extend: bool = False,
            nproc: Optional[int] = None):
        """
        Do nstep of GCMC
        extend : If we extend a simulation or restart a new one
        """
        self.prepare_gcmc(int(nstep), nproc, extend)
        processes = []
        for state in self.states:
            proc = multiprocessing.Process(target=run_one_state,
                                           args=(state, nstep))
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()
        self.complete_sim()

    def prepare_gcmc(self, nstep, nproc, extend):
        """
        Prepare gcmc before calling run_state
        1. Look at the number of processor
        2. Create the result folder if necessary
        3. Start the writer object
        4. Restart simulation if necessary
        5. Write GCMC.params
        """
        # 1. Looking at nproc
        if nproc is None:
            nproc = os.cpu_count()
        if nproc > self.nstate:
            nproc = self.nstate
        elif nproc < self.nstate:
            print("Not Enough processor for the number of state\n" +
                  "This program is not optimized for that situation")
        print(f"Using {nproc} state processors and 1 writer thread.")

        # 2. Prepare result folder
        if Path(self.result_fn).exists():
            if not extend:
                shutil.rmtree(self.result_fn)
                Path(self.result_fn).mkdir()
        else:
            if extend:
                e = "Cannot extend simulation :\n"
                e += f"Result folder not found {self.result_fn}."
                raise RuntimeError(e)
            Path(self.result_fn).mkdir()

        # 3. Start the writer object
        self.writer.start_writer(self.states[0].struct, self.nstate)
        # 4. Restart simulation if extend
        if extend:
            self.restart_simulation()

        # 5. Write GCMC.params
        prob = self.states[0].prob
        s = f"Nstep : {nstep}\n"
        s += f"Nstate : {self.nstate}\n"
        s += f"Mu : {self.mu}\n"
        s += f"Temperature (K) : {self.states[0].T}\n"
        s += f"Probability [MOV, SWP, ADD, DEL, VOP, VOM] : {prob}\n"
        s += f"RNG seed: {rng_seed}\n"
        s += f"Number of processors : {nproc}\n"
        s += f"Structure : {self.states[0].struct}\n"
        s += f"{self.states[0].calc}"
        self.writer.write_params(s)

    def complete_sim(self):
        self.writer.stop_writer()
