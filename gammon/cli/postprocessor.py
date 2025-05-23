import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ..constants import MOV, SWP, ADD, DEL, VOP, VOM


class PostProcessor:
    def __init__(self, outpath: Path, mu):
        if Path("PostProcess").exists():
            shutil.rmtree("PostProcess")
        self.folder = Path("PostProcess")
        self.folder.mkdir()
        self.mu = mu

        with open(outpath, encoding="utf-8") as f:
            lines = f.readlines()

        # Determine the dimension of data
        self.max_step, self.max_state = 0, 0
        for line in lines:
            if line.strip().startswith("-"):
                continue
            parts = line.strip().split()
            step, state = parts[:2]

            if int(step) > self.max_step:
                self.max_step = int(step)
            if int(state) > self.max_state:
                self.max_state = int(state)

        # Fill data with the line of outpath
        self.data = np.zeros([self.max_step, self.max_state+1, 4])

        op_map = {'MOV': MOV, 'SWP': SWP, 'ADD': ADD,
                  'DEL': DEL, 'VOP': VOP, 'VOM': VOM}
        for line in lines:
            if line.strip().startswith("-"):
                continue

            parts = line.strip().split()
            step = int(parts[0])
            state = int(parts[1])
            op = op_map.get(parts[2])
            is_accepted = int(parts[3])
            nH = int(parts[4])
            E = float(parts[5])
            self.data[step-1, state, 0] = op
            self.data[step-1, state, 1] = is_accepted
            self.data[step-1, state, 2] = nH
            self.data[step-1, state, 3] = E

        self.calc_ratios()

    def calc_ratios(self):
        """
        Calculate accepted, tried operations and ratios
        """
        op_map = {'MOV': MOV, 'SWP': SWP, 'ADD': ADD,
                  'DEL': DEL, 'VOP': VOP, 'VOM': VOM}
        self.operations = list(op_map.keys())
        op_indices = list(op_map.values())

        # Initialize storage for state-specific data
        self.state_results = []

        for state_idx, mu_value in enumerate(self.mu):
            # Extract data for the current state
            state_op_data = self.data[:, state_idx, 0]
            state_acceptance_data = self.data[:, state_idx, 1]

            # Calculate tried, accepted, and ratios
            tried = [np.sum(state_op_data == op_idx) for op_idx in op_indices]
            accepted = [np.sum((state_op_data == op_idx) &
                               (state_acceptance_data == 1))
                        for op_idx in op_indices]
            acc_ratios = [round(acc / tr * 100, 2) if tr > 0 else 0
                          for acc, tr in zip(accepted, tried)]
            tot_op = sum(tried)
            op_ratios = [round(tr / tot_op * 100, 2) if tot_op > 0 else 0
                         for tr in tried]

            # Store results for this state
            self.state_results.append({
                'state': state_idx,
                'mu': mu_value,
                'tried': tried,
                'accepted': accepted,
                'acc_ratios': acc_ratios,
                'op_ratios': op_ratios,
            })

    def do_graphE(self):
        for state in range(self.max_state+1):
            plt.plot(self.data[:, state, 3],
                     label=f'Mu={self.mu[int(state)]:.3f}')
        plt.xlabel('Step')
        plt.ylabel('Energy (eV)')
        plt.title('GCMC Energy')
        plt.legend(loc="upper right")
        plt.savefig(self.folder / "graphE.pdf")
        plt.clf()

    def do_graph_nH(self):
        for state in range(self.max_state+1):
            plt.plot(self.data[:, state, 2],
                     label=f'Mu={self.mu[int(state)]:.3f}')
        plt.xlabel('Step')
        plt.ylabel('nH')
        plt.title('GCMC absorbed H')
        plt.legend(loc="upper right")
        plt.savefig(self.folder / "graphnH.pdf")
        plt.clf()

    def acceptation_rate(self):
        """
        Write Operations.dat.
        """
        output_path = self.folder / 'Operations.dat'
        with open(output_path, 'w') as file:
            # Write the header
            file.write("Operation Analysis\n")
            file.write("=" * 40 + "\n")

            # Write state-specific results
            file.write("State-Specific Results:\n")
            file.write("=" * 40 + "\n")
            for state_result in self.state_results:
                file.write(f"State {state_result['state'] + 1}:\n")
                file.write(f"  Chemical Potential : {state_result['mu']}\n")
                file.write(f"{'Operation':<10}{'Accept (%)':<12}"
                           f"{'Accepted':<10}"
                           f"{'Tried':<10}{'Op (%)':<10}\n")
                file.write("-" * 40 + "\n")
                for op, acc_rate, accepted, tried, op_ratio in zip(
                    self.operations,
                    state_result['acc_ratios'],
                    state_result['accepted'],
                    state_result['tried'],
                    state_result['op_ratios'],
                ):
                    file.write(f"{op:<10}{acc_rate:<12}{accepted:<10}"
                               f"{tried:<10}{op_ratio:<10}\n")
                file.write("\n")

    def do_mean(self):
        """
        Calculate the mean of nH and E.
        Estimates the number of thermalisation steps
        """
        # Step 1: Estimate the thermalisation steps
        self.find_thermalisation()

        self.mean_nH_list = []
        self.mean_E_list = []
        self.wt_H2_list = []
        fn = self.folder / "absorption.dat"
        with open(fn, "w") as f:
            for state_idx, mu_value in enumerate(self.mu):
                # Thermalisation step for this state
                tt = self.thermalisation_points[state_idx]
                if tt is None:
                    print(f"State {state_idx + 1} (Mu = {mu_value}):"
                          f" Thermalisation not found.")
                    continue

                # Step 2: Calculate mean values using thermalisation step
                mean_nH = np.mean(self.data[tt:, state_idx, 2])
                mean_E = np.mean(self.data[tt:, state_idx, 3])

                # Step 3: Calculate Mtot, MH2, and wt% H2
                Mtot = mean_nH * 1.008 + 144.24 * 6 + 24.305 * 2 + 58.693 * 28
                MH2 = mean_nH * 1.008 / Mtot
                wt_H2 = MH2 * 100

                self.mean_nH_list.append(mean_nH)
                self.mean_E_list.append(mean_E)
                self.wt_H2_list.append(wt_H2)

                # Step 4: Print the results
                f.write(f"State {state_idx + 1} (Mu = {mu_value} eV/at):\n")
                f.write(f"  Thermalisation: {tt} steps\n")
                f.write(f"  Mean nH: {mean_nH}\n")
                f.write(f"  Mean Energy (Excluding Kin energy): {mean_E}\n")
                f.write(f"  Assuming Nd6Mg2Ni28H_x, {wt_H2:.2f} wt% H2\n")
                f.write("-" * 40 + "\n")

    def find_thermalisation(self):
        """
        Estimate the thermalisation point for each state.
        """
        jump = 50
        tol_percent = 0.02
        nstep = np.shape(self.data)[0]

        # Create an array of potential thermalisation steps
        therm_steps = np.arange(0, nstep // 2 + 1, jump)
        self.thermalisation_points = []

        for state_idx, mu_value in enumerate(self.mu):
            therm_point = None
            for therm in therm_steps:
                # Calculate the mean value after thermalisation
                mean_after_therm = self.mean_nH(therm, state_idx)
                curr_val = np.mean(self.data[therm:therm+jump, state_idx, 2])
                diff = abs(curr_val - mean_after_therm)

                # Division by zero is bad
                if mean_after_therm <= 0.01:
                    therm_point = therm
                    break

                todivide = abs(curr_val + mean_after_therm)/2
                if diff / todivide <= tol_percent:
                    therm_point = therm
                    break

            if therm_point is None:
                print(f"State {state_idx + 1}: Thermalisation not found."
                      f" Using nstep/2 = {therm_steps[-1]}")
                therm_point = therm_steps[-1]

            else:
                print(f"State {state_idx + 1}: Thermalisation at "
                      f"{therm_point}")
            self.thermalisation_points.append(therm_point)

    def mean_nH(self, thermalisation, state):
        return np.average(self.data[thermalisation:, state, 2])

    def mean_potE(self, thermalisation, state):
        return np.average(self.data[thermalisation:, state, 3])

    def do_abs_graph(self):
        """
        Generate a scatter of wt% H_2 w.r.t. mu
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.mu, self.wt_H2_list, color='blue',
                    label="wt% H2", s=50)
        plt.title("Hydrogen Weight Percentage vs. Chemical Potential")
        plt.xlabel("Chemical Potential (Mu) eV/at")
        plt.ylabel("wt% H2")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save the plot
        output_path = self.folder / "abs_graph.pdf"
        plt.savefig(output_path)
        plt.close()
