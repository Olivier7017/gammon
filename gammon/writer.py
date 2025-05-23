import threading
import multiprocessing
import time
from pathlib import Path
from datetime import datetime
import pickle

from ase.io.trajectory import TrajectoryWriter

from .utilities import make_logo


class Writer:
    def __init__(self, result_fn, struct, nstate):
        """
        Class that writes in files using a thread
        """
        self.result_fn = result_fn
        self.write_queue = multiprocessing.Queue(maxsize=2000)

    def start_writer(self, struct, nstate):
        """
        Start the writer object and go into its loop
        """
        # Start the actual writing loop
        self.create_files(struct, nstate)
        self.writer = threading.Thread(target=self._writer_function,
                                       daemon=True)
        self.writer.start()

    def create_files(self, struct, nstate):
        # GCMC.out
        self.start_time = datetime.now()
        if not Path(self.result_fn / "GCMC.out").exists():
            time = self.start_time.strftime(
                    "-Starting time : %a %d %b %Y at %Hh%M")
            with open(self.result_fn / "GCMC.out", "a", encoding="utf-8") as f:
                f.write(make_logo())
                f.write(time + "\n")
                f.write(f"-{'Step':<7} {'State':<5} {'Op':<4} " +
                        f"{'isAccepted':<10} {'nH':<5} " +
                        f"{'curr_E(meV)':<16}\n")

        # state.traj
        for i in range(nstate):
            fn_traj = self.result_fn / f"state{i}.traj"
            traj_writer = TrajectoryWriter(fn_traj, mode='w')
            traj_writer.write(struct.atoms)
            traj_writer.close()

    def stop_writer(self):
        end_time = datetime.now()
        elapsed_time = str(end_time - self.start_time)
        with open(self.result_fn / "GCMC.out", "a", encoding="utf-8") as f:
            f.write(f"-Simulation time : {elapsed_time}\n")
            f.write("-This was Gammon by Olivier Nadeau (c)\n")

        if self.writer:
            self.write_queue.put(None)  # Send stop signal
            self.writer.join()  # Wait for writer to finish

    def _writer_function(self):
        """
        The function in which the writing thread loop
        """
        while True:
            if self.write_queue.empty():  # No busy waiting
                time.sleep(0.2)  # 200 ms

            item = self.write_queue.get()

            if item is None:  # Program termination
                break

            self._write_files(item)  # Actual writing

    def _write_files(self, item):
        """
        Write data to GCMC.out, stateX.traj, rng_stateX.traj
        """
        # Write GCMC.out data
        with open(item["fn_out"], "a", encoding="utf-8") as f_out:
            for line in item["outdata"]:
                f_out.write(line)

        # Write trajectory data
        traj_writer = TrajectoryWriter(item["fn_traj"], mode='a')
        traj_writer.write(item["trajdata"])
        traj_writer.close()

        # Write RNG data
        with open(item["fn_rng"], "wb") as f_rng:
            pickle.dump(item["restart_data"], f_rng)

    def write_params(self, s):
        with open(self.result_fn / "GCMC.params", "w", encoding="utf-8") as f:
            f.write(s)

    def _warn_queue_size(self):
        """
        Warn if queue > 75% of its max size
        """
        queue_size = self.write_queue.qsize()
        threshold = self.write_queue._maxsize * 0.75
        if queue_size > threshold:
            msg = f"Warning: write_queue size is {queue_size},"
            msg += f" exceeding 75% of the max size ({threshold})."
            print(msg)
