import os
import subprocess

total_nsites = 138


def find_data_path(root):
    """
    Navigate upwards to find data folder
    """
    datapath = root.absolute()
    while not os.path.exists(datapath / "data"):
        datapath = datapath.parent
        if datapath == datapath.parent:  # Reached the root directory
            r = root.absolute()
            e = f"No 'data' folder found in any parent directories of: {r}"
            raise ValueError(e)

    return datapath / "data"


def has_lammps():
    try:
        subprocess.check_output(["lmp_serial", "-h"],
                                stderr=subprocess.STDOUT, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def has_lammps_mace():
    try:
        output = subprocess.check_output(["lmp_serial", "-h"],
                                         stderr=subprocess.STDOUT, text=True)
        return "mace" in output.lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def has_lammps_pace():
    try:
        output = subprocess.check_output(["lmp_serial", "-h"],
                                         stderr=subprocess.STDOUT, text=True)
        return " pace" in output.lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
