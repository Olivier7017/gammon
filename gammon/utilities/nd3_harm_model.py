import numpy as np
from pathlib import Path


def get_nd3_data(quantum_corr=True, hse_corr=True):
    """
    Returns Nd3MgNi14-2H absorption sites information :
      e0 (meV)
      x (Reduced coordinates)
      ifc (eV/ang**2)
      quantum_corr : If we add the quantum correction (300 K) in e0
      hse_corr : If we add the hse06 correction in e0
    """
    # Else python will load it from your current dir
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "nd3_data"

    ifcs = np.load(data_dir / "nd3_ifc.npy")
    xreds = np.load(data_dir / "nd3_xred.npy")
    e0s = np.load(data_dir / "nd3_e0s.npy")

    if quantum_corr:
        vibes = np.load(data_dir / "nd3_vibe.npy")
        e0s += vibes
    if hse_corr:
        hse = np.load(data_dir / "nd3_hse.npy")
        e0s += hse

    return e0s, xreds, ifcs
