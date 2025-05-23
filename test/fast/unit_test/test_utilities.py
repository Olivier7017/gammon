import numpy as np

from ase.io import read
from ase.build import make_supercell

from gammon.utilities import find_multiplicity

from ...conftest import find_data_path
from ... import context  # noqa


def test_fcc(root):
    """
    Test on gammon.utilities: find_multiplicity
    using FCC conventional cell.

    These functions don't work on every possible case but it is good enough
    for my current application. Feel free to improve it if needed
    """
    datapath = find_data_path(root)
    prim_fn = datapath / "cubic_test_prim.cif"
    conv_fn = datapath / "cubic_test_conv.cif"

    prim = read(prim_fn)
    conv = read(conv_fn)

    mult1 = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    mult2 = np.array([[-3, 3, 3], [3, -3, 3], [3, 3, -3]])
    mult3 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    mult4 = mult2

    prim1 = make_supercell(prim, mult1)
    prim2 = make_supercell(prim, mult2)
    conv3 = make_supercell(conv, mult3)

    m1 = find_multiplicity(prim, prim1)
    m2 = find_multiplicity(prim, prim2)
    m3 = find_multiplicity(prim, conv)
    m4 = find_multiplicity(prim, conv3)

    if not np.allclose(mult1, m1):
        raise ValueError("Mult1 failed")
    if not np.allclose(mult2, m2):
        raise ValueError("Mult2 failed")
    if not np.allclose(mult1, m3):
        raise ValueError("Mult3 failed")
    if not np.allclose(mult4, m4):
        raise ValueError("Mult4 failed")
