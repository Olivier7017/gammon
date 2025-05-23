import numpy as np

from gammon import Structure
from gammon.calculator import HarmonicCalc
from ...conftest import find_data_path
from ... import context  # noqa


def test_harmonic(root):
    """
    Make sure the energy calculated by interaction_energy of HarmonicCalc is
    what I expect it to be for the simplest case and no symmetry
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "cubic_test_conv.cif"

    # Prepare the parameters
    rcut = 0.15
    rcut_swit = 1
    irr_sites = [[0.5, 0.5, 0.5]]
    harmonic_coef = [[15, np.array([[4.5, 0., 0.],
                                    [0., 4.5, 0.],
                                    [0., 0., 4.5]])]]

    # Prepare the object
    struct = Structure(struct_fn,
                       irr_sites,
                       site_rcut=rcut)
    calc = HarmonicCalc(coef_irrsites=harmonic_coef,
                        rcut_swit=rcut_swit)
    calc.prepare_calc(struct)

    cell = struct.atoms.cell
    assert np.all(cell.angles() == 90.)
    h_atoms = [np.array([[0.2, 0.5, 0.38]]), [0]]
    h_atoms_xang = h_atoms[0][0]*cell.lengths()
    sites_xang = np.array(irr_sites[0])*cell.lengths()

    auto_E = calc.get_interaction_energy(struct.atoms, h_atoms)

    d = np.linalg.norm(sites_xang - h_atoms_xang)
    manual_E = d**2 * 1/2 * harmonic_coef[0][1] + harmonic_coef[0][0]

    cell = struct.atoms.get_cell()
    assert [np.sum(cell[i]) == cell[i, i] for i in range(3)]  # Diagonal
    assert [np.allclose(cell[i, i], cell[0, 0]) for i in range(3)]  # Cubic
    for i in range(3):
        assert manual_E[i][i] - auto_E < 1e-6  # Actual test
