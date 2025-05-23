import numpy as np

from gammon import GCMC, Structure
from gammon.calculator import HarmonicCalc
from gammon.cli import PostProcessor
from gammon.utilities import get_nd3_data

from ...conftest import find_data_path, total_nsites
from ... import context  # noqa


def test_analytical_nohessian(root):
    """
    Make sure mock_GCMC on an non-interacting, static lattice, no potential
    give the expected absorption <N> and <E> determined analytically.

    The system absorbs :
          A lot if E-mu >> 0
          Not at all if E-mu << 0
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    e, x, f = get_nd3_data()

    # 1. Initialize the object
    expected_nsites = total_nsites
    rcut = 0.2
    rcut_swit = 0.
    chem_pot = 0.01
    kB = 8.6173303e-5  # eV
    struct = Structure(struct_fn, x, site_rcut=rcut)
    assert len(struct.abs_sites.abs_sites) == total_nsites
    coef = [0., np.array([[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.]])]
    Edxdydz = [coef for _ in range(len(x))]
    calc = HarmonicCalc(coef_irrsites=Edxdydz,
                        rcut_swit=rcut_swit)

    nstate = 1
    prob = np.array([0.25, 0.25, 0.25, 0.25, 0., 0.])
    T = 300
    gcmc = GCMC(struct=struct, nstate=nstate, prob=prob,
                T=T, mu=chem_pot, calc=calc, buffer_size=1)

    # 2. Theoric expectation
    # <nH> = nsites / (1+e^{-beta*mu} tdbw3 / V_one_site)
    tdbw3 = gcmc.states[0].tdbw3
    V_one_site = 4/3 * np.pi * struct.abs_sites.abs_sites[0].rcut**3
    analytical_nH = (expected_nsites /
                     (1 + np.exp(-chem_pot/(kB*T)) * tdbw3 / V_one_site))
    analytical_E = -chem_pot * analytical_nH

    # 3. Run the simulation for 6000 steps
    gcmc.run(6000)

    # 4. Post_process <nH> and <E>
    thermalisation = 4000
    pp = PostProcessor("Results/GCMC.out", [0.01])
    sim_nH = pp.mean_nH(thermalisation, 0)
    sim_E = pp.mean_potE(thermalisation, 0)
    add_acc_ratios = pp.state_results[0]['acc_ratios'][2]
    del_acc_ratios = pp.state_results[0]['acc_ratios'][3]

    print("Simulated <nH>:", sim_nH)
    print("Analytical <nH>:", analytical_nH)
    print("Simulated <E>:", sim_E)
    print("Analytical <E>:", analytical_E)
    print("Add acceptance ratio :", add_acc_ratios)
    print("Del acceptance ratio :", del_acc_ratios)

    # Only 2000 step of production after a too small thermalisation
    # so we cannot expect a precision as if we would have done 50k
    # 20% may crash, will need to adjust the tol in that case
    tol = 0.2
    err_nH = abs(analytical_nH - sim_nH) / (analytical_nH + sim_nH)
    err_E = abs(analytical_E - sim_E) / (analytical_E + sim_E)

    assert abs(add_acc_ratios - 100) < 1e-5  # All add should be accepted
    assert err_nH < tol
    assert err_E < tol
