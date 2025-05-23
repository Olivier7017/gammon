import numpy as np

from gammon import GCMC, Structure
from gammon.calculator import HarmonicCalc
from gammon.constants import h_eV, mH_eV
from gammon.cli import PostProcessor
from gammon.utilities import get_nd3_data

from ...conftest import find_data_path, total_nsites
from ... import context  # noqa


def test_analytical_hessian(root):
    """
    Make sure mock_GCMC on an non-interacting, static lattice with an
    harmonic potential give the expected absorption <N> and <E>
    determined analytically.
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    e0, x, ifc = get_nd3_data()

    # 1. Define GCMC object
    rcut = 0.25
    rcut_swit = 0.
    chem_pot = 0.0875

    struct = Structure(struct_fn, x, site_rcut=rcut)
    assert len(struct.abs_sites.abs_sites) == total_nsites
    coefs = [[e0[i], ifc[i]] for i in range(len(e0))]
    calc = HarmonicCalc(coef_irrsites=coefs,
                        rcut_swit=rcut_swit)

    nstate = 1
    T = 300

    # Do the real simulation
    prob = np.array([0.25, 0.25, 0.25, 0.25, 0., 0.])
    gcmc = GCMC(struct=struct, nstate=nstate, prob=prob,
                T=T, mu=chem_pot, calc=calc, buffer_size=100)

    # 2. An error of about 0.5 hydrogen.
    gcmc.run(100000)  # Should run 5e+5 to get better results
    thermalisation = 40000
    pp = PostProcessor("Results/GCMC.out", [0.01])
    sim_nH = pp.mean_nH(thermalisation, 0)
    sim_E = pp.mean_potE(thermalisation, 0)

    # 3. Compute the expected values
    E_theo, nH_theo = 0, 0
    beta = gcmc.states[0].beta
    for site in gcmc.states[0].calc.all_sites:
        # Defining some variable I'll need for later
        IFC = site[2]
        klx, kly, klz = np.linalg.eig(IFC)[0]
        E_abs = site[1]

        term1 = (2*np.pi / (beta*h_eV))**3
        term2 = np.sqrt(mH_eV**(3) / (klx * kly * klz))
        term3 = np.exp(-beta * (E_abs - chem_pot))
        xl = term1 * term2 * term3
        nH_theo += xl / (1+xl)
        # px, py, pz are not degree of freedom. They do not contribute to E
        E_theo += (xl/(1+xl)) * (E_abs - chem_pot + 3/(2*beta))

    # 4. Assert to make sure everything is ok
    print(f"nH (sim, theo): ({sim_nH}, {nH_theo})")
    print(f"E (sim, theo): ({sim_E}, {E_theo})")
    assert abs(sim_nH - nH_theo) < 0.5, f"Sim/Expected <N>: {sim_nH} {nH_theo}"
    assert abs(sim_E - E_theo) < 0.5, f"Sim/Expected <E>: {sim_E} {E_theo} eV"
