import numpy as np
import copy

from gammon import GCMC, Structure
from gammon.calculator import HarmonicCalc
from gammon.utilities import get_nd3_data
from ...conftest import find_data_path, total_nsites
from ... import context  # noqa

rcut = 0.2


def test_metropolis(root):
    """
    Make sure the metropolis algorithm give the correct distribution
    in energy
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"

    # 1. Define GCMC object
    rcut_swit = 2.1
    chem_pot = 0
    e, x, ifc = get_nd3_data()
    irr_sites = x
    harmonic_coef = [[e[i], ifc[i]] for i in range(len(e))]

    struct = Structure(struct_fn, irr_sites, site_rcut=rcut)
    assert len(struct.abs_sites.abs_sites) == total_nsites
    calc = HarmonicCalc(coef_irrsites=harmonic_coef,
                        rcut_swit=rcut_swit)

    nstate = 1
    prob = np.array([0.25, 0.25, 0.25, 0.25, 0., 0.])
    T = 300
    gcmc = GCMC(struct=struct, nstate=nstate, prob=prob,
                T=T, mu=chem_pot, calc=calc, buffer_size=1)
    gcmc.rngs = [np.random.default_rng()]

    print(gcmc.states)
    print(gcmc.states[0])
    print(gcmc.states[0].beta)
    # 2. Test metropolis
    e_test = [2e-1, 5e-2, 2e-4]
    ntry = 1e4
    for e in e_test:
        naccepted = 0
        for n in range(int(ntry)):
            naccepted += gcmc.states[0].metropolis(
                    curr_E=0, new_E=e, prefactor=1)

        theo = np.exp(-gcmc.states[0].beta * e)
        assert abs(naccepted/ntry - theo) < 1e-2


def test_variable_prob(root):
    """
    Make sure the probability is correctly computed with different
    absorption site available and hydrogen atom
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"

    # 1. Define GCMC object
    rcut_swit = 2.1
    chem_pot = 0
    e, x, ifc = get_nd3_data()
    irr_sites = x
    harmonic_coef = [[e[i], ifc[i]] for i in range(len(e))]

    struct = Structure(struct_fn, irr_sites, site_rcut=rcut)
    assert len(struct.abs_sites.abs_sites) == total_nsites
    calc = HarmonicCalc(coef_irrsites=harmonic_coef,
                        rcut_swit=rcut_swit)

    nstate = 1
    prob = np.array([0.25, 0.25, 0.25, 0.25, 0., 0.])
    T = 300
    gcmc = GCMC(struct=struct, nstate=nstate, prob=prob,
                T=T, mu=chem_pot, calc=calc, buffer_size=1)

    # 2. Redefine struct in gcmc to have 0, 1 and 96 H atom
    struct_0 = copy.copy(struct)
    xred_h1 = [struct.abs_sites.abs_sites[0].xred]
    struct_1 = add_h_atoms(copy.copy(struct), xred_h1, 0)
    struct_96 = copy.copy(struct)
    for idx, site in enumerate(struct.abs_sites.abs_sites):
        if idx % 2 == 1:
            h_xred = struct.abs_sites.abs_sites[idx]
            struct_96 = add_h_atoms(struct_96, h_xred, idx)
    gcmc.structs = [struct_0, struct_1, struct_96]

    assert gcmc.structs[0].nH == 0
    assert gcmc.structs[1].nH == 1
    assert gcmc.structs[2].nH == len(struct.abs_sites.abs_sites)/2

    # 3. Test calc_prob
    V_one_site = gcmc.structs[0].get_absorption_volume() / total_nsites
    V = gcmc.structs[0].get_volume()

    for i in range(3):
        manual = 0.25 * (V_one_site * (total_nsites-gcmc.structs[i].nH) / V)
        estimated = manual / (0.75+manual)
        prob = gcmc.states[0].calc_prob()[2]
        assert estimated-prob < 1e-6


def add_h_atoms(struct, h_xred, idx):
    if struct.nH != 0:
        struct.h_atoms = [np.vstack([struct.h_atoms[0], h_xred]),
                          np.append(struct.h_atoms[1], idx)]
    else:
        struct.h_atoms = [np.array([h_xred]), np.array([idx])]

    return struct
