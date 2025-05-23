import numpy as np

from gammon import Structure, GCMC
from gammon.calculator import HarmonicCalc

from ...conftest import find_data_path
from ... import context  # noqa


def test_basis_symmetry(root):
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"

    rcut = 0.05
    rcut_swit = 1

    # Prepare the object
    irr_site_list = [[2/3, 2/3, 0.88404693872], [0.123, 0.456, 0.789],
                     [0.032956852013, 0.26647842601, 0.08404693872]]  # xred
    dr_list = [[0.001, 0., 0.], [0., 0.21, 0.],
               [0.0128321, 0.00931282, 0.33128291]]  # xang

    for irr_site, dr in zip(irr_site_list, dr_list):
        abs_sites = np.array([irr_site])
        harmonic_coef = [[0.001, np.array([[0.1, 0., 0.],
                                           [0., 0.2, 0.],
                                           [0., 0., 12.]])]]
        struct = Structure(struct_fn=struct_fn,
                           primxred_irrsites=abs_sites,
                           site_rcut=rcut)
        calc = HarmonicCalc(coef_irrsites=harmonic_coef,
                            rcut_swit=rcut_swit)
        nstate = 1
        chem_pot = [0.]
        T = 300
        prob = [0.5, 0.2, 0.15, 0.15, 0., 0.]
        gcmc = GCMC(struct=struct, nstate=nstate, calc=calc,
                    prob=prob, mu=chem_pot, T=T, buffer_size=1)

        allsite = [s.xred for s in struct.abs_sites.abs_sites]
        dr = struct.atoms.cell.scaled_positions(dr)  # xred

        # Site 1 : angle = 0 deg
        dr1 = dr
        site1 = np.array(irr_site)
        idx1 = np.where(np.all(np.isclose(allsite, site1), axis=1))[0][0]
        xred_h1 = np.array(struct.abs_sites.abs_sites[idx1].xred + dr1)
        h_atoms1 = [np.array([xred_h1]), np.array([idx1])]
        E1 = gcmc.states[0].calc.get_interaction_energy(struct.atoms,
                                                        h_atoms1)

        # Site 2 : angle = 120 deg,
        # Apply the rotation in real space
        rot120 = np.array([[-1/2, -np.sqrt(3)/2, 0],
                           [np.sqrt(3)/2, -1/2, 0],
                           [0, 0, 1]])
        dr2 = struct.atoms.cell.scaled_positions(
                np.dot(rot120, struct.atoms.cell.cartesian_positions(dr1)))
        site2 = struct.atoms.cell.scaled_positions(
                np.dot(rot120, struct.atoms.cell.cartesian_positions(site1)))
        site2 = np.mod(site2, 1.)
        site2 = np.where(abs(site2 - 1) < 1e-5, 0, site2)  # np.mod is bad
        idx2 = np.where(np.all(np.isclose(allsite, site2), axis=1))[0][0]
        xred_h2 = np.array(struct.abs_sites.abs_sites[idx2].xred+dr2)
        h_atoms2 = [np.array([xred_h2]), np.array([idx2])]

        # Calculate the actual energy and assert energy is invariant under R
        E2 = gcmc.states[0].calc.get_interaction_energy(struct.atoms,
                                                        h_atoms2)

        assert np.isclose(E1, E2, atol=1e-7), \
               "Different energy between symmetric site"
