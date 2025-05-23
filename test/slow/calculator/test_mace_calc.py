import numpy as np
import pytest

from gammon import Structure
from gammon.calculator import MaceCalc

from ...conftest import find_data_path, has_lammps_mace
from ... import context  # noqa


@pytest.mark.skipif(not has_lammps_mace(),
                    reason="LAMMPS not compiled with MACE package")
def test_basis_symmetry(root):
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    mace_fn = datapath / "Potential" / "mace_mp_0b.model-lammps.pt"
    rcut = 0.2

    # Prepare the object
    irr_site_list = [[0.01, 0., 0.88404693872], [0.123, 0.456, 0.789],
                     [0.032956852013, 0.26647842601, 0.88404693872]]  # xred
    dr_list = [[0.001, 0., 0.], [0., 0.21, 0.],
               [0.0128321, 0.00931282, 0.33128291]]  # xang

    for irr_site, dr in zip(irr_site_list, dr_list):
        abs_sites = np.array([irr_site])
        calc = MaceCalc(model_fn=mace_fn,
                        specorder="Nd Mg Ni H",
                        lmp_cmd="lmp_mace")
        struct = Structure(struct_fn,
                           abs_sites,
                           site_rcut=rcut)
        allsite = [s.xred for s in struct.abs_sites.abs_sites]
        dr = struct.atoms.cell.scaled_positions(dr)  # xred

        # Site 1 : angle = 0 deg
        dr1 = dr
        site1 = np.array(irr_site)
        idx1 = np.where(np.all(np.isclose(allsite, site1), axis=1))[0][0]
        xred_h1 = np.array(struct.abs_sites.abs_sites[idx1].xred + dr1)
        xang_h1 = struct.atoms.cell.cartesian_positions(xred_h1)  # xang
        h_atoms1 = [np.array([xang_h1]), np.array([idx1])]  # xang
        calc.get_potential_energy(struct.atoms, h_atoms1, mu=0.)
