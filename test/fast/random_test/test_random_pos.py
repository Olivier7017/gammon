import numpy as np
from ase.geometry import get_distances

from gammon import Structure, AbsorptionSite
from ...conftest import find_data_path
from ... import context  # noqa


def test_random_abs_site():
    rcut = 2
    xred = [0.1, 0.4, 0.651]
    site = AbsorptionSite(xred=xred, xang=[0.2, 0.8, 1.302], rcut=rcut)
    site.set_rng(np.random.default_rng(1234))

    ntry = 10000
    random_sites = []
    for _ in range(ntry):
        random_sites.append(site.get_random_pos())
    random_sites = np.array(random_sites)

    mean = np.mean(random_sites, axis=0)
    mean_norm = np.mean(np.linalg.norm(random_sites, axis=1))
    var_norm = np.var(np.linalg.norm(random_sites, axis=1))

    real_mean = [0, 0, 0]
    real_mean_norm = 3 * rcut / 4
    real_var_norm = (3/5 - 9/16) * rcut**2
    """
    p(r) := d/dr (4*pi*r^3/3) /4*pi*rcut^3/3)
    real_mean_norm = int_0^rcut r * p(r)
    real_var_norm = int_0^rcut p(r) (r-real_mean_norm)
    """
    assert np.allclose(mean, real_mean, atol=5e-2), \
           f"Mean mismatch: {mean} != {real_mean}"
    assert np.isclose(mean_norm, real_mean_norm, atol=5e-2), \
           f"Mean norm mismatch: {mean_norm} != {real_mean_norm}"
    assert np.isclose(var_norm, real_var_norm, atol=5e-2), \
           f"Variance norm mismatch: {var_norm} != {real_var_norm}"


def test_random_multiple_sites(root):
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"

    ntry = 1000
    abs_sites = [[0.123, 0.456, 0.789], [0.1, 0.1, 0.1]]
    rcut = 2.
    struct = Structure(struct_fn, abs_sites, site_rcut=rcut)
    site0, site1 = [], []
    s0 = struct.abs_sites.idx_irrsites[0]
    s1 = struct.abs_sites.idx_irrsites[1]
    xred0 = struct.abs_sites.atoms.cell.cartesian_positions(
            np.array(struct.abs_sites.abs_sites[s0].xred))
    xred1 = struct.abs_sites.atoms.cell.cartesian_positions(
            np.array(struct.abs_sites.abs_sites[s1].xred))

    struct.set_rng(np.random.default_rng(1234))

    for _ in range(ntry):
        toadd0 = struct.abs_sites.atoms.cell.cartesian_positions(
                struct.abs_sites.get_random_pos(site=s0))
        toadd1 = struct.abs_sites.atoms.cell.cartesian_positions(
                struct.abs_sites.get_random_pos(site=s1))
        site0.append(toadd0)
        site1.append(toadd1)

    # We need to take into account pbc to find the real distance
    # [a][b][c] -> a=Get xred distance separately instead of norm
    #              b=Get dist with xred0 only
    #              c=Get all dist except xred0 with himself
    dist0_xred = get_distances(np.append(xred0, site0).reshape(-1, 3),
                               cell=struct.atoms.cell,
                               pbc=struct.atoms.get_pbc())[0][0][1:]
    dist1_xred = get_distances(np.append(xred1, site1).reshape(-1, 3),
                               cell=struct.atoms.cell,
                               pbc=struct.atoms.get_pbc())[0][0][1:]

    # Should be : 1. 0,0,0
    #             2. 3 * rcut / 4
    #             3. (3/5 - 9/16) * rcut**2
    mean0 = np.mean(dist0_xred, axis=0)
    mean_norm0 = np.mean(np.linalg.norm(dist0_xred, axis=1))
    var_norm0 = np.var(np.linalg.norm(dist0_xred, axis=1))

    # Should be : 1. 0,0,0
    #             2. 3 * rcut / 4
    #             3. (3/5 - 9/16) * rcut**2
    mean1 = np.mean(dist1_xred, axis=0)
    mean_norm1 = np.mean(np.linalg.norm(dist1_xred, axis=1))
    var_norm1 = np.var(np.linalg.norm(dist1_xred, axis=1))

    real_mean = [0, 0, 0]
    real_mean_norm = 3 * rcut / 4
    real_var_norm = (3/5 - 9/16) * rcut**2

    assert np.allclose(mean0, real_mean, atol=1e-1), \
           f"Mean mismatch: {mean0} != {real_mean}"
    assert np.isclose(mean_norm0, real_mean_norm, atol=1e-1), \
           f"Mean norm mismatch: {mean_norm0} != {real_mean_norm}"
    assert np.isclose(var_norm0, real_var_norm, atol=1e-1), \
           f"Variance norm mismatch: {var_norm0} != {real_var_norm}"
    assert np.allclose(mean1, real_mean, atol=1e-1), \
           f"Mean mismatch: {mean1} != {real_mean}"
    assert np.isclose(mean_norm1, real_mean_norm, atol=1e-1), \
           f"Mean norm mismatch: {mean_norm1} != {real_mean_norm}"
    assert np.isclose(var_norm1, real_var_norm, atol=1e-1), \
           f"Variance norm mismatch: {var_norm1} != {real_var_norm}"
