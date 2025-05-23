import numpy as np

from ase.build import make_supercell
from ase import Atoms
from ase.cell import Cell
from ase.geometry import find_mic, cellpar_to_cell


def make_logo():
    logo = """-----------------------------------
-|░█▀▀░█▀█░█▄█░█▄█░█▀█░█▀█ |
-|░█░█░█▀█░█░█░█░█░█░█░█░█ |
-|░▀▀▀░▀░▀░▀░▀░▀░▀░▀▀▀░▀░▀ |
-----------------------------------
"""
    return logo


def add_best_hydrogen(struct, calc):
    """
    Add an hydrogen atom in the absorption site with the lowest energy
    """
    crystal = struct.atoms
    h_atoms = struct.h_atoms
    abs_sites = struct.abs_sites.abs_sites
    keep_E = 1e+12
    keep_h = None

    init_E = calc.get_potential_energy(crystal, h_atoms=h_atoms)
    print(f"Initial Structure: {init_E}")

    for idx, site in enumerate(abs_sites):
        if idx in h_atoms[1]:  # Site already filled
            continue
        new_h = [np.array(h_atoms[0]), np.array(h_atoms[1])]
        if new_h is None or len(new_h[0]) == 0:
            new_h = [np.array([site.xred]), np.array([idx])]
        else:
            new_h[0] = np.append(new_h[0], [site.xred], axis=0)
            new_h[1] = np.append(new_h[1], idx)
        new_E = calc.get_potential_energy(crystal=crystal, h_atoms=new_h)

        if new_E < keep_E:
            keep_E = new_E
            keep_h = new_h
    print(f"Final Structure: {keep_E} with h in {keep_h[1]}")
    return keep_h


def find_multiplicity(primat, atoms):
    """
    I could not find this in ase or spglib
    Receive a primitive atoms and find the multiplicity to get to the supercell
    Assuming C' = M P -> M = C' P^-1, with M = mult, C=Supercell, P=Primitive

    I can deal with the change from prim to conv for FCC, BCC.
    However, I don't want to add them to the FCC or BCC cell. If you use a
    conv cell, you will provide the absorption site in this conv cell.
    """
    # 0. Ase is strange. There is an hidden parameters for the vector a, b, c.
    #    If I start with 60 degree and modify to get a 90 degree cell.
    #    So I use cellpar to avoid this bug.
    cell = cellpar_to_cell(primat.cell.cellpar())
    primat.set_cell(cell, scale_atoms=True)
    cell = cellpar_to_cell(atoms.cell.cellpar())
    atoms.set_cell(cell, scale_atoms=True)
    mult = np.eye(3)

    # 1. Look at angles to determine if FCC or BCC conversion is needed
    ang_prim = np.round(primat.cell.angles(), 2)
    ang_atoms = np.round(atoms.cell.angles(), 2)
    if not np.allclose(ang_prim, ang_atoms):
        bcc_angle = np.round(np.degrees(np.arccos(-1/3)), 2)

        if np.allclose(ang_prim, [60, 60, 60]) \
                and np.allclose(ang_atoms, [90, 90, 90]):
            # FCC primitive → conventional
            mult = np.array([[-1., 1., 1.],
                             [1., -1., 1.],
                             [1., 1., -1.]])

        elif np.allclose(ang_prim, [bcc_angle]*3) \
                and np.allclose(ang_atoms, [90, 90, 90]):
            # BCC primitive → conventional
            mult = np.array([[0.,  1.,  1.],
                             [1.,  0.,  1.],
                             [1.,  1.,  0.]])
        else:
            e = "I haven't programmed the transformation to this conventional"
            e += f" cell. Sorry. \nprim = {ang_prim}\natoms = {ang_atoms}"
            raise ValueError(e)

    # Remake the cell to remove the hidden a,b,c parameter
    new_primat = make_supercell(primat, mult)
    cell = cellpar_to_cell(new_primat.cell.cellpar())
    new_primat.set_cell(cell, scale_atoms=True)

    # 2. Check if supercell
    inv_cell = np.linalg.inv(new_primat.cell.array)
    sc_mult = np.dot(atoms.cell.array, inv_cell)
    total_mult = np.dot(sc_mult, mult)

    # 3. Sanity check
    _verify_mult(primat, atoms, total_mult)

    return total_mult


def _verify_mult(primat, atoms, mult):
    supercell = make_supercell(primat, mult)
    if not are_equivalent(atoms, supercell):
        raise ValueError("Could not find the correct multiplicity of the cell")


def are_equivalent(at1, at2, atol=1e-3):
    """
    Return True if both item are equivalent with a difference of atol in
    scaled coordinates allowed
    Can receive the np.array directly or the structure
    """
    if isinstance(at1, Atoms):
        c1 = at1.get_cell()
        c2 = at2.get_cell()
        # Comparing arrays isnt enough test/unit_test/test_utilities:test_fcc
        if not (np.allclose(c1.lengths(), c2.lengths(), atol=atol) or
                np.allclose(c1.angles(), c2.angles(), atol=atol)):
            return False

        # Create a cell with unit vector
        c = [c1.array[i] / np.linalg.norm(c1.array[i]) for i in range(3)]
        c1 = Cell(c)
        c2 = Cell(c)
        p1 = at1.get_scaled_positions()
        p2 = at2.get_scaled_positions()
    else:
        p1 = at1
        p2 = at2
        c1 = Cell(np.eye(3))
        c2 = Cell(np.eye(3))

    if np.shape(p1) != np.shape(p2):
        return False

    if np.shape(p1)[0] == 1:  # Trivial if only 1 atom
        return True

    c = Cell(np.eye(3))

    # Find the distance between p1[0] and p1[i]
    dist1 = []
    for i in range(len(p1)):
        dist1.append(find_mic(p1[0]-p1[i],
                     cell=c,
                     pbc=True)[0])
    dist1 = np.abs(dist1)

    # Try every atom of c2 as the origin atom
    for j in range(len(p2)):
        dist2 = []
        for jj in range(len(p2)):
            dist2.append(find_mic(p2[jj]-p2[j],
                         cell=c,
                         pbc=True)[0])
        dist2 = np.abs(dist2)

        # Compare every distance of both array
        for xi1 in dist1:
            at_matched = False
            for xi2 in dist2:
                if np.allclose(xi1, xi2, atol=atol):
                    at_matched = True
                    break  # We found a match for xi1

            if not at_matched:
                print(f"Could not find a match for {xi1} and {xi2}")
                break  # No match for xi1. Try a new central atom for dist2

        if at_matched:  # Every xi1 had an equivalent xi2
            return True

    # Tried with every atom at the origin but could not match xi1 with xi2
    return False


def xred_allclose(xred1, xred2, atol=1e-3):
    xred1 = np.asarray(xred1)
    xred2 = np.asarray(xred2)
    if xred1.shape[-1] != 3 or xred2.shape != xred1.shape:
        e = "xred1 and xred2 must have the same shape ending in (3,)"
        raise ValueError(e)

    delta = np.abs(xred1 - xred2)
    delta = np.minimum(delta, 1.0 - delta)
    return np.all(delta < atol)
