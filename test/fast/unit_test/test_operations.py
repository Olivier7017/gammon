import numpy as np
import copy

from ase.geometry import get_distances

from gammon import Structure
from gammon.utilities import get_nd3_data

from ...conftest import find_data_path, total_nsites
from ... import context  # noqa

rcut = 0.2


def test_opadd(root):
    """
    Make sure the add operation works as expected
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    e, x, f = get_nd3_data()

    # 1. Define struct object
    struct = Structure(struct_fn, x, site_rcut=rcut)
    struct.set_rng(np.random.default_rng())
    assert len(struct.abs_sites.abs_sites) == total_nsites

    # 2. Test add operation
    initial_struct = copy.deepcopy(struct)
    new_at, new_H = struct.op_add()

    # 3. Make sure everything is ok
    assert [np.allclose(initial_struct.h_atoms[0], struct.h_atoms[0])], \
        "struct h_atoms has changed"
    assert initial_struct.atoms == struct.atoms, \
        "Structure has changed during add"
    assert initial_struct.atoms == new_at, "Atoms has changed during add"
    e4 = f"Incorrect nH atoms after add {len(new_H[1])} != {struct.nH}+1"
    assert len(new_H[1]) - struct.nH == 1, e4


def test_opdel(root):
    """
    Make sure the del operation works as expected
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    e, x, f = get_nd3_data()

    # 1. Initialize struct object
    struct = Structure(struct_fn, x, site_rcut=rcut)
    struct.set_rng(np.random.default_rng())
    assert len(struct.abs_sites.abs_sites) == total_nsites
    for idx, site in enumerate(struct.abs_sites.abs_sites):
        if idx % 2 == 1:
            h_xred = struct.abs_sites.abs_sites[idx].xred
            struct = add_h_atoms(struct, h_xred, idx)
    initial_h_count = struct.nH
    assert initial_h_count == total_nsites/2, "Incorrect nH during test setup"

    # 2. Do delete operation
    initial_struct = copy.deepcopy(struct)
    new_at, new_H = struct.op_del()

    # 3. Make sure everything is as expected
    assert initial_struct.atoms == struct.atoms, \
        "Structure has changed during delete"
    assert initial_struct.atoms == new_at, "Atoms have changed during delete"
    assert [np.allclose(initial_struct.h_atoms[0], struct.h_atoms[0])], \
        "struct h_atoms has changed"
    assert len(new_H[1]) == struct.nH - 1, "Incorrect nH atoms after delete"


def test_opswp(root):
    """
    Ensure swap operation works as expected
    """
    # 1. Setup: Initialize the structure object
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    e, x, f = get_nd3_data()

    struct = Structure(struct_fn, x,
                       site_rcut=rcut, only_add_empty=True)
    struct.set_rng(np.random.default_rng())
    assert len(struct.abs_sites.abs_sites) == total_nsites
    for idx, site in enumerate(struct.abs_sites.abs_sites):
        if idx % 2 == 1:
            h_xred = struct.abs_sites.abs_sites[idx].xred
            struct = add_h_atoms(struct, h_xred, idx)

    initial_h_count = struct.nH
    assert initial_h_count == total_nsites/2, "Incorrect nH during test setup"

    # 2. Perform the swap operation
    initial_struct = copy.deepcopy(struct)
    new_at, new_H = struct.op_swap()

    # 3. Ensure everything is okay
    assert initial_struct.atoms == struct.atoms, \
        "Original structure changed during swap"
    assert np.allclose(initial_struct.h_atoms[0], struct.h_atoms[0]), \
        "Shape of hydrogen atom positions in original structure changed"
    assert struct.nH == len(new_H[1]), "nH changed after swap operation"

    moved_atoms = []
    for i, h_pos in enumerate(new_H[0]):
        if not any(np.allclose(h_pos, h_orig) for h_orig in struct.h_atoms[0]):
            moved_atoms.append([new_H[0][i], new_H[1][i]])
    initial_moved_atoms = []
    for i, h_pos in enumerate(struct.h_atoms[0]):
        if not any(np.allclose(h_pos, h_new) for h_new in new_H[0]):
            initial_moved_atoms.append([struct.h_atoms[0][i],
                                        struct.h_atoms[1][i]])
    assert len(moved_atoms) == 1, "More than one H moved after swp"
    assert len(initial_moved_atoms) == 1, "More than one H moved after swp"
    assert not np.allclose(moved_atoms[0][0], initial_moved_atoms[0][0]), \
        "Swapped atom didn't move position"
    # This must be impossible since struct.only_add_empty==True
    assert moved_atoms[0][1] != initial_moved_atoms[0][1], \
        "Swapped atom didn't change site"


def test_opmov(root):
    """
    Ensure the move operation works as expected
    This contains two cases :
     1. H atom move
     2. Crystal atom move
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    e, x, f = get_nd3_data()

    # 1. Initialize struct object :
    # struct1 == Moving h_atom, struct2 == moving crystal atom
    struct1 = Structure(struct_fn, x,
                        site_rcut=rcut, max_crystal_mov=0.)
    struct1.set_rng(np.random.default_rng())
    assert len(struct1.abs_sites.abs_sites) == total_nsites
    for idx, site in enumerate(struct1.abs_sites.abs_sites):
        if idx % 2 == 1:
            h_xred = struct1.abs_sites.abs_sites[idx].xred
            struct1 = add_h_atoms(struct1, h_xred, idx)
    struct2 = Structure(struct_fn, x,
                        site_rcut=rcut, max_crystal_mov=0.5)
    struct2.set_rng(np.random.default_rng())

    initial_h_count = struct1.nH
    assert initial_h_count == total_nsites/2, "Incorrect nH during test setup"

    # 2. Do move operation
    initial_struct1 = copy.deepcopy(struct1)
    new_at1, new_H1 = struct1.op_move()

    initial_struct2 = copy.deepcopy(struct2)
    new_at2, new_H2 = struct2.op_move()

    # 3. Make sure everything is as expected for struct1
    assert new_at1 == initial_struct1.atoms == struct1.atoms, \
        "Crystal lattice changed"
    assert len(new_H1[1]) == struct1.nH == initial_struct1.nH, \
        "natoms of hydrogen changed"
    assert [np.array_equal(h1, h2) for h1, h2 in
            zip(struct1.h_atoms, initial_struct1.h_atoms)], \
        "Structure modified during the move operation"

    assert set(struct1.h_atoms[1]) == set(new_H1[1]), \
        "h_atoms changed site during the move"
    moved_atoms = []
    for i in range(len(new_H1[0])):
        if not any(np.allclose(new_H1[0][i], h) for h in struct1.h_atoms[0]):
            moved_atoms.append(new_H1[0][i])
            site_idx = new_H1[1][i]
    assert len(moved_atoms) == 1, \
        "More than one h_atom moved during mov operation"
    for j in range(len(struct1.h_atoms[1])):
        if struct1.h_atoms[1][j] == site_idx:
            tod = np.array([struct1.h_atoms[0][j], moved_atoms[0]])
            d = get_distances(tod,
                              cell=struct1.atoms.cell,
                              pbc=struct1.atoms.pbc)[1][0, 1]
            assert d < 2*rcut, "The h_atom moved farther than expected"

    # 4. Make sure everything is as expected for struct2
    assert initial_struct2.atoms == struct2.atoms, \
        "struct2 crystal lattice changed during crystal move operation"
    assert [np.array_equal(h1, h2) for h1, h2 in
            zip(struct2.h_atoms, initial_struct2.h_atoms)], \
        "struct2.h_atoms modified during the crystal move operation"
    assert [np.array_equal(h1, h2) for h1, h2 in
            zip(new_H2, struct2.h_atoms)], \
        "new_H2 modified during the crystal move operation"

    moved_atoms = []
    for i, atom in enumerate(new_at2.positions):
        if not np.allclose(atom, struct2.atoms.positions[i]):
            moved_atoms.append((i, atom))
    assert len(moved_atoms) == 1, \
        "More than one atom moved during the move operation"
    moved_idx, moved_pos = moved_atoms[0]
    original_pos = struct2.atoms.positions[moved_idx]
    distance = np.linalg.norm(moved_pos - original_pos)
    assert distance < struct2.max_crystal_mov, \
        f"Atom {moved_idx} moved farther than allowed"


def test_opvop(root):
    """
    Make sure the vop operation works as expected
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    e, x, f = get_nd3_data()

    # 1. Initialize struct object
    struct = Structure(struct_fn, x, site_rcut=rcut)
    struct.set_rng(np.random.default_rng())
    assert len(struct.abs_sites.abs_sites) == total_nsites

    for idx, site in enumerate(struct.abs_sites.abs_sites):
        if idx % 2 == 1:
            h_xred = struct.abs_sites.abs_sites[idx].xred
            struct = add_h_atoms(struct, h_xred, idx)

    initial_h_count = struct.nH
    assert initial_h_count == total_nsites/2, "Incorrect nH during test setup"

    # 2. Do delete operation
    initial_struct = copy.deepcopy(struct)
    new_at, new_H = struct.op_vop()

    # 3. Make sure everything is as expected
    xred_before = struct.atoms.get_scaled_positions()
    xred_after = new_at.get_scaled_positions()

    assert initial_struct.atoms == struct.atoms, \
        "Structure has changed during delete"
    assert np.allclose(xred_before, xred_after), \
        "Atoms have changed during delete"
    assert np.allclose(initial_struct.h_atoms[0], struct.h_atoms[0]), \
        "h_atoms has changed during vop operation"
    assert np.allclose(initial_struct.h_atoms[1], struct.h_atoms[1]), \
        "h_atoms has changed during vop operation"
    assert np.allclose(struct.h_atoms[0], new_H[0]), \
        "h_atoms has changed during vop operation"
    assert np.allclose(struct.h_atoms[1], new_H[1]), \
        "h_atoms has changed during vop operation"

    v1 = struct.atoms.get_volume()
    v2 = new_at.get_volume()
    assert v2-v1 > 0
    assert (v2-v1) < struct.max_volume_change, \
        "Volume changed more than allowed"


def test_opvom(root):
    """
    Make sure the vop operation works as expected
    """
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    e, x, f = get_nd3_data()

    # 1. Initialize struct object
    struct = Structure(struct_fn, x, site_rcut=rcut)
    struct.set_rng(np.random.default_rng())
    assert len(struct.abs_sites.abs_sites) == total_nsites

    for idx, site in enumerate(struct.abs_sites.abs_sites):
        if idx % 2 == 1:
            h_xred = struct.abs_sites.abs_sites[idx].xred
            struct = add_h_atoms(struct, h_xred, idx)

    initial_h_count = struct.nH
    assert initial_h_count == total_nsites/2, "Incorrect nH during test setup"

    # 2. Do delete operation
    initial_struct = copy.deepcopy(struct)
    new_at, new_H = struct.op_vom()

    # 3. Make sure everything is as expected
    xred_before = struct.atoms.get_scaled_positions()
    xred_after = new_at.get_scaled_positions()

    assert initial_struct.atoms == struct.atoms, \
        "Structure has changed during delete"
    assert np.allclose(xred_before, xred_after), \
        "Atoms have changed during delete"
    assert np.allclose(initial_struct.h_atoms[0], struct.h_atoms[0]), \
        "h_atoms has changed during vom operation"
    assert np.allclose(initial_struct.h_atoms[1], struct.h_atoms[1]), \
        "h_atoms has changed during vom operation"
    assert np.allclose(struct.h_atoms[0], new_H[0]), \
        "h_atoms has changed during vom operation"
    assert np.allclose(struct.h_atoms[1], new_H[1]), \
        "h_atoms has changed during vom operation"

    v1 = struct.atoms.get_volume()
    v2 = new_at.get_volume()
    assert v2-v1 < 0
    assert abs(v2-v1) < struct.max_volume_change, \
        "Volume changed more than allowed"


def add_h_atoms(struct, h_xred, idx):
    if struct.nH != 0:
        struct.h_atoms = [np.vstack([struct.h_atoms[0], h_xred]),
                          np.append(struct.h_atoms[1], idx)]
    else:
        struct.h_atoms = [np.array([h_xred]), np.array([idx])]

    return struct
