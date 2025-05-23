from pathlib import Path
import numpy as np

from ase.io import read
from ase.atoms import Atoms
from ase.build import make_supercell

from gammon import GCMC, Structure
from gammon.calculator import GraceCalc
from gammon.utilities import get_nd3_xred


def main():
    gcmc = create_GCMC()
    # test_sites(gcmc)
    nstep = 50000
    extend = False
    gcmc.run(nstep, extend=extend)


def create_GCMC():
    d = "/home/nadeauo/Projects/10-mockgcmc/mock-gcmc"
    struct_fn = "/home/nadeauo/Projects/10-mockgcmc/iteration1_h15.cif"
    # I got to set h_atoms manually if I want to start with predefined h_atoms
    # h_atoms : [xred, site_idx]
    crystal = get_crystal(struct_fn)
    crystal.write("ortho_nd3.cif")

    rcut_site = 0.5
    abs_sites = get_ortho_abs_sites()
    grace_fn = Path(d + "/test/data/Potential/Nd3MgNi14_iteration1.yaml")
    # evaluate_H2(mace_fn, calctype)  # -13.545005078171126
    chem_pot = -13.545005078171126
    calc = GraceCalc(chem_pot=chem_pot,
                     model_fn=grace_fn,
                     lmp_cmd="lmp_grace",
                     specorder="H Mg Nd Ni")
    struct = Structure("ortho_nd3.cif",
                       abs_sites,
                       site_rcut=rcut_site)

    # I got to set h_atoms manually to start with already known h_atoms
    # h_atoms : [xred, site_idx]
    # 1. I take the reducible abs_sites list
    # 2. I multiply my abs_sites to get into the orthorhombic cell
    # 3. I get a np.array of h_atoms which I set to struct
    all_sites = [a.xred for a in struct.abs_sites.abs_sites]
    h_atoms = extract_hatoms(all_sites, struct_fn)
    struct.h_atoms = h_atoms

    # Monte Carlo parameters :
    nstate = 1
    T = 300
    # prob = np.array([0.25,0.1,0.25,0.25, 0.075, 0.075])
    prob = np.array([0.75, 0.25, 0., 0., 0., 0.])  # Only move atoms in sites
    gcmc = GCMC(struct=struct, nstate=nstate,
                calc=calc, prob=prob, T=T)
    return gcmc


def get_ortho_abs_sites():
    """ ABS SITE OF UNIT CELL IN REDUCED COORDINATES"""
    xred = get_nd3_xred()
    multiplicity = np.array([[2, 0, 0],
                             [1, 2, 0],
                             [0, 0, 1]])
    dumb_cell = np.eye(3)
    unit_sites = Atoms("H"*len(xred), scaled_positions=np.array(xred),
                       cell=dumb_cell)
    supercell = make_supercell(unit_sites, multiplicity)
    ortho_sites = supercell.get_scaled_positions()
    return np.array(ortho_sites)


def test_sites(gcmc):
    abs_sites = gcmc.structs[0].abs_sites.abs_sites
    calc = gcmc.calc
    crystal = gcmc.structs[0].atoms
    grace_chempot_h = 0

    # DFT Energy (eV/conf)
    no_H = -5.49295467071757E+05  # From occopt7 calc
    occopt = 7

    if occopt == 7:
        dft_H = -3.17426435951326E+01/2
        dft_e = np.array([-5.49311534548648E+05,
                          -5.49311431831188E+05,
                          -5.49311239169182E+05,
                          -5.49311532440579E+05,
                          -5.49311220792162E+05,
                          -5.49311344365028E+05,
                          -5.49311354214139E+05,
                          -5.49311336780734E+05,
                          -5.49311381666534E+05,
                          -5.49311315102401E+05,
                          -5.49311267228472E+05,
                          -5.49311268752947E+05,
                          -5.49311300047637E+05,
                          -5.49311271353347E+05,
                          -5.49311232767303E+05,
                          -5.49311322469733E+05,
                          -5.49311337368527E+05,
                          -5.49311289729321E+05,
                          -5.49311528032685E+05])
    elif occopt == 3:
        dft_H = 0

    dft_e = dft_e - (no_H + 1 * dft_H)
    # ACE_E
    if len(crystal) == 36:
        crystal = crystal.repeat((2, 2, 1))
        # write("supercell.cif", crystal)
    if len(crystal) != 144:
        raise ValueError(f"Cannot compare to DFT natom={len(crystal)}")
    mlip_e = []
    E2 = calc.get_interaction_energy_LAMMPS(crystal)
    for idx, abs_site in enumerate(abs_sites):
        h_atoms = Atoms(symbols='H', scaled_positions=[abs_site.xred],
                        cell=crystal.cell, pbc=crystal.pbc)
        atoms = crystal + h_atoms
        E1 = calc.get_interaction_energy_LAMMPS(atoms)
        mlip_e.append(E1 - (E2 + 1*grace_chempot_h))

    for i in range(len(abs_sites)):
        print(f"{dft_e[i]} | {mlip_e[i]}")


def get_crystal(struct_fn):
    """
    Read the structure from a CIF file.
    Remove each H_atoms and only keep the crystal
    """
    atoms = read(struct_fn)
    hydrogen_mask = np.array(atoms.get_chemical_symbols()) == 'H'
    crystal = atoms[~hydrogen_mask]
    return crystal


def extract_hatoms(all_sites, struct_fn):
    """
    Read the structure from a CIF file and returns
    - h_atoms : [np.array([xred1, ...]), np.array([site_idx1, ...])]
    """
    atoms = read(struct_fn)

    # Identify hydrogen atoms
    hydrogen_mask = np.array(atoms.get_chemical_symbols()) == 'H'

    # Separate hydrogen atoms
    h_atoms_coords = atoms.get_scaled_positions()[hydrogen_mask]

    # Find the closest site to each H atom
    sites_index = []
    for coord in h_atoms_coords:
        # Compute the distance to all sites and find the closest site
        distances = np.linalg.norm(all_sites - coord, axis=1)
        closest_site_idx = np.argmin(distances)  # Index of the closest site
        sites_index.append(closest_site_idx)

    # Convert sites_index to a numpy array
    h_atoms = [h_atoms_coords, np.array(sites_index)]
    print(h_atoms)
    return h_atoms


if __name__ == "__main__":
    main()
