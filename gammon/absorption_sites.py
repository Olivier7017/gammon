from collections import Counter
import numpy as np

from ase import Atoms
from ase.spacegroup import get_spacegroup
from ase.geometry import find_mic
from ase.data import chemical_symbols
from ase.geometry import get_distances
from ase.build import make_supercell

from . import AbsorptionSite
from .utilities import xred_allclose


class AbsorptionSites:
    """
    A class to deal with absorption sites
    self.atoms : Ase.atoms object (without hydrogen)
    self.abs_sites : A list of AbsorptionSite object
    self.idx_irrsites : The index of the irreducible sites in abs_sites
    self.sp : The spacegroup of the object
    """

    def __init__(self, irr_sites, atoms, primatoms, mult, rcut):
        """
        irr_sites : [[X,Y,Z], ...]
            A list of irreducible absorption sites
            in reduced coordinates of the primitive cell

        sp : Ase.spacegroup
        """
        self.atoms = atoms
        self.sp = get_spacegroup(primatoms, symprec=1e-5)
        sites = self.find_equivalent_sites(irr_sites,
                                           primatoms,
                                           mult)
        self.abs_sites = []
        for i, xred in enumerate(sites):
            xang = self.atoms.cell.cartesian_positions(xred)
            abs_site = AbsorptionSite(xred, xang, rcut=rcut)
            abs_site.neighbors = self.find_four_neighbors(xred)
            self.abs_sites.append(abs_site)

        self.set_idx_irrsites(irr_sites, primatoms, mult)
        self.warn_sites_overlap()

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng
        for site in self.abs_sites:
            site.set_rng(rng)

    def find_four_neighbors(self, site):
        """
        Find a string representing the four closest neighbors
        i.e. : "NdNi3"
        """
        cartesian_site = np.dot(site, self.atoms.get_cell())
        dist = []
        for pos in self.atoms.positions:
            d = pos - cartesian_site
            mic_dist = find_mic(d, self.atoms.get_cell(),
                                self.atoms.get_pbc())[0]
            dist.append(mic_dist)
        closest_indices = np.argsort([np.linalg.norm(d) for d in dist])[:4]
        closest_symbols = self.atoms[closest_indices].get_chemical_symbols()
        return el_list_to_string(closest_symbols)

    def find_equivalent_sites(self, irrsites, primatoms, mult):
        """
        Find the reduced positions equivalent to xred in Atoms
        irrsites : [[X,Y,Z],[X,Y,Z],...]
        Position in the primitive cell in reduced coords of the absorption
        sites

        Maybe this should be in Structure too
        """
        # Apply the symmetry of the spacegroup to the sites
        sites, kinds = self.sp.equivalent_sites(irrsites, symprec=0.01)
        primat_sites = Atoms(symbols="H"*len(sites),
                             scaled_positions=sites,
                             cell=primatoms.get_cell(),
                             pbc=primatoms.get_pbc())

        # Transform all the sites of primitive cell to the supercell
        supersites = make_supercell(primat_sites, mult)
        return supersites.get_scaled_positions()

    def set_idx_irrsites(self, irrsites, primatoms, mult):
        """
        Find the index of every irrsites in the list of absorption sites
        """
        idx_irrsites = []
        abs_sites = np.array([s.xred for s in self.abs_sites])
        for irrsite in irrsites:
            at_irrsite = Atoms(symbols="H",
                               scaled_positions=[irrsite],
                               cell=primatoms.get_cell(),
                               pbc=primatoms.get_pbc())
            multed_irrsite = make_supercell(at_irrsite, mult)
            actual_irrsite = multed_irrsite.get_scaled_positions()[0]
            found = False
            for i, abssite in enumerate(abs_sites):
                if xred_allclose(abssite, actual_irrsite, atol=1e-3):
                    idx_irrsites.append(i)
                    found = True
                    break
            if not found:
                e = f"Irreducible Site not found in abs_sites: {irrsite}"
                raise ValueError(e)
        self.idx_irrsites = idx_irrsites

    def __str__(self):
        """
        Print irreducible site : XYZ, Multiplicity, Neighbor
        """
        s = ""
        for idx in self.idx_irrsites:
            xred = self.abs_sites[idx].xred
            xang = self.abs_sites[idx].xang

            s += f"Irreducible sites at index {idx}\n"
            s += f"Position (xred): {xred}\n"
            s += f"Position (xang): {xang}\n"
            # mult = self.find_equivalent_sites(xred)
            # s += f"  Multiplicity : {len(mult)}\n"
            s += f"  Neighbors : {self.abs_sites[idx].neighbors}\n"
        return s

    def random_site(self):
        """Return the index of a random site."""
        return self.rng.integers(0, len(self.abs_sites))

    def random_empty_site(self, occupied_site):
        """
        Return the index of a random empty site.

        occupied_site : np.array
        Index of all site that are occupied
        """
        if len(self.abs_sites) == len(occupied_site):
            return None
        while True:
            idx = self.random_site()
            if idx not in occupied_site:
                return idx

    def __len__(self):
        return len(self.abs_sites)

    def get_random_pos(self, site):
        """
        Return a random position in site.
        Every point in the sphere is equally likely

        site is the index of the site
        """
        xred_site = self.abs_sites[site].xred
        dxyz_ang = self.abs_sites[site].get_random_pos()
        dxyz = self.atoms.cell.scaled_positions(dxyz_ang)

        # We need to do % twice according to ase
        coords = (xred_site-dxyz) % 1.0
        return coords % 1.0

    def warn_sites_overlap(self):
        """
        Give a warning if there is overlap between two absorption sites
        """
        xangs = np.array([abs_sites.xang for abs_sites in self.abs_sites])
        # Get distance [at1, at2]
        d = get_distances(p1=xangs,
                          cell=self.atoms.cell,
                          pbc=self.atoms.pbc)[1]
        d += np.eye(len(d)) * np.max(d)  # Remove diagonal elements

        # Find the two sites with the minimal distance
        for i in range(len(d)):
            j = np.argmin(d[i])
            dist = d[i, j]
            rcut = self.abs_sites[i].rcut

            # Print a warning is it is lower than rcut
            while dist < 2*rcut:
                w = f"WARNING : There is an overlap between Abs_site {i} "
                w += f"and Abs_site {j} at position {xangs[i]} {xangs[j]}. "
                w += f"Distance between them: {dist:.4f} ang."
                print(w)

                # Prevent duplicate warnings
                d[i, j] += rcut
                j = np.argmin(d[i])
                dist = d[i, j]


def el_list_to_string(el_list):
    """
    Transform an element list following the rules:
    1. If multiple elements : Nd Ni Ni Ni -> NdNi3
    2. Put in alphabetical order
    3. If H in the neighbor, raise ValueError

    Parameters:
    el_list (list): List of elements (e.g., ["Ni", "Nd", "Ni", "Ni"])

    Returns:
    str: Concatenated element symbols with counts (e.g., "NdNi3")

    Raises:
    ValueError: If "H" is in the list or any invalid element is found.
    """
    if "H" in el_list:
        raise ValueError("Invalid neighbor found: H")
    for el in el_list:
        if el not in chemical_symbols:
            raise ValueError(f"Invalid element found: {el}")
    el_counts = Counter(el_list)
    sorted_elements = sorted(el_counts.keys())
    result = ''.join(f"{el}{el_counts[el] if el_counts[el] > 1 else ''}"
                     for el in sorted_elements)
    return result
