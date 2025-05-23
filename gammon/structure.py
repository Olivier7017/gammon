from pathlib import Path
from typing import Annotated, Literal, TypeVar
import copy
import numpy as np
import numpy.typing as npt
import spglib

from ase import Atoms
from ase.io import read, write

from . import AbsorptionSites
from .constants import MOV, SWP, ADD, DEL, VOP, VOM
from .constants import XRED_IDX, ISITE_IDX

DType = TypeVar("DType", bound=np.generic)
ArrayNx3 = Annotated[npt.NDArray[DType], Literal["N", 3]]


class Structure():
    """
    atoms : Ase.atoms
    abs_site : AbsorptionSite
    """
    def __init__(self,
                 struct_fn: Path,
                 primxred_irrsites: ArrayNx3[float],
                 site_rcut: float,
                 atol: float = 1e-2,
                 max_percent_vol: float = 2.,
                 only_add_empty: bool = True,
                 max_crystal_mov: float = 0.,
                 ):
        """
        s : Path
        Filename of a cif file contanining the atomic structure

        primxred_irrsites : array of shape (Nsite, 3). 3=[X,Y,Z]
        Position of the irreducible absorption sites in reduced coords
        of the primitive cell

        site_rcut : float
        Maximal distance in angstrom around a H-site where hydrogen can wander

        atol : float
        Precision on symmetry

        max_percent_vol : float
        Max volume change in percent to consider for VOP and VOM operation

        only_add_empty : bool
        If we only try to add atom in empty site

        max_crystal_mov : float
        Maximal displacement of a crystal atom in the mov operation (ang)
        """
        self.initial_atoms = read(struct_fn)
        self.atoms = copy.deepcopy(self.initial_atoms)
        self.curr_E = None
        self.primxred_irrsites = primxred_irrsites
        spgcell = (self.atoms.cell,
                   self.atoms.get_scaled_positions(),
                   self.atoms.numbers)
        prim_spg = spglib.find_primitive(spgcell, symprec=1e-5)
        lattice, xreds, atnum = prim_spg
        # self.primat = Atoms(symbols=atnum,
        #                    scaled_positions=xreds,
        #                    cell=lattice,
        #                    pbc=self.atoms.get_pbc())
        # self.mult = find_multiplicity(self.primat, self.atoms)

        self.primat = self.atoms
        self.mult = np.eye(3)

        # h_atoms : [xred, site_idx]
        self.h_atoms = [np.array([]), np.array([])]
        self.abs_sites = AbsorptionSites(primxred_irrsites,
                                         self.atoms,
                                         primatoms=self.primat,
                                         mult=self.mult,
                                         rcut=site_rcut)
        self.max_volume_change = self.get_volume() * (max_percent_vol/100)
        self.only_add_empty = only_add_empty
        self.max_crystal_mov = max_crystal_mov

    @property
    def nH(self):
        return len(self.h_atoms[1])

    def save(self, filename):
        """
        Save the structure as a cif file
        """
        nH = len(self.h_atoms[1])
        if nH == 0:
            write(filename, self.atoms)
        ase_h = Atoms(symbols='H' * nH,
                      scaled_positions=self.h_atoms[0],
                      cell=self.atoms.cell,
                      pbc=self.atoms.pbc)
        atoms = self.atoms + ase_h
        write(filename, atoms)
        return atoms

    def add_h_atoms(self, H, site_idx, xred):
        """
        Add an H atom in site site_idx at xred to H
        H has the format self.h_atoms (I should make a class ...)
        Returns the updated h_atoms
        """
        new_H = H.copy()
        if len(new_H[ISITE_IDX]) != 0:
            new_H[XRED_IDX] = np.vstack([new_H[XRED_IDX], xred])
            new_H[ISITE_IDX] = np.append(new_H[ISITE_IDX], site_idx)
        else:
            new_H[XRED_IDX] = np.array([xred])
            new_H[ISITE_IDX] = np.array([site_idx])
        return new_H

    def remove_h_atoms(self, initial_h_atoms, idx):
        """
        Remove the H atom at index idx to initial_h_atoms
        Returns the updated h_atoms
        """
        return [np.delete(initial_h_atoms[XRED_IDX], idx, axis=0),
                np.delete(initial_h_atoms[ISITE_IDX], idx)]

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng
        self.abs_sites.set_rng(rng)

    def try_operation(self, op):
        """
        Try an operation according to Metropolis algorithm
        Return new_at, new_H formatted like self.atoms, self.h_atoms
        """
        if op == MOV:
            return self.op_move()
        elif op == SWP:
            return self.op_swap()
        elif op == ADD:
            return self.op_add()
        elif op == DEL:
            return self.op_del()
        elif op == VOP:
            return self.op_vop()
        elif op == VOM:
            return self.op_vom()
        else:
            raise ValueError("Error S1: Operation selection failed")

    def op_move(self):
        """
        Try a move operation according to Metropolis algorithm
        Move to a random position in the same site
        Return new_at, new_H formatted like self.atoms, self.h_atoms
        """
        is_crystal, at_idx = self.random_atoms()

        if at_idx is None:  # There is no H atom in the structure
            return None, None

        elif is_crystal:  # Displace a crystal atom
            return self.displace_crystal_atom(at_idx), self.h_atoms

        else:  # Move an hydrogen atom anywhere in the site
            site_idx = self.h_atoms[ISITE_IDX][at_idx]
            new_hxred = self.abs_sites.get_random_pos(site=site_idx)
            tmp_H = self.remove_h_atoms(self.h_atoms, at_idx)
            new_H = self.add_h_atoms(tmp_H, site_idx, new_hxred)
            return self.atoms, new_H

    def op_swap(self):
        """
        Create a new system with a swap operation
        Swap to a random position in any site
        Return new_at, new_H formatted like self.atoms, self.h_atoms
        """
        H_idx = self.random_h_atoms()
        if self.only_add_empty:
            site_idx = self.abs_sites.random_empty_site(self.h_atoms[1])
        else:
            site_idx = self.abs_sites.random_site()

        if H_idx is None or site_idx is None:
            return None, None

        new_hxred = self.abs_sites.get_random_pos(site_idx)
        tmp_H = self.remove_h_atoms(self.h_atoms, H_idx)
        new_H = self.add_h_atoms(tmp_H, site_idx, new_hxred)
        return self.atoms, new_H

    def op_add(self):
        """
        Create a new system with one more hydrogen atom
        Add an H atom in a random position of a random absorption site
        Return new_at, new_H formatted like self.atoms, self.h_atoms
        """
        if self.only_add_empty:
            site_idx = self.abs_sites.random_empty_site(self.h_atoms[1])
        else:
            site_idx = self.abs_sites.random_site()

        if site_idx is None:
            return None, None

        new_hxred = self.abs_sites.get_random_pos(site=site_idx)
        new_H = self.add_h_atoms(self.h_atoms, site_idx, new_hxred)
        return self.atoms, new_H

    def op_del(self):
        """
        Create a new system with one less hydrogen atom
        Delete a random h_atom in the structure
        Return new_at, new_H formatted like self.atoms, self.h_atoms
        """
        H_idx = self.random_h_atoms()
        if H_idx is None:
            return None, None
        new_H = self.remove_h_atoms(self.h_atoms, H_idx)
        return self.atoms, new_H

    def op_vop(self):
        """
        Create a new system with a random increase in the volume
        Note : V < new_V < V + dV
        To respect detailed balance, dV = 2% of the FIRST STEP volume
        Scale H position according to this change
        Return new_at, new_H formatted like self.atoms, self.h_atoms
        """
        V = self.atoms.get_volume()
        vol_change = self.rng.random() * self.max_volume_change
        scaling = ((V+vol_change)/V)**(1/3)
        new_at = self.atoms.copy()
        new_at.set_cell(self.atoms.cell * scaling, scale_atoms=True)
        return new_at, self.h_atoms

    def op_vom(self):
        """
        Create a new system with a random decrease in volume
        Note : V - dV < new_V < V
        To respect detailed balance, dV = 2% of the FIRST STEP volume
        Scale H position according to this change
        Return new_at, new_H formatted like self.atoms, self.h_atoms
        """
        V = self.atoms.get_volume()
        vol_change = self.rng.random() * self.max_volume_change
        scaling = ((V-vol_change)/V)**(1/3)
        new_at = self.atoms.copy()
        new_at.set_cell(self.atoms.cell * scaling, scale_atoms=True)
        return new_at, self.h_atoms

    def random_atoms(self):
        """
        Select a random atom. Can be a crystal lattice atom iif the
        lattice atom can move
        The probability between each atom is uniform
        Return :
            is_crystal : True if it is part of the crystal
            idx : The index of the atoms (in self.atoms or self.h_atoms)
        """
        if self.max_crystal_mov == 0.:
            return False, self.random_h_atoms()

        tot_natom = len(self.atoms) + self.nH
        idx = self.rng.integers(0, tot_natom)
        if idx >= len(self.atoms):  # Move an H atom
            idx = idx - len(self.atoms)
            return False, idx
        else:  # Displace a crystal atoms
            return True, idx

    def random_h_atoms(self):
        """
        Select a random hydrogen atom.
        Return :
            The index in self.h_atoms
        """
        if self.nH == 0:
            return None

        h_idx = self.rng.integers(0, self.nH)
        return h_idx

    def displace_crystal_atom(self, idx):
        """
        Create a new system where an atom of the crystal is displaced
        by a random value between 0 and self.max_crystal_mov (angstrom)
        """
        d = self.random_displacement()
        new_at = self.atoms.copy()
        new_at.positions[idx] += d
        return new_at

    def random_displacement(self):
        """
        Random displacement of [0, max_crystal_mov] around a sphere
        """
        while True:
            pos = self.rng.uniform(low=-1, high=1, size=3)
            if np.linalg.norm(pos) < 1:
                return pos * self.max_crystal_mov

    def get_absorption_volume(self):
        """
        Return the volume in which an H atom can be placed (ang)
        """
        V = np.sum([4/3*np.pi*self.abs_sites.abs_sites[i].rcut**3
                    for i in range(len(self.abs_sites))])
        return V

    def get_volume(self):
        """
        Get the volume of the atom object
        """
        return self.atoms.get_volume()

    def __str__(self):
        s = f"Volume of the cell: {self.get_volume()}\n"
        s += f"Absorption sites:\n{str(self.abs_sites)}"
        s += f"H_atoms: {self.h_atoms}"
        return s
