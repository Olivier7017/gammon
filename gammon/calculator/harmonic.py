from typing import Annotated, Literal, TypeVar
import copy
import numpy as np
import numpy.typing as npt

from ase import Atoms
from ase.build import make_supercell
from ase.geometry import get_distances, find_mic
from ase.spacegroup import get_spacegroup

from ..utilities import are_equivalent, xred_allclose
from .calculator import Calculator

DType = TypeVar("DType", bound=np.generic)
ArrayNx4 = Annotated[npt.NDArray[DType], Literal["N", 4]]


class HarmonicCalc(Calculator):
    def __init__(self,
                 coef_irrsites,
                 rcut_swit: float = 2.1):
        """
        coef_sites : array of shape (n_irr_sites, V)[float]
          where V = [E0, M] for every irreducible site
          with M being a 3x3 np.array of : [[dfx/dx, dfy/dx, dfz/dx],
                                            [dfx/dy, dfy/dy, dfz/dy],
                                            [dfx/dz, dfy/dz, dfz/dz]]
          (in eV, eV/ang**2)

        rcut_swit : float
          The radius of the Switendick exclusion sphere between H (ang)
        """
        self.coef_irrsites = coef_irrsites
        self.rcut_swit = rcut_swit
        self.all_sites = None  # To set after finding symmetry

    def prepare_calc(self, struct):
        """
        Take the irreducible sites coefficients and apply all symmetry
        to find all of the possible site in the structure.

        Calculate the coefficients of the new sites
         e.g. dx^2 changes with a 120 rotation

        self.all_sites = [idx][xyz, Emin, Hess] ->
         idx = Index of the reducible sites
         xang (np.array(3)): Coordinates of the site (ang)
         Emin (float): Minimum energy of the absorption site (eV)
         Hess (np.array(3x3)): Hessian Matrix (eV/ang**2)
        """
        abs_sites = struct.abs_sites
        prim_allsites = []  # [xang, e0, ifc]

        # 1. Compute the equivalent sites in the unit cell
        irr_sites = struct.primxred_irrsites
        fake_mult = np.diag([1, 1, 1])
        prim_sites = abs_sites.find_equivalent_sites(irr_sites,
                                                     struct.primat,
                                                     fake_mult)

        # 2. Find the associated IFC for each sites
        for i, site in enumerate(prim_sites):
            # 2.1 Find the irreduccible site associated to site
            eq = abs_sites.find_equivalent_sites(site,
                                                 struct.primat,
                                                 fake_mult)
            isfound = False
            for irr_idx, irr_site in enumerate(irr_sites):
                match = [xred_allclose(irr_site, s) for s in eq]

                # 2.2 Apply the symmetry between irrsite and site to the IFC
                if any(match):  # irr is in eq
                    isfound = True
                    fixed_coef = self.apply_symmetry(
                            irr=irr_site,
                            site=site,
                            coef=self.coef_irrsites[irr_idx],
                            atoms=struct.primat)
                    xang = struct.primat.cell.cartesian_positions(site)
                    a_site = [xang, fixed_coef[0], fixed_coef[1]]
                    prim_allsites.append(a_site)
                    break

            if not isfound:
                msg = "Equivalent site not found for abs_sites\n"
                msg += f"Absorption sites (xred): {site}\n"
                msg += f"Equivalent sites: {eq}\n"
                msg += "Irreduccible sites:"
                msg += str([abs_sites.abs_sites[irr].xred
                            for irr in abs_sites.idx_irrsites])
                raise ValueError(msg)

        # 2. Use multiplicity to copy the site to the supercell
        # slower than computing it directly with my code, but safer
        self.all_sites = []
        for site in prim_allsites:
            tomult = Atoms(symbols="H",
                           positions=[site[0]],
                           cell=struct.primat.get_cell(),
                           pbc=struct.primat.get_pbc())
            supercell_sites = make_supercell(tomult, struct.mult)
            for new_pos in supercell_sites.get_positions():
                tmp_site = copy.copy(site)
                tmp_site[0] = new_pos
                self.all_sites.append(tmp_site)

        # 3. Verify that everything worked fine
        abs_xred = np.array([s.xred for s in struct.abs_sites.abs_sites])
        calc_xred = np.array([struct.atoms.cell.scaled_positions(s[0])
                              for s in self.all_sites])

        if not are_equivalent(abs_xred, calc_xred):
            e = "Something went wrong with HarmonicCalc symmetries:"
            e += "struct.abs_sites != HarmonicCalc.abs_sites"
            raise ValueError(e)

    def apply_symmetry(self, coef, irr, site, atoms):
        """
        Apply symmetry/reflexion/rotation to irr_coef to get the coeff
        of every equivalent absorption sites.
        coef: [E0, V] -> See HarmonicCalc.__init__

        irr: [X,Y,Z]
         xred positions of the irreducible site

        site:
         xred positions of the actual_site

        atoms: ase.Atoms
         Atoms object to find the symmetry operation
        """
        # Step 1: Find the symmetry operations path between irr and site
        sp = get_spacegroup(atoms, symprec=1e-5)
        symop_abc = []
        for rot, trans in sp.get_symop():
            new_pos = np.mod(np.dot(rot, irr) + trans, 1.)  # Apply symop
            if np.allclose(new_pos, site):
                symop_abc.append([rot, trans])

        # Step 2 :
        B = atoms.cell.array.T
        coef_list = []  # Hessian after applying operation
        test = []  # Position after applying operation

        """
        Let's recap the derivation :
         1. f = np.dot(H, v)
         2. E = -1/2 np.dot(v^T, f)

        To change a basis, I can directly apply the basis on the vector :
        (v' = B^-1 v)
        Since B^-1 v is still a vector, I can directly apply H on it :
        f' = H B^-1 v
        But to get f in the old basis, I retransform back :
        f = (B H B^-1) v
        Such that :
        f' = H' v'
        E = -1/2 v'^T f'
        """
        # For all equivalent way to get to a site
        for rot, trans in symop_abc:
            rot_xyz = self.find_rot_xyz(op=rot, basis=B)
            test = rot_xyz @ atoms.cell.cartesian_positions(irr)
            test = np.mod(atoms.cell.scaled_positions(test) + trans, 1.)
            test = np.where(abs(test-1) < 1e-5, 0, test)

            # Test if xred is equivalent with both path
            if not np.allclose(test, site):
                msg = "rot in xyz and abc doesn't give the same site\n"
                msg += f"XYZ: {test}\n, ABC: {site}\n"
                msg += f"rot: {rot}\n, trans: {trans}\n, rot_xyz: {rot_xyz}\n"
                raise ValueError(msg)
            H = coef[1]
            res = rot_xyz @ H @ np.linalg.inv(rot_xyz)
            coef_list.append(res)

        for i in range(len(coef_list)):
            c = coef_list[i]
            symop = symop_abc[i]
            # 2e-2 eV/ang is too small due to dft precision ?
            if not np.allclose(c, coef_list[0], atol=1e-1):
                print(coef_list[0])
                print(rot_xyz)
                # 120deg rotation  ROTXYZ [[-0.5       -0.8660254  0.       ]
                #  [ 0.8660254 -0.5        0.       ]
                #  [ 0.         0.         1.       ]]

                print(f"The site is {irr}")
                s = ("Something went wrong, 2 different path to get to the" +
                     " same site gave different Hessian coef. Here is the " +
                     f"symmetry operation list (rot-trans):\n {symop}\n" +
                     "Here is the different dfpt_coefficient" +
                     f"\n{c}\n And here the original hessian\n" +
                     f"{coef_list[0]}\n and the original xred site {irr}")
                raise ValueError(s)
        return [coef[0], coef_list[0]]

    def find_rot_xyz(self, op, basis):
        """
        Transform an operation matrix inside a basis (abc) to
        the same operation but on the xyz matrix.

        x_new = (B^T)^-1 x_old
        Such that
        O_abc = B.T @ O @ (B^T)^-1
        """
        return basis @ (op @ np.linalg.inv(basis))

    def get_potential_energy(self, crystal, h_atoms, mu=0, proc_id=0):
        """
        Return the energy of atoms in eV :
         E = sum_i (Ei_abs + r_i.T H r_i) - N*u

        crystal: ase.Atoms
         The absorbant without hydrogen

        h_atoms: The list inside Structure
         Contains the positions of the atoms (xred)

        mu: float
        The chemical potential of H_2/2 (eV/at)
        """
        is_ok = True
        if self.rcut_swit > 0:
            is_ok = self.test_switendick(h_atoms,
                                         crystal.cell,
                                         crystal.get_pbc())

        if is_ok:
            E = self.get_interaction_energy(crystal, h_atoms)
            return E - mu * len(h_atoms[1])
        else:  # Switendick not respected, giving dumb high energy
            return 9e+12

    def test_switendick(self, h_atoms, cell, pbc):
        """
        We need to convert in xang before using get_distances
        """
        if len(h_atoms[1]) > 0:
            # Here I could use p1 and p2 if I'm sure the moved atom is
            # always p1 = h_atoms[0][-1] and then p2 = h_atoms[0][:-1]

            dist = get_distances(h_atoms[0], cell=cell, pbc=pbc)[0]
            # dist will have dimension : [atom1, atom2, xyz]
            for atom1 in range(len(dist)):
                for atom2 in range(len(dist[atom1])):
                    if atom1 == atom2:
                        continue
                    d = cell.array.T @ dist[atom1][atom2]
                    if sum([x**2 for x in d])**(1/2) < self.rcut_swit:
                        return False
        return True

    def get_interaction_energy(self, crystal, h_atoms):
        """
        Calculate the energy of the structure
        Note : h_atoms is [xred, idx_site]
        """
        E = 0
        # mic_dist: Finds the minimal distance according to pbc between
        #  h_atoms and the center of the site
        for idx in range(len(h_atoms[1])):
            site_idx = h_atoms[1][idx]
            xang_site = self.all_sites[site_idx][0]
            xang_atom = crystal.cell.cartesian_positions(h_atoms[0][idx])
            E0 = self.all_sites[site_idx][1]
            c = self.all_sites[site_idx][2]

            dist = xang_site - xang_atom
            # # # IMPORTANT DIST MUST BE IN ANGSTROM
            mic = find_mic(dist,
                           crystal.get_cell(),
                           crystal.get_pbc())[0]
            # E0 + 1/2 r^T d^2E/dr^2 r
            toadd = E0 + 0.5 * np.dot(mic.T, np.dot(c, mic))
            E += toadd

        return E

    def __str__(self):
        E0s = [self.coef_irrsites[i][0]
               for i in range(len(self.coef_irrsites))]
        d2s = [self.coef_irrsites[i][1:]
               for i in range(len(self.coef_irrsites))]
        s = "Harmonic Calculator:\n"
        s += f"  Switendick criterion : {self.rcut_swit} ang\n"
        s += f"  Irreducible sites E0 (eV): {E0s}\n"
        s += f"  d^2E/dr^2 in xyz (eV/ang): {d2s}\n"
        return s
