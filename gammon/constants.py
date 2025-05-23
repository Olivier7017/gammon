# Programmation constant
rng_seed = 7017
MOV, SWP, ADD, DEL, VOP, VOM = 0, 1, 2, 3, 4, 5
E_IDX, NH_IDX = 0, 1  # For gcmc.properties
XRED_IDX, ISITE_IDX = 0, 1  # For structs.h_atoms

# Conversion factor
J2eV = 6.241509074461E+18
da2kg = 1.66053907e-27
ang2m = 1e-10

# Physical constant
kB_eV = 8.6173303e-5  # eV/K
h_eV = 4.135667696e-15  # eV*s
mH_da = 1.007825  # Dalton
mH_eV = mH_da * J2eV * da2kg * ang2m**2  # eV s^2 / ang^-2
