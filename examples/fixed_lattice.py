import numpy as np

from gammon import GCMC, Structure
from gammon.calculator import HarmonicCalc
from gammon.utilities import get_nd3_data


def main():
    # Monte Carlo parameters :
    nstep = 500000
    T = 298
    extend = False

    # Structure parameters :
    struct_fn = "Nd3MgNi14-2H.cif"
    rcut_site = 0.  # 0.5 means about (100/540) ang**3 of absorption sites
    e0, xred_sites, ifc = get_nd3_data(hse_corr=False, quantum_corr=False)

    nstate = 8
    chem_pot = np.linspace(-0.25, 0.1, 8)
    coefs = [[e0[i], ifc[i]] for i in range(len(e0))]
    Edxdydz = coefs
    rcut_swit = 2.1
    mcm = 0.
    calc = HarmonicCalc(coef_irrsites=Edxdydz,
                        rcut_swit=rcut_swit)

    struct = Structure(struct_fn,
                       xred_sites,
                       site_rcut=rcut_site,
                       only_add_empty=True,
                       max_crystal_mov=mcm)

    prob = np.array([0.25, 0.25, 0.25, 0.25, 0., 0.])
    gcmc = GCMC(struct=struct, nstate=nstate, prob=prob, T=T, calc=calc,
                mu=chem_pot, buffer_size=1000)
    gcmc.run(nstep, extend=extend)


if __name__ == "__main__":
    main()
