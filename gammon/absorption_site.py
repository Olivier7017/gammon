import numpy as np


class AbsorptionSite:
    """
    Represent one H absorption site
    """

    def __init__(self, xred, xang, rcut):
        """
        xred : [X,Y,Z] -> Reduced/Fractional coordinates
        xang : [X,Y,Z] -> Angstrom coordinates
        neighbors : "NdNi3" or equivalent
        #min_e : GS on the site
        #de : The value of a in :
        #  E = (a[1]x^2 + a[2]y^2 + a[3]z^2) * energy
        rcut : Range of the absorption site
        """
        self.xred = xred
        self.xang = xang
        self.rcut = rcut

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def get_random_pos(self):
        """
        Fast uniform distribution around a sphere
        We need to remove all point outside the sphere to
        get an uniform distribution using this method

        Return :
        dxyz : A random position around the site in ang
        """
        while True:
            pos = self.rng.uniform(low=-1, high=1, size=3)
            if np.linalg.norm(pos) < 1:
                return pos*self.rcut

    def __str__(self):
        xred_str = [f"{self.xred[i]:.2f}" for i in range(3)]
        xang_str = [f"{self.xang[i]:.2f}" for i in range(3)]
        return (
            f"Center (xred): {xred_str}\n"
            f"Center (xang): {xang_str}\n"
        )
