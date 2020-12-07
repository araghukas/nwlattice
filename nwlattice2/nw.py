import nwlattice2.base as base
from nwlattice2.base import np
from nwlattice2.sizes import NanowireSize
from nwlattice2.utilities import ROOT2, ROOT3
from nwlattice2.planes import FCCb, FCCa, FCCc


class FCCPristine111(base.ANanowireLattice):
    def __init__(self, scale, n_xy=None, nz=None, width=None, length=None):
        size = NanowireSize(scale, 1 / ROOT3, n_xy, nz, width, length)
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area

        base_planes = [
            FCCa(1 / ROOT2, size.n_xy - 1),
            FCCb(1 / ROOT2, size.n_xy),
            FCCc(1 / ROOT2, size.n_xy - 1)
        ]

        # main structural logic
        planes = []
        vr = np.zeros((size.nz, 3))
        unit_vr = np.array([ROOT2 / 4, ROOT2 * ROOT3 / 12, 0.])
        for i in range(size.nz):
            planes.append(base_planes[i % 3])
            vr[i] += (i % 3) * unit_vr
        super().__init__(size, planes, vr)
        self._v_center_com = -unit_vr

    @classmethod
    def get_supercell(cls, scale, n_xy=None, width=None):
        return cls(scale, n_xy=n_xy, width=width, nz=3)

    @staticmethod
    def get_n_xy(scale, width):
        return FCCb.get_n_xy(scale, width / ROOT2)

    @staticmethod
    def get_width(scale, n_xy):
        return FCCb.get_width(scale, n_xy) / ROOT2


class ZBPristine111(FCCPristine111):
    def __init__(self, scale, n_xy=None, nz=None, width=None, length=None):
        super().__init__(scale, n_xy, nz, width, length)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))
