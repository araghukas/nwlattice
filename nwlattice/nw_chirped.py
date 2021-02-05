import numpy as np

from nwlattice import base
from nwlattice import sizes
from nwlattice.planes import FCCa, FCCb, FCCc
from nwlattice.utilities import ROOT2, ROOT3, ROOT6


class FCCTwinChirped(base.ANanowireLatticePeriodic):
    """
       Constant-width periodically twinning face-centered cubic nanowire with axis
       along [111]. Cross-section is hexagonal.
       """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 q: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param period: approximated twinning period
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param q: (overrides `period`) number of planes in a half-period
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        size = self.get_size(scale, width, length, period, n_xy, nz, q)
        if force_cyclic:
            old_nz = size.nz
            old_length = size.length
            size.fix_nz(self.get_cyclic_nz(size.nz, 2 * size.q))
            self.print("forced cyclic structure in periodic z: (unit nz = %d) "
                       "(nz: %d -> %d) "
                       "(length: %.2f -> %.2f)"
                       % (2 * size.q, old_nz, size.nz, old_length, size.length))

        base_planes = [
            FCCa(1 / ROOT2, size.n_xy - 1),
            FCCb(1 / ROOT2, size.n_xy),
            FCCc(1 / ROOT2, size.n_xy - 1)
        ]

        # main structural logic: -[A-B-C]- stacking with -[A-B-A]- transitions
        plane_index = self.get_plane_index(size.nz, size.q)
        planes = []
        vr = np.zeros((size.nz, 3))
        unit_vr = np.array([ROOT2 / 4, ROOT6 / 12, 0.])

        i = 0  # array index
        j = 0  # base planes index
        d = 1  # base planes index offset
        while i < size.nz:
            print("{:>4} {:>4} {:>4}".format(i, j, j % 3))
            if i in plane_index:
                d = d + 1 if d % 3 != 2 else 1
            planes.append(base_planes[j % 3])
            vr[i] = (j % 3) * unit_vr
            j += d
            i += 1

        super().__init__(size, planes, vr)
        self._v_center_com = -unit_vr

    @classmethod
    def get_supercell(cls, scale, width=None, period=None, n_xy=None, q=None):
        nz = cls.get_cyclic_nz(0, 2 * q)
        return cls(scale, width=width, period=period, n_xy=n_xy, nz=nz, q=q,
                   force_cyclic=False)  # already cyclic by definition

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCb.get_width(scale, n_xy) / ROOT2

    @staticmethod
    def get_q(scale: float, period: float) -> int:
        return round(ROOT3 * period / 2 / scale)

    @staticmethod
    def get_period(scale: float, q: int) -> float:
        return scale * 2 * q / ROOT3

    @staticmethod
    def get_plane_index(nz: int, q: int) -> set:
        top = 0
        index = [0]
        while q > 0:
            for i in range(2):
                index.append(top + q)
                top += q
            q -= 1

        return set(index)

    def get_size(self, scale, width=None, length=None, period=None,
                 n_xy=None, nz=None, q=None):
        size = sizes.NanowireSizePeriodic(
            scale, 1 / ROOT3, n_xy, nz, q, width, length, period)
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        size._q_func = self.get_q
        size._period_func = self.get_period
        return size
