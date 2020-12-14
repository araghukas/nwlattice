from nwlattice import base
from nwlattice.sizes import NanowireSize, NanowireSizePeriodic
from nwlattice.utilities import ROOT2, ROOT3, ROOT6
from nwlattice.planes import FCCb, FCCa, FCCc, SqFCCa, SqFCCb, TwFCC

import numpy as np


class FCCPristine111(base.ANanowireLattice):
    """
    Pristine face-centered cubic nanowire with axis along [111].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None,
                 force_cyclic: bool = True):
        dz = 1 / ROOT3
        size = self._assign_rules(
            NanowireSize(scale, dz, n_xy, nz, width, length)
        )
        if force_cyclic:
            old_nz = size.nz
            old_length = size.length
            size.fix_nz(self.get_cyclic_nz(size.nz, 3))
            self.print("forced cyclic structure in periodic z: (unit nz = %d) "
                       "(nz: %d -> %d) "
                       "(length: %.2f -> %.2f)"
                       % (3, old_nz, size.nz, old_length, size.length))

        base_planes = [
            FCCa(1 / ROOT2, size.n_xy - 1),
            FCCb(1 / ROOT2, size.n_xy),
            FCCc(1 / ROOT2, size.n_xy - 1)
        ]

        # main structural logic: periodic -[A-B-C]- stacking of planes
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
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy: int) -> float:
        return FCCb.get_width(scale, n_xy)

    def _assign_rules(self, size):
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size


class FCCPristine100(base.ANanowireLattice):
    """
    Pristine face-centered cubic nanowire with axis along [100].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None,
                 force_cyclic: bool = True):
        dz = 0.5
        size = self._assign_rules(
            NanowireSize(scale, dz, n_xy, nz, width, length)
        )
        if force_cyclic:
            old_nz = size.nz
            old_length = size.length
            size.fix_nz(self.get_cyclic_nz(size.nz, 2))
            self.print("forced cyclic structure in periodic z: (unit nz = %d) "
                       "(nz: %d -> %d) "
                       "(length: %.2f -> %.2f)"
                       % (2, old_nz, size.nz, old_length, size.length))
        base_planes = [
            SqFCCa(1.0, size.n_xy),
            SqFCCb(1.0, size.n_xy)
        ]

        # main structural logic: periodic -[A-B]- stacking of planes
        planes = []
        vr = np.zeros((size.nz, 3))
        for i in range(size.nz):
            planes.append(base_planes[i % 2])
        super().__init__(size, planes, vr)

    @classmethod
    def get_supercell(cls, scale, n_xy=None, width=None):
        return cls(scale, n_xy=n_xy, width=width, nz=2)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return SqFCCa.get_n_xy(scale, width)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return SqFCCa.get_width(scale, n_xy)

    def _assign_rules(self, size):
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size


class FCCTwin(base.ANanowireLatticePeriodic):
    """
    Constant-width periodically twinning face-centered cubic nanowire with axis
    along [111].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 q: int = None,
                 width: float = None,
                 length: float = None,
                 period: float = None,
                 force_cyclic: bool = True):
        dz = 1 / ROOT3
        size = self._assign_rules(
            NanowireSizePeriodic(scale, dz, n_xy, nz, q, width, length, period)
        )
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
        j = 0
        for i in range(size.nz):
            if i in plane_index:
                j += 1  # skip upcoming `base_plane` item (mod 3)
            planes.append(base_planes[j % 3])
            vr[i] += (j % 3) * unit_vr
            j += 1
        super().__init__(size, planes, vr)
        self._v_center_com = -unit_vr

    @classmethod
    def get_supercell(cls, scale, n_xy=None, q=None, width=None, period=None):
        nz = cls.get_cyclic_nz(0, 2 * q)
        return cls(scale, n_xy=n_xy, q=q, width=width, period=period, nz=nz,
                   force_cyclic=False)  # already cyclic by definition

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCb.get_width(scale, n_xy)

    @staticmethod
    def get_p(scale: float, period: float) -> int:
        return round(ROOT3 * period / 2 / scale)

    @staticmethod
    def get_period(scale: float, q: int) -> float:
        return scale * 2 * q / ROOT3

    @staticmethod
    def get_plane_index(nz: int, q: int) -> set:
        index = []
        include = True
        for i in range(nz):
            if i % q == 0:
                include = not include
            if include:
                index.append(i)
        return set(index)

    def _assign_rules(self, size):
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        size._q_func = self.get_p
        size._period_func = self.get_period
        return size


class FCCTwinFaceted(base.ANanowireLatticePeriodic):
    """
    Faceted twinning face-centered cubic nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 q: int = None,
                 width: float = None,
                 length: float = None,
                 period: float = None,
                 force_cyclic: bool = True):

        dz = 1 / ROOT3
        size = self._assign_rules(
            NanowireSizePeriodic(scale, dz, n_xy, nz, q, width, length, period)
        )

        if size.q >= size.n_xy:
            raise ValueError(
                "period {:.2f} (q = {:d}) is too large for width "
                "{:.2f} (n_xy = {:d})"
                .format(size.period, size.q, size.width, size.n_xy)
            )

        if force_cyclic:
            old_nz = size.nz
            old_length = size.length
            size.fix_nz(self.get_cyclic_nz(size.nz, 2 * size.q))
            self.print("forced cyclic structure in periodic z: (unit nz = %d) "
                       "(nz: %d -> %d) "
                       "(length: %.2f -> %.2f)"
                       % (2 * size.q, old_nz, size.nz, old_length, size.length))

        # main structural logic:
        # -[A-B-C]- stacking, planes cycle through constant-perimeter variants
        m_cycle = self.get_m_cycle(size.nz, 0, size.q)
        planes = [TwFCC(1 / ROOT2, size.n_xy, m) for m in m_cycle]
        vr = np.zeros((size.nz, 3))
        super().__init__(size, planes, vr)

    @classmethod
    def get_supercell(cls, scale, n_xy=None, q=None, width=None, period=None):
        nz = cls.get_cyclic_nz(0, 2 * q)
        return cls(scale, n_xy=n_xy, q=q, width=width, period=period, nz=nz,
                   force_cyclic=False)  # already cyclic by definition

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCb.get_width(scale, n_xy)

    @staticmethod
    def get_p(scale: float, period: float) -> int:
        return round(ROOT3 * period / 2 / scale)

    @staticmethod
    def get_period(scale: float, q: int) -> float:
        return scale * 2 * q / ROOT3

    @staticmethod
    def get_m_cycle(nz, m0, q):
        """
        List of `m_xy` indices for TwFCC in an FCCTwinFaceted instance.

        Example, for (nz=9, m0=0, q=4): -> [0, 1, 2, 3, 4, 3, 2, 1, 0]

        Example, for (nz=12, m0=4, q=7): -> [4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1]

        :param nz: number of planes stacked
        :param m0: `m_xy` index of first TwFCC in the list
        :param q: the number of planes in one twin segment
        :return: list of `m_xy` indices for TcFCC planes
        """
        m_cycle = [m0]
        step = 1
        count = 0
        while count < nz - 1:
            next_q = m_cycle[-1] + step
            m_cycle.append(next_q)
            if next_q == q or next_q == 0:
                step *= -1
            count += 1
        return m_cycle

    def _assign_rules(self, size):
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        size._q_func = self.get_p
        size._period_func = self.get_period
        return size


class HexPristine0001(base.ANanowireLattice):
    """
    Pristine hexagonal nanowire with axis along [0001].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None,
                 force_cyclic: bool = True):
        dz = 1 / ROOT3
        size = self._assign_rules(
            NanowireSize(scale, dz, n_xy, nz, width, length)
        )
        if force_cyclic:
            old_nz = size.nz
            old_length = size.length
            size.fix_nz(self.get_cyclic_nz(size.nz, 2))
            self.print("forced cyclic structure in periodic z: (unit nz = %d) "
                       "(nz: %d -> %d) "
                       "(length: %.2f -> %.2f)"
                       % (2, old_nz, size.nz, old_length, size.length))
        base_planes = [
            FCCa(1 / ROOT2, size.n_xy - 1),
            FCCb(1 / ROOT2, size.n_xy),
        ]

        # main structural logic: periodic -[A-B-C]- stacking of planes
        planes = []
        vr = np.zeros((size.nz, 3))
        unit_vr = np.array([ROOT2 / 4, ROOT2 * ROOT3 / 12, 0.])
        for i in range(size.nz):
            planes.append(base_planes[i % 2])
            vr[i] += (i % 2) * unit_vr
        super().__init__(size, planes, vr)
        self._v_center_com = -unit_vr

    @classmethod
    def get_supercell(cls, scale, n_xy=None, width=None):
        return cls(scale, n_xy=n_xy, width=width, nz=2)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy: int) -> float:
        return FCCb.get_width(scale, n_xy)

    def _assign_rules(self, size):
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size


class FCCRandomHex(base.ANanowireLattice):
    """
    Face-centered cubic nanowire with a specific fraction of hexagonal planes
    substituted in at random locations. Axis along [111].
    """

    def __init__(self,
                 scale: float,
                 hex_fraction: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None):

        dz = 1 / ROOT3
        size = self._assign_rules(
            NanowireSize(scale, dz, n_xy, nz, width, length)
        )
        base_planes = [
            FCCa(1 / ROOT2, size.n_xy - 1),
            FCCb(1 / ROOT2, size.n_xy),
            FCCc(1 / ROOT2, size.n_xy - 1)
        ]

        # main structural logic: -[A-B-C]- with random -[A-B-A]- stacking
        planes = []
        vr = np.zeros((size.nz, 3))
        unit_vr = np.array([ROOT2 / 4, ROOT2 * ROOT3 / 12, 0.])
        plane_index = self.get_plane_index(size.nz, hex_fraction)
        j = 0
        for i in range(size.nz):
            if i in plane_index:
                j += -2 * (i % 2)
            planes.append(base_planes[j % 3])
            vr[i] += (j % 3) * unit_vr
            j += 1
        super().__init__(size, planes, vr)
        self._v_center_com = -unit_vr

    @classmethod
    def get_supercell(cls, scale, wz_fraction,
                      n_xy=None, nz=None, width=None, length=None):
        # can not guarantee unit smaller than `self` in random lattice
        return cls(scale, wz_fraction, n_xy, nz, width, length)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCb.get_width(scale, n_xy)

    @staticmethod
    def get_plane_index(nz: int, fraction: float) -> set:
        from random import sample
        if abs(fraction) > 1.:
            fraction = 1.
        if fraction < 0.:
            fraction = 1. + fraction
        assert 0 <= fraction <= 1

        k = int(nz * fraction)
        if k == 0:
            k = 1

        # `k` random integers between 0 and `nz`
        index = sample([i for i in range(nz)], k)
        return set(index)

    @property
    def supercell(self):
        # can not guarantee unit smaller than `self` in random lattice
        return self

    def _assign_rules(self, size):
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size


# Derived classes
# ------------------------------------------------------------------------------
class ZBPristine111(FCCPristine111):
    """
    Pristine zincblende nanowire with axis along [111].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None):
        super().__init__(scale, n_xy, nz, width, length)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondPristine111(FCCPristine111):
    """
    Pristine diamond nanowire with axis along [111].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None):
        super().__init__(scale, n_xy, nz, width, length)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class ZBPristine100(FCCPristine100):
    """
    Pristine zincblende nanowire with axis along [100].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None):
        super().__init__(scale, n_xy, nz, width, length)
        self.add_basis(2, [0.25, 0.25, 0.25])
        self._v_center_com = -0.125 * np.ones(3)


class DiamondPristine100(FCCPristine100):
    """
    Pristine diamond nanowire with axis along [100].
    """

    def __init__(self, scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None):
        super().__init__(scale, n_xy, nz, width, length)
        self.add_basis(1, [0.25, 0.25, 0.25])
        self._v_center_com = -0.125 * np.ones(3)


class ZBTwin(FCCTwin):
    """
    Constant-width periodically twinning zincblende nanowire with axis along
    [111].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 q: int = None,
                 width: float = None,
                 length: float = None,
                 period: float = None,
                 force_cyclic: bool = True):
        super().__init__(scale, n_xy=n_xy, nz=nz, q=q,
                         width=width, length=length, period=period,
                         force_cyclic=force_cyclic)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondTwin(FCCTwin):
    """
    Constant-width periodically twinning diamond nanowire with axis along [111].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 q: int = None,
                 width: float = None,
                 length: float = None,
                 period: float = None,
                 force_cyclic: bool = True):
        super().__init__(scale, n_xy=n_xy, nz=nz, q=q,
                         width=width, length=length, period=period,
                         force_cyclic=force_cyclic)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class ZBTwinFaceted(FCCTwinFaceted):
    """
    Faceted twinning zincblende nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 q: int = None,
                 width: float = None,
                 length: float = None,
                 period: float = None,
                 force_cyclic: bool = True):
        super().__init__(scale, n_xy, nz, q, width, length, period,
                         force_cyclic)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondTwinFaceted(FCCTwinFaceted):
    """
    Faceted twinning diamond nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 q: int = None,
                 width: float = None,
                 length: float = None,
                 period: float = None,
                 force_cyclic: bool = True):
        super().__init__(scale, n_xy, nz, q, width, length, period,
                         force_cyclic)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class WZPristine0001(HexPristine0001):
    """
    Pristine wurtzite nanowire with axis along [0001].
    """

    def __init__(self,
                 scale: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None):
        super().__init__(scale, n_xy, nz, width, length)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBRandomWZ(FCCRandomHex):
    """
    Zincblende nanowire with a specific fraction of wurtzite planes
    substituted in at random locations. Axis along [111].
    """

    def __init__(self,
                 scale: float,
                 wz_fraction: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None):
        super().__init__(scale, wz_fraction, n_xy, nz, width, length)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondRandomWZ(FCCRandomHex):
    """
    Diamond nanowire with a specific fraction of 'wurtzite-but-same-atom' planes
    substituted in at random locations. Axis along [111].
    """

    def __init__(self,
                 scale: float,
                 wz_fraction: float,
                 n_xy: int = None,
                 nz: int = None,
                 width: float = None,
                 length: float = None):
        super().__init__(scale, wz_fraction, n_xy, nz, width, length)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))
