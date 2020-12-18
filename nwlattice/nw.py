from nwlattice import base
from nwlattice import sizes
from nwlattice.utilities import ROOT2, ROOT3, ROOT6
from nwlattice.planes import FCCb, FCCa, FCCc, SqFCCa, SqFCCb, TwFCC

import numpy as np


class FCCPristine111(base.ANanowireLattice):
    """
    Pristine face-centered cubic nanowire with axis along [111].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):

        size = self.get_size(scale, width, length, n_xy, nz)
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
    def get_supercell(cls, scale, width=None, n_xy=None):
        return cls(scale, width=width, n_xy=n_xy, nz=3)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy: int) -> float:
        return FCCb.get_width(scale, n_xy)

    def get_size(self, scale, width=None, length=None, n_xy=None, nz=None):
        size = sizes.NanowireSize(scale, 1 / ROOT3, n_xy, nz, width, length)
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

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):

        size = self.get_size(scale, width, length, n_xy, nz)
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
    def get_supercell(cls, scale, width=None, n_xy=None):
        return cls(scale, width=width, n_xy=n_xy, nz=2)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return SqFCCa.get_n_xy(scale, width)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return SqFCCa.get_width(scale, n_xy)

    def get_size(self, scale, width=None, length=None, n_xy=None, nz=None):
        size = sizes.NanowireSize(scale, 0.5, n_xy, nz, width, length)
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

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 q: int = None, force_cyclic: bool = True):
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
    def get_supercell(cls, scale, width=None, period=None, n_xy=None, q=None):
        nz = cls.get_cyclic_nz(0, 2 * q)
        return cls(scale, width=width, period=period, n_xy=n_xy, nz=nz, q=q,
                   force_cyclic=False)  # already cyclic by definition

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCb.get_width(scale, n_xy)

    @staticmethod
    def get_q(scale: float, period: float) -> int:
        return round(ROOT3 * period / 2 / scale) // 2

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


class BinnedFCCTwin(base.ACompoundNanowireLattice):
    """
    A compound nanowire with X-Y-X-Y-X structure, where
        X ~ FCCPristine111
        Y ~ FCCTwin
    """

    def __init__(self, scale: float, width: float = None, lengths: list = None,
                 period: float = None, n_xy: int = None, nzs: list = None,
                 q: int = None):

        if lengths is None:
            lengths = [None, None]
        elif len(lengths) != 2:
            raise ValueError("must specify 2 lengths for this type of wire")

        if nzs is None:
            nzs = [None, None]
        elif len(nzs) != 2:
            raise ValueError("must specify 2 nz's for this type of wire")

        nw_list = [
            FCCPristine111(scale, width, lengths[0], n_xy, nzs[0], True),
            FCCTwin(scale, width, lengths[1], period, n_xy, nzs[1], q, True),
            FCCPristine111(scale, width, lengths[0], n_xy, nzs[0], True),
            FCCTwin(scale, width, lengths[1], period, n_xy, nzs[1], q, True),
            FCCPristine111(scale, width, lengths[0], n_xy, nzs[0], True)
        ]

        # construct `nw_index` parameter
        nz_bin = nw_list[0].size.nz
        nz_mid = nw_list[1].size.nz
        nz = 3 * nz_bin + 2 * nz_mid  # i.e. X-Y-X-Y-X

        n1 = nz_bin
        n2 = n1 + nz_mid
        n3 = n2 + nz_bin
        n4 = n3 + nz_mid
        n5 = n4 + nz_bin

        nw_index = []
        for i in range(nz):
            # append index of corresponding source lattice in `nw_list`
            if 0 <= i < n1:
                nw_index.append(0)  # first bin
            elif n1 <= i < n2:
                nw_index.append(1)  # first wire
            elif n2 <= i < n3:
                nw_index.append(2)  # middle bin
            elif n3 <= i < n4:
                nw_index.append(3)  # second wire
            elif n4 <= i < n5:
                nw_index.append(4)  # final bin
            else:
                raise IndexError("exceeded nz-bounds of compound structure")
        super().__init__(nw_list, nw_index, nw_list[1]._size_obj)

    @property
    def supercell(self):
        return self

    @classmethod
    def get_supercell(cls, *args, **kwargs):
        # dummy class method; can not guarantee supercell in general
        return cls(*args, **kwargs)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCTwin.get_n_xy(scale, width)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCTwin.get_width(scale, n_xy)

    def get_size(self, *args):
        # dummy method: `self.size` set by superclass
        return self.size


class FCCTwinFaceted(base.ANanowireLatticePeriodic):
    """
    Faceted twinning face-centered cubic nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 m0: int = 0, q: int = None, force_cyclic: bool = True):

        size = self.get_size(scale, width, length, period, n_xy, nz, q)
        if size.q >= size.n_xy:
            raise ValueError(
                "period {:.2f} (q = {:d}) is too large for width "
                "{:.2f} (n_xy = {:d})".format(size.period, size.q, size.width,
                                              size.n_xy)
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
        m_cycle = self.get_m_cycle(size.nz, m0, size.q)
        planes = [TwFCC(1 / ROOT2, size.n_xy, m) for m in m_cycle]
        vr = np.zeros((size.nz, 3))
        super().__init__(size, planes, vr)

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
        return FCCb.get_width(scale, n_xy)

    @staticmethod
    def get_q(scale: float, period: float) -> int:
        return round(ROOT3 * period / 2 / scale) // 2

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


class BinnedFCCTwinFaceted(base.ACompoundNanowireLattice):
    """
    A compound nanowire with X-Y-X-Y-X structure, where
        X ~ FCCPristine111
        Y ~ FCCTwinFaceted
    """

    # TODO: add q0 parameter back into FCCTwinFaceted so smoother transitions
    def __init__(self, scale: float, width: float = None, lengths: list = None,
                 period: float = None, n_xy: int = None, nzs: list = None,
                 q: int = None):

        if lengths is None:
            lengths = [None, None]
        elif len(lengths) != 2:
            raise ValueError("must specify 2 lengths for this type of wire")

        if nzs is None:
            nzs = [None, None]
        elif len(nzs) != 2:
            raise ValueError("must specify 2 nz's for this type of wire")

        nw1 = FCCTwinFaceted(scale, width, lengths[1], period, n_xy, nzs[1],
                             0, q, True)
        nw0 = FCCPristine111(scale, length=lengths[0], n_xy=nw1.size.n_xy + 1,
                             nz=nzs[0], force_cyclic=True)
        nw3 = FCCTwinFaceted(scale, width, lengths[1], period, n_xy, nzs[1],
                             nw1.size.q - 1, q, True)

        nw_list = [nw0, nw1, nw0, nw3, nw0]

        # construct `nw_index` parameter
        nz_bin = nw_list[0].size.nz
        nz_mid = nw_list[1].size.nz
        nz = 3 * nz_bin + 2 * nz_mid  # i.e. X-Y-X-Y-X

        n1 = nz_bin
        n2 = n1 + nz_mid
        n3 = n2 + nz_bin
        n4 = n3 + nz_mid
        n5 = n4 + nz_bin

        nw_index = []
        for i in range(nz):
            # append index of corresponding source lattice in `nw_list`
            if 0 <= i < n1:
                nw_index.append(0)  # first bin
            elif n1 <= i < n2:
                nw_index.append(1)  # first wire
            elif n2 <= i < n3:
                nw_index.append(2)  # middle bin
            elif n3 <= i < n4:
                nw_index.append(3)  # second wire
            elif n4 <= i < n5:
                nw_index.append(4)  # final bin
            else:
                raise IndexError("exceeded nz-bounds of compound structure")
        super().__init__(nw_list, nw_index, nw_list[1]._size_obj)

    @property
    def supercell(self):
        return self

    @classmethod
    def get_supercell(cls, *args, **kwargs):
        # dummy class method; can not guarantee supercell in general
        return cls(*args, **kwargs)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCTwinFaceted.get_n_xy(scale, width)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCTwinFaceted.get_width(scale, n_xy)

    def get_size(self, *args):
        # dummy method: `self.size` set by superclass
        return self.size


class HexPristine0001(base.ANanowireLattice):
    """
    Pristine hexagonal nanowire with axis along [0001].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        size = self.get_size(scale, width, length, n_xy, nz)
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
    def get_supercell(cls, scale, width=None, n_xy=None):
        return cls(scale, width=width, n_xy=n_xy, nz=2)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy: int) -> float:
        return FCCb.get_width(scale, n_xy)

    def get_size(self, scale, width=None, length=None, n_xy=None, nz=None):
        size = sizes.NanowireSize(scale, 1 / ROOT3, n_xy, nz, width, length)
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

    def __init__(self, scale: float, width: float = None, length: float = None,
                 fraction: float = 0.5, n_xy: int = None, nz: int = None):

        size = self.get_size(scale, width, length, fraction, n_xy, nz)
        base_planes = [
            FCCa(1 / ROOT2, size.n_xy - 1),
            FCCb(1 / ROOT2, size.n_xy),
            FCCc(1 / ROOT2, size.n_xy - 1)
        ]

        # main structural logic: -[A-B-C]- with random -[A-B-A]- stacking
        planes = []
        vr = np.zeros((size.nz, 3))
        unit_vr = np.array([ROOT2 / 4, ROOT2 * ROOT3 / 12, 0.])
        plane_index = self.get_plane_index(size.nz, fraction)
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
    def get_supercell(cls, scale, width=None, length=None, fraction=0.5,
                      n_xy=None, nz=None):
        # can not guarantee unit smaller than `self` in random lattice
        return cls(scale, width, length, fraction, n_xy, nz)

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

    def get_size(self, scale, width=None, length=None, fraction=0.5,
                 n_xy=None, nz=None):
        size = sizes.NanowireSizeRandom(scale, 1 / ROOT3, fraction, n_xy, nz,
                                        width, length)
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

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None):
        super().__init__(scale, width, length, n_xy, nz)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondPristine111(FCCPristine111):
    """
    Pristine diamond nanowire with axis along [111].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None):
        super().__init__(scale, width, length, n_xy, nz)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class ZBPristine100(FCCPristine100):
    """
    Pristine zincblende nanowire with axis along [100].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None):
        super().__init__(scale, width, length, n_xy, nz)
        self.add_basis(2, [0.25, 0.25, 0.25])
        self._v_center_com = -0.125 * np.ones(3)


class DiamondPristine100(FCCPristine100):
    """
    Pristine diamond nanowire with axis along [100].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None):
        super().__init__(scale, width, length, n_xy, nz)
        self.add_basis(1, [0.25, 0.25, 0.25])
        self._v_center_com = -0.125 * np.ones(3)


class ZBTwin(FCCTwin):
    """
    Constant-width periodically twinning zincblende nanowire with axis along
    [111].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 q: int = None, force_cyclic: bool = True):
        super().__init__(scale, width, length, period, n_xy, nz, q,
                         force_cyclic)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class BinnedZBTwin(BinnedFCCTwin):
    """
    A compound nanowire with X-Y-X-Y-X structure, where
        X ~ ZBPristine111
        Y ~ ZBTwin
    """

    def __init__(self, scale: float, width: float = None, lengths: list = None,
                 period: float = None, n_xy: int = None, nzs: int = None,
                 q: int = None):
        super().__init__(scale, width, lengths, period, n_xy, nzs, q)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondTwin(FCCTwin):
    """
    Constant-width periodically twinning diamond nanowire with axis along [111].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 q: int = None, force_cyclic: bool = True):
        super().__init__(scale, width, length, period, n_xy, nz, q,
                         force_cyclic)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class ZBTwinFaceted(FCCTwinFaceted):
    """
    Faceted twinning zincblende nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 q: int = None, force_cyclic: bool = True):
        super().__init__(scale, width, length, period, n_xy, nz, q,
                         force_cyclic)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class BinnedZBTwinFaceted(BinnedFCCTwinFaceted):
    """
    A compound nanowire with X-Y-X-Y-X structure, where
        X ~ ZBPristine111
        Y ~ ZBTwin
    """

    def __init__(self, scale: float, width: float = None, lengths: list = None,
                 period: float = None, n_xy: int = None, nzs: int = None,
                 q: int = None):
        super().__init__(scale, width, lengths, period, n_xy, nzs, q)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondTwinFaceted(FCCTwinFaceted):
    """
    Faceted twinning diamond nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 q: int = None, force_cyclic: bool = True):
        super().__init__(scale, width, length, period, n_xy, nz, q,
                         force_cyclic)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class WZPristine0001(HexPristine0001):
    """
    Pristine wurtzite nanowire with axis along [0001].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None):
        super().__init__(scale, width, length, n_xy, nz)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBRandomWZ(FCCRandomHex):
    """
    Zincblende nanowire with a specific fraction of wurtzite planes
    substituted in at random locations. Axis along [111].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 fraction: float = 0.5, n_xy: int = None, nz: int = None):
        super().__init__(scale, width, length, fraction, n_xy, nz)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondRandomWZ(FCCRandomHex):
    """
    Diamond nanowire with a specific fraction of 'wurtzite-but-same-atom' planes
    substituted in at random locations. Axis along [111].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 fraction: float = 0.5, n_xy: int = None, nz: int = None):
        super().__init__(scale, width, length, fraction, n_xy, nz)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))
