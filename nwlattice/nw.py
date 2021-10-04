from nwlattice import base
from nwlattice import sizes
from nwlattice import indices
from nwlattice.utilities import ROOT2, ROOT3, ROOT6
from nwlattice.planes import FCCb, FCCa, FCCc, SqFCCa, SqFCCb, TwFCC

import numpy as np

# TODO: FCCPristine100 may have wrong z-periodicity vs. simulation box


# Simple nanowires
# ------------------------------------------------------------------------------
class FCCPristine111(base.NanowireLattice):
    """
    Pristine face-centered cubic nanowire with axis along [111]. Cross-section
    is hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell 
        """

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
        unit_vr = np.array([ROOT2 / 4, ROOT6 / 12, 0.])
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
        return FCCb.get_width(scale, n_xy) / ROOT2

    def get_size(self, scale, width=None, length=None, n_xy=None, nz=None):
        size = sizes.NanowireSize(scale, 1 / ROOT3, n_xy, nz, width, length)
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size


class FCCPristine100(base.NanowireLattice):
    """
    Pristine face-centered cubic nanowire with axis along [100]. Cross-section
    is square.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell 
        """

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


class FCCTwin(base.NanowireLatticePeriodic):
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
        j = 0
        for i in range(size.nz):
            if i in plane_index:
                j += 1  # skip upcoming `base_plane` item (mod 3)
            planes.append(base_planes[j % 3])
            vr[i] = (j % 3) * unit_vr
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
        return FCCb.get_width(scale, n_xy) / ROOT2

    @staticmethod
    def get_q(scale: float, period: float) -> int:
        return round(ROOT3 * period / 2 / scale)

    @staticmethod
    def get_period(scale: float, q: int) -> float:
        return scale * 2 * q / ROOT3

    @staticmethod
    def get_plane_index(nz: int, q: int) -> list:
        index = []
        include = True
        for i in range(nz):
            if i % q == 0:
                include = not include
            if include:
                index.append(i)
        return index

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


class FCCTwinA(base.NanowireLatticeArbitrary):
    """
       Constant-width arbitrarily twinning face-centered cubic nanowire with axis
       along [111]. Cross-section is hexagonal.
    """

    DEFAULT_INDEXER = indices.LinearDecrease(1)

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, indexer: callable = None):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param indexer: a function int -> (int, list) that returns nz and twin point indices
        """
        if indexer is None:
            indexer = FCCTwinA.DEFAULT_INDEXER

        size = self.get_size(indexer, scale, width, length, n_xy, nz)

        base_planes = [
            FCCa(1 / ROOT2, size.n_xy - 1),
            FCCb(1 / ROOT2, size.n_xy),
            FCCc(1 / ROOT2, size.n_xy - 1)
        ]

        plane_index = size.index
        planes = []
        vr = np.zeros((size.nz, 3))
        unit_vr = np.array([ROOT2 / 4, ROOT6 / 12, 0.])

        # lattice twins at each `i` in `plane_index`
        i = 0  # array index
        j = 0  # base planes index
        d = 1  # base planes index offset
        while i < size.nz:
            if i in plane_index:
                d = d + 1 if d % 3 != 2 else 1
            j += d
            planes.append(base_planes[j % 3])
            vr[i] = (j % 3) * unit_vr
            i += 1

        super().__init__(size, planes, vr)
        self._v_center_com = -unit_vr

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCb.get_width(scale, n_xy) / ROOT2

    def get_size(self, indexer: callable, scale, width=None, length=None, n_xy=None, nz=None):
        size = sizes.NanowireSizeArbitrary(scale, 1 / ROOT3, n_xy, nz, width, length)
        size._indexer = indexer
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size


class FCCTwinFaceted(base.NanowireLatticePeriodic):
    """
    Faceted twinning face-centered cubic nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    Cross-section is constant-perimeter hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 m0: int = 0, q: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param period: approximated twinning period
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param m0: second index of first `TwFCC` plane in vertical stack,
                   `m0=0` corresponds to a regular hexagonal `TwFCC` plane
        :param q: (overrides `period`) number of planes in a half-period
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
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
        return FCCb.get_width(scale, n_xy) / ROOT2

    @staticmethod
    def get_q(scale: float, period: float) -> int:
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

        step = 1
        m_cycle = [m0]
        i = 0
        while i < nz - 1:
            next_q = m_cycle[-1] + step
            m_cycle.append(next_q)
            if next_q == q or next_q == 0:
                step *= -1
            i += 1
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


class FCCTwinFacetedA(base.NanowireLatticeArbitrary):
    """
    Faceted twinning face-centered cubic nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    Cross-section is constant-perimeter hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, m0: int = 0,
                 indexer: callable = None):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param indexer: a function int -> (int, list) that returns nz and twin point indices
        """

        if indexer is None:
            tmp_size = self.get_size(indices.Empty(), scale, width, length, n_xy, nz)
            indexer = indices.LinearDecrease(m=1, q_max=tmp_size.n_xy - 1)

        size = self.get_size(indexer, scale, width, length, n_xy, nz)
        plane_index = size.index

        step = 1
        i = 0
        next_q = m0 + step
        m_cycle = [next_q]

        q_max = size.n_xy - 1
        if next_q == q_max or next_q == 0 or (i in plane_index):
            step *= -1
        i += 1

        while i < size.nz:
            next_q = m_cycle[-1] + step
            m_cycle.append(next_q)

            # twin if structure forces it or if `plane_index` says so
            if next_q == q_max or next_q == 0 or (i in plane_index):
                step *= -1
            i += 1

        planes = [TwFCC(1 / ROOT2, size.n_xy, m) for m in m_cycle]
        vr = np.zeros((size.nz, 3))
        super().__init__(size, planes, vr)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCb.get_width(scale, n_xy) / ROOT2

    def get_size(self, indexer: callable, scale, width=None, length=None, n_xy=None, nz=None):
        size = sizes.NanowireSizeArbitrary(scale, 1 / ROOT3, n_xy, nz, width, length)
        size._indexer = indexer
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size


class HexPristine0001(base.NanowireLattice):
    """
    Pristine hexagonal nanowire with axis along [0001]. Cross-section is
    hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
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
            FCCa(1 / ROOT2, size.n_xy),
            FCCc(1 / ROOT2, size.n_xy)
        ]

        # main structural logic: periodic -[A-C]- stacking of planes
        planes = []
        vr = np.zeros((size.nz, 3))
        unit_vr = np.array([ROOT2 / 4, ROOT6 / 12, 0.])
        for i in range(size.nz):
            planes.append(base_planes[i % 2])
            vr[i] += (i % 2) * 2 * unit_vr
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
        return FCCb.get_width(scale, n_xy) / ROOT2

    def get_size(self, scale, width=None, length=None, n_xy=None, nz=None):
        size = sizes.NanowireSize(scale, 1 / ROOT3, n_xy, nz, width, length)
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size


class FCCRandomHex(base.NanowireLattice):
    """
    Face-centered cubic nanowire with a specific fraction of hexagonal planes
    substituted in at random locations. Axis along [111]. Cross-section is
    hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 fraction: float = 0.5, n_xy: int = None, nz: int = None,
                 force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param fraction: between 0 and 1, the fraction random of hexagonal
                         stacking faults
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
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

        if force_cyclic:
            trunc = 0
            while not planes[0].fits(planes[-1]):
                if trunc > 3:
                    raise ValueError("could not force cyclic structure")
                planes = planes[:-1]
                vr = vr[:-1]
                trunc += 1
                size.fix_nz(size.nz - 1)
            self.print("truncated planes list by %d to force cyclic" % trunc)

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
        return FCCb.get_width(scale, n_xy) / ROOT2

    @staticmethod
    def get_plane_index(nz: int, fraction: float) -> list:
        from random import sample
        if abs(fraction) > 1.:
            fraction = 1.
        if fraction < 0.:
            fraction += 1.
        assert 0 <= fraction <= 1

        k = int(nz * fraction)
        if k == 0:
            k = 1

        # `k` random integers between 0 and `nz`
        index = sample([i for i in range(nz)], k)
        return index

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


# Derived nanowires
# ------------------------------------------------------------------------------
class ZBPristine111(FCCPristine111):
    """
    Pristine zincblende nanowire with axis along [111]. Cross-section is
    hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        super().__init__(scale, width, length, n_xy, nz,
                         force_cyclic=force_cyclic)
        # v2 = np.array([1 / 2 / ROOT2, -1 / 2 / ROOT6, -1 / 4 / ROOT3])
        v2 = np.array([0., 0., ROOT3 / 4])
        self.add_basis(2, v2)


class DiamondPristine111(FCCPristine111):
    """
    Pristine diamond nanowire with axis along [111]. Cross-section is hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        super().__init__(scale, width, length, n_xy, nz,
                         force_cyclic=force_cyclic)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class ZBPristine100(FCCPristine100):
    """
    Pristine zincblende nanowire with axis along [100]. Cross-section is square.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        super().__init__(scale, width, length, n_xy, nz,
                         force_cyclic=force_cyclic)
        self.add_basis(2, [0.25, 0.25, 0.25])
        self._v_center_com = -0.125 * np.ones(3)


class DiamondPristine100(FCCPristine100):
    """
    Pristine diamond nanowire with axis along [100].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell

        """
        super().__init__(scale, width, length, n_xy, nz,
                         force_cyclic=force_cyclic)
        self.add_basis(1, [0.25, 0.25, 0.25])
        self._v_center_com = -0.125 * np.ones(3)


class ZBTwin(FCCTwin):
    """
    Constant-width periodically twinning zincblende nanowire with axis along
    [111]. Cross-section is hexagonal.
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
        super().__init__(scale, width, length, period, n_xy, nz, q,
                         force_cyclic)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBTwinA(FCCTwinA):
    """
       Constant-width arbitrarily twinning zincblende nanowire with axis
       along [111]. Cross-section is hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, indexer: callable = None):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param indexer: a function int -> (int, list) that returns nz and twin point indices
        """
        super().__init__(scale, width, length, n_xy, nz, indexer)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondTwin(FCCTwin):
    """
    Constant-width periodically twinning diamond nanowire with axis along [111].
    Cross-section is hexagonal.
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
        super().__init__(scale, width, length, period, n_xy, nz, q,
                         force_cyclic)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class ZBTwinFaceted(FCCTwinFaceted):
    """
    Faceted twinning zincblende nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    Cross-section is hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 m0: int = 0, q: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param period: approximated twinning period
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param m0: second index of first `TwFCC` plane in vertical stack,
                   `m0=0` corresponds to a regular hexagonal `TwFCC` plane
        :param q: (overrides `period`) number of planes in a half-period
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        super().__init__(scale, width, length, period, n_xy, nz, m0, q,
                         force_cyclic)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBTwinFacetedA(FCCTwinFacetedA):
    """
    Faceted twinning zincblende nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    Cross-section is constant-perimeter hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, m0: int = 0,
                 indexer: callable = None):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param indexer: a function int -> (int, list) that returns nz and twin point indices
        """
        super().__init__(scale, width, length, n_xy, nz, m0, indexer)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondTwinFaceted(FCCTwinFaceted):
    """
    Faceted twinning diamond nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    Cross-section is hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 period: float = None, n_xy: int = None, nz: int = None,
                 m0: int = 0, q: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param period: approximated twinning period
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param m0: second index of first `TwFCC` plane in vertical stack,
                   `m0=0` corresponds to a regular hexagonal `TwFCC` plane
        :param q: (overrides `period`) number of planes in a half-period
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        super().__init__(scale, width, length, period, n_xy, nz, m0, q,
                         force_cyclic)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))


class WZPristine0001(HexPristine0001):
    """
    Pristine wurtzite nanowire with axis along [0001]. Cross-section is
    hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        super().__init__(scale, width, length, n_xy, nz,
                         force_cyclic=force_cyclic)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBRandomWZ(FCCRandomHex):
    """
    Zincblende nanowire with a specific fraction of wurtzite planes
    substituted in at random locations. Axis along [111]. Cross-section is
    hexagonal.
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 fraction: float = 0.5, n_xy: int = None, nz: int = None,
                 force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param fraction: between 0 and 1, the fraction random of hexagonal
                         stacking faults
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        super().__init__(scale, width, length, fraction, n_xy, nz, force_cyclic)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class DiamondRandomWZ(FCCRandomHex):
    """
    Diamond nanowire with a specific fraction of 'wurtzite-but-same-atom' planes
    substituted in at random locations. Axis along [111].
    """

    def __init__(self, scale: float, width: float = None, length: float = None,
                 fraction: float = 0.5, n_xy: int = None, nz: int = None,
                 force_cyclic: bool = True):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param fraction: between 0 and 1, the fraction random of hexagonal
                         stacking faults
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param force_cyclic: adjust nz (length) to nearest multiple of supercell
        """
        super().__init__(scale, width, length, fraction, n_xy, nz, force_cyclic)
        self.add_basis(1, np.array([0., 0., ROOT3 / 4]))
