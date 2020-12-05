import nwlattice.base as base
import nwlattice.stacks as stacks

from nwlattice.planes import HexPlane, SquarePlane
from nwlattice.utilities import ROOT3


class FCCPristine111G(base.AStackGeometry):
    def nz(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def validate_args(self, diameter, length, p, nz):
        if not (diameter or p):
            raise ValueError("must specify either `diameter` or `p`")
        if not (length or nz):
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, diameter, length, p=None, nz=None, z_periodic=False):
        super().__init__(a0)
        self.validate_args(diameter, length, p, nz)
        nz = self.nz(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 3)
        self.nz = nz
        self.p = self.xy_index(diameter) if p is None else p

        self.length = length
        self.diameter = diameter


class FCCPristine100G(base.AStackGeometry):
    def nz(self, length: float) -> int:
        return 1 + round(2 * length / self.a0)

    def xy_index(self, side_length: float) -> int:
        return SquarePlane.index_for_diameter(self.a0, side_length)

    def validate_args(self, side_length, length, r, nz):
        if not (side_length or r):
            raise ValueError("must specify either `diameter` or `r`")
        if not (length or nz):
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, side_length, length, r=None, nz=None,
                 z_periodic=False):
        super().__init__(a0)
        self.validate_args(side_length, length, r, nz)
        nz = self.nz(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 2)
        self.nz = nz
        self.r = self.xy_index(side_length) if r is None else r

        self.length = length
        self.side_length = side_length


class FCCTwin111G(base.AStackGeometry):
    def nz(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def get_q_max(self, period: float) -> int:
        return round(ROOT3 * period / 2 / self.a0)

    def get_period(self) -> float:
        return 2 * self.a0 * self.q_max / ROOT3

    def validate_args(self, diameter, length, period, index, p, nz, q_max):
        if not (diameter or p):
            raise ValueError("must specify either `diameter` or `p`")
        if not (length or nz):
            raise ValueError("must specify either `length` or `nz`")
        if period is None and q_max is None and index is None:
            raise ValueError(
                "must specify either `period`, `q_max`, or `index`")

    def __init__(self, a0, diameter, length, period, index=None, p=None,
                 nz=None, q_max=None, z_periodic=False):
        super().__init__(a0)
        self.validate_args(diameter, length, period, index, p, nz, q_max)
        self.q_max = self.get_q_max(period) if q_max is None else q_max
        nz = self.nz(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 2 * self.q_max)
        self.nz = nz
        self.p = self.xy_index(diameter) if p is None else p
        self.index = self.get_index(nz, self.q_max) if index is None else index

        self.diameter = diameter
        self.length = length
        self.period = self.get_period()

    @staticmethod
    def get_index(nz, q_max):
        index = []
        include = True
        for i in range(nz):
            if i % q_max == 0:
                include = not include
            if include:
                index.append(i)
        return index


class FCCTwinFacetedG(base.AStackGeometry):

    def nz(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def get_q_max(self, period: float) -> int:
        return round(ROOT3 * period / 2 / self.a0)

    def get_period(self) -> float:
        return 2 * self.a0 * self.q_max / ROOT3

    def validate_args(self, diameter, length, period, p, nz, q_max):
        if not (diameter or p):
            raise ValueError("must specify either `diameter` or `p`")
        if not (length or nz):
            raise ValueError("must specify either `length` or `nz`")
        if not (period or q_max):
            raise ValueError("must specify either `period` or `q_max`")

    def __init__(self, a0, diameter, length, period, p=None, nz=None,
                 q_max=None, q0=0, z_periodic=False):
        super().__init__(a0)
        self.validate_args(diameter, length, period, p, nz, q_max)
        self.p = self.xy_index(diameter)
        q_max = self.get_q_max(period) if q_max is None else q_max
        if q_max >= self.p:
            q_max = self.p - 1
        self.q_max = q_max
        nz = self.nz(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 2 * self.q_max)
        self.nz = nz
        self.q0 = q0 if 0 <= q0 <= self.q_max else 0

        self.diameter = diameter
        self.length = length
        self.period = self.get_period()


class HexPristine0001G(base.AStackGeometry):
    def nz(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def validate_args(self, diameter, length, p, nz):
        if not (diameter or p):
            raise ValueError("must specify either `diameter` or `p`")
        if not (length or nz):
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, diameter, length, p=None, nz=None, z_periodic=False):
        super().__init__(a0)
        self.validate_args(diameter, length, p, nz)
        nz = self.nz(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 2)
        self.nz = nz
        self.p = self.xy_index(diameter) if p is None else p

        self.length = length
        self.diameter = diameter


class FCCTwinPaddedG(base.APaddedStackGeometry):
    @property
    def nz_bottom(self) -> int:
        if self._nz_bottom is None:
            self._nz_bottom = ((self.pg.nz - self.nz_core) // 2) // 3 * 3
        return self._nz_bottom

    @property
    def nz_core(self) -> int:
        if self._nz_core is None:
            nz_core = self.cg.nz
            nz_pad = self.pg.nz
            k = 2 * self.cg.q_max
            while nz_pad - nz_core < 2:
                if nz_core <= 0:
                    raise ValueError("can't fit given dimensions, nz_core = 0")
                nz_core = base.AStackGeometry.get_cyclic_nz(
                    nz_core - k, k, False)[0]
            self._nz_core = nz_core
        return self._nz_core

    @property
    def nz_top(self) -> int:
        if self._nz_top is None:
            self._nz_top = self.nz_bottom
        return self._nz_top

    def __init__(self, a0, diameter, length, period):
        cg = stacks.FCCTwin.geometry(a0, diameter, length, period,
                                     z_periodic=True)
        pg = stacks.FCCPristine111.geometry(a0, diameter, length,
                                            z_periodic=False)
        super().__init__(cg, pg)
        self.p = cg.p
        self.q_max = cg.get_q_max(period)


class FCCTwinFacetedPaddedG(base.APaddedStackGeometry):
    @property
    def nz_bottom(self) -> int:
        if self._nz_bottom is None:
            self._nz_bottom = ((self.pg.nz - self.nz_core) // 2) // 3 * 3
        return self._nz_bottom

    @property
    def nz_core(self) -> int:
        if self._nz_core is None:
            nz_core = self.cg.nz
            nz_pad = self.pg.nz
            k = 2 * self.cg.q_max
            while nz_pad - nz_core < 2:
                if nz_core <= 0:
                    raise ValueError("can't fit given dimensions, nz_core = 0")
                nz_core = base.AStackGeometry.get_cyclic_nz(
                    nz_core - k, k, False)[0]
            self._nz_core = nz_core
        return self._nz_core

    @property
    def nz_top(self) -> int:
        if self._nz_top is None:
            self._nz_top = self.nz_bottom
        return self._nz_top

    def __init__(self, a0, diameter, length, period):
        cg = stacks.FCCTwinFaceted.geometry(a0, diameter, length, period)
        pg = stacks.FCCPristine111.geometry(a0, diameter, length)
        super().__init__(cg, pg)
        self.p = cg.p
        self.q_max = cg.get_q_max(period)
