import nwlattice.dimensions_base as base
import nwlattice.stacks as stacks

from nwlattice.planes import HexPlane, SquarePlane
from nwlattice.utilities import ROOT3


class FCCPristine111GP(base.AStackGeometry):
    def z_index(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def validate_args(self, diameter, length, p, nz):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, diameter, length, p=None, nz=None, z_periodic=False):
        super().__init__(a0)
        self.validate_args(diameter, length, p, nz)
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 3)
        self.nz = nz
        self.p = self.xy_index(diameter) if p is None else p

        self.length = length
        self.diameter = diameter


class FCCPristine100GP(base.AStackGeometry):
    def z_index(self, length: float) -> int:
        return 1 + round(2 * length / self.a0)

    def xy_index(self, side_length: float) -> int:
        return SquarePlane.index_for_diameter(self.a0, side_length)

    def validate_args(self, side_length, length, r, nz):
        if side_length is None and r is None:
            raise ValueError("must specify either `diameter` or `r`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, side_length, length, r=None, nz=None,
                 z_periodic=False):
        super().__init__(a0)
        self.validate_args(side_length, length, r, nz)
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 2)
        self.nz = nz
        self.r = self.xy_index(side_length) if r is None else r

        self.length = length
        self.side_length = side_length


class FCCTwin111GP(base.AStackGeometry):
    def z_index(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def get_q_max(self, period: float) -> int:
        return round(ROOT3 * period / 2 / self.a0)

    def get_period(self) -> float:
        return 2 * self.a0 * self.q_max / ROOT3

    def validate_args(self, diameter, length, period, index, p, nz, q_max):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")
        if period is None and q_max is None and index is None:
            raise ValueError(
                "must specify either `period`, `q_max`, or `index`")

    def __init__(self, a0, diameter, length, period, index=None, p=None,
                 nz=None, q_max=None, z_periodic=False):
        super().__init__(a0)
        self.validate_args(diameter, length, period, index, p, nz, q_max)
        self.q_max = self.get_q_max(period) if q_max is None else q_max
        nz = self.z_index(length) if nz is None else nz
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


class FCCTwinFacetedGP(base.AStackGeometry):

    def z_index(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def get_q_max(self, period: float) -> int:
        return round(ROOT3 * period / 2 / self.a0)

    def get_period(self) -> float:
        return 2 * self.a0 * self.q_max / ROOT3

    def validate_args(self, diameter, length, period, p, nz, q_max):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")
        if period is None and q_max is None:
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
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 2 * self.q_max)
        self.nz = nz
        self.q0 = q0 if 0 <= q0 <= self.q_max else 0

        self.diameter = diameter
        self.length = length
        self.period = self.get_period()


class HexPristine0001GP(base.AStackGeometry):
    def z_index(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def validate_args(self, diameter, length, p, nz):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, diameter, length, p=None, nz=None, z_periodic=False):
        super().__init__(a0)
        self.validate_args(diameter, length, p, nz)
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = base.AStackGeometry.get_cyclic_nz(nz, 2)
        self.nz = nz
        self.p = self.xy_index(diameter) if p is None else p

        self.length = length
        self.diameter = diameter


class FCCTwinPaddedStackGeometry(base.APaddedStackGeometry):
    @property
    def nz_bottom(self) -> int:
        if self._nz_bottom is None:
            self._nz_bottom = ((self.p_sg.nz - self.nz_core) // 2) // 3 * 3
        return self._nz_bottom

    @property
    def nz_core(self) -> int:
        if self._nz_core is None:
            nz_core = self.c_sg.nz
            nz_pad = self.p_sg.nz
            k = 2 * self.c_sg.q_max
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
        c_sg = stacks.FCCTwin.sg(a0, diameter, length, period, z_periodic=True)
        p_sg = stacks.FCCPristine111.sg(a0, diameter, length, z_periodic=False)
        super().__init__(c_sg, p_sg)
        self.p = c_sg.p
        self.q_max = c_sg.get_q_max(period)
