from nwlattice.base import StackGeometryParser
from nwlattice.planes import *
from nwlattice.utilities import ROOT3


class FCC111GP(StackGeometryParser):
    def z_index(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def validate_args(self, diameter, length, p, nz):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, diameter, length, p, nz, z_periodic):
        super().__init__(a0)
        self.validate_args(diameter, length, p, nz)
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = StackGeometryParser.get_cyclic_z_index(nz, 3)
        self.nz = nz
        self.p = self.xy_index(diameter) if p is None else p

        self.length = length
        self.diameter = diameter


class FCC100GP(StackGeometryParser):
    def z_index(self, length: float) -> int:
        return 1 + round(2 * length / self.a0)

    def xy_index(self, side_length: float) -> int:
        return SquarePlane.index_for_diameter(self.a0, side_length)

    def validate_args(self, side_length, length, r, nz):
        if side_length is None and r is None:
            raise ValueError("must specify either `diameter` or `r`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, side_length, length, r, nz, z_periodic):
        super().__init__(a0)
        self.validate_args(side_length, length, r, nz)
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = StackGeometryParser.get_cyclic_z_index(nz, 2)
        self.nz = nz
        self.r = self.xy_index(side_length) if r is None else r

        self.length = length
        self.side_length = side_length


class TwinFCC111GP(StackGeometryParser):
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

    def __init__(self, a0, diameter, length, period, index, p, nz, q_max,
                 z_periodic):
        super().__init__(a0)
        self.validate_args(diameter, length, period, index, p, nz, q_max)
        self.q_max = self.get_q_max(period) if q_max is None else q_max
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = StackGeometryParser.get_cyclic_z_index(nz, 2 * self.q_max)
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


class FacetedTwinFCC111GP(StackGeometryParser):

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

    def __init__(self, a0, diameter, length, period, p, nz, q_max, q0,
                 z_periodic):
        super().__init__(a0)
        self.validate_args(diameter, length, period, p, nz, q_max)
        self.p = self.xy_index(diameter)
        q_max = self.get_q_max(period) if q_max is None else q_max
        if q_max >= self.p:
            q_max = self.p - 1
        self.q_max = q_max
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = StackGeometryParser.get_cyclic_z_index(nz, 2 * self.q_max)
        self.nz = nz
        self.q0 = q0 if 0 <= q0 <= self.q_max else 0

        self.diameter = diameter
        self.length = length
        self.period = self.get_period()


class Hexagonal0001GP(StackGeometryParser):
    def z_index(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.index_for_diameter(self.a0, xy_length)

    def validate_args(self, diameter, length, p, nz):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

    def __init__(self, a0, diameter, length, p, nz, z_periodic):
        super().__init__(a0)
        self.validate_args(diameter, length, p, nz)
        nz = self.z_index(length) if nz is None else nz
        if z_periodic:
            nz = StackGeometryParser.get_cyclic_z_index(nz, 2)
        self.nz = nz
        self.p = self.xy_index(diameter) if p is None else p

        self.length = length
        self.diameter = diameter
