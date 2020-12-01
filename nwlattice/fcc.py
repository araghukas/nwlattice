from nwlattice.base import Geometry
from nwlattice.planes import *
from nwlattice.utilities import ROOT3


class FaceCenteredCubic111(Geometry):
    def z_index(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.get_index_for_diameter(self.a0, xy_length)

    def parse_dims(self, diameter, length, p, nz, z_periodic) -> tuple:
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        if nz is None:
            nz = self.z_index(length)

        if p is None:
            p = self.xy_index(diameter)

        if z_periodic:
            old_nz = nz
            nz = super().get_cyclic_z_index(nz, 3)
            super().print("forced z periodicity, adjusted nz: %d --> %d"
                          % (old_nz, nz))

        return diameter, length, nz, p


class FaceCenteredCubic100(Geometry):
    def z_index(self, length: float) -> int:
        return 1 + round(2 * length / self.a0)

    def xy_index(self, side_length: float) -> int:
        return SquarePlane.get_index_for_diameter(self.a0, side_length)

    def parse_dims(self, side_length, length, r, nz, z_periodic) -> tuple:
        if side_length is None and r is None:
            raise ValueError("must specify either `diameter` or `r`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        if nz is None:
            nz = self.z_index(length)

        if r is None:
            r = self.xy_index(side_length)

        if z_periodic:
            old_nz = nz
            nz = super().get_cyclic_z_index(nz, 2)
            super().print("forced z periodicity, adjusted nz: %d --> %d"
                          % (old_nz, nz))

        return side_length, length, r, nz


class TwinFaceCenteredCubic111(Geometry):
    def z_index(self, length: float) -> int:
        return round(ROOT3 * length / self.a0)

    def xy_index(self, xy_length: float) -> int:
        return HexPlane.get_index_for_diameter(self.a0, xy_length)

    def q_max(self, period: float) -> int:
        return round(ROOT3 * period / 2 / self.a0)

    def period(self, q_max: int) -> float:
        return 2 * self.a0 * q_max / ROOT3

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

    def parse_dims(self, diameter, length, period, p, nz, q_max,
                   z_periodic, index=None, faceted=False) -> tuple:
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")
        if period is None and q_max is None and index is None:
            raise ValueError(
                "must specify either `period`, `q_max`, or `index`")

        if nz is None:
            nz = self.z_index(length)

        if p is None:
            p = self.xy_index(diameter)

        if q_max is None:
            q_max = self.q_max(period)

        if faceted and q_max >= p:
            q_max = p - 1
            old_period = period
            period = self.period(q_max)
            super().print("Period is too large, corrected: %f --> %f "
                          % (old_period, period))

        if not faceted and index is None:
            index = self.get_index(q_max, nz)

        if z_periodic:
            old_nz = nz
            nz = super().get_cyclic_z_index(nz, 2 * q_max)
            super().print("forced z periodicity, adjusted nz: %d --> %d"
                          % (old_nz, nz))

        if not faceted:
            return diameter, length, period, index, p, nz, q_max
        return diameter, length, period, p, nz, q_max


class Hexagonal0001(FaceCenteredCubic111):

    def parse_dims(self, diameter, length, p, nz, z_periodic) -> tuple:
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        if nz is None:
            nz = self.z_index(length)

        if p is None:
            p = self.xy_index(diameter)

        if z_periodic:
            old_nz = nz
            nz = super().get_cyclic_z_index(nz, 2)
            super().print("forced z periodicity, adjusted nz: %d --> %d"
                          % (old_nz, nz))

        return diameter, length, nz, p
