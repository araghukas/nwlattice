import numpy as np

from nwlattice.utilities import ROOT2, ROOT3
from nwlattice.base import AStackLattice
from nwlattice.planes import HexPlane, TwinPlane, SquarePlane


class FCCPristine111(AStackLattice):
    def __init__(self, nz, p):
        # construct smallest list of unique planes
        scale = 1 / ROOT2
        unit_dxy = np.array([0.35355339, 0.20412415, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True

        # construct whole list of planes
        planes = []
        dz = np.zeros((nz, 3))
        dxy = np.zeros((nz, 3))
        for i in range(nz):
            planes.append(base_planes[i % 3])
            dz[i][2] = i / ROOT3
            dxy[i] += (i % 3) * unit_dxy

        self._v_center_com = -unit_dxy
        super().__init__(planes, dz, dxy)

    @property
    def type_name(self):
        return "FCCPristine111"

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None,
                        p=None, nz=None, z_periodic=True):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        if z_periodic:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p
        stk = cls(nz, p)
        stk._scale = a0
        return stk

    def write_map(self, file_path):
        raise


class FCCPristine100(AStackLattice):
    def __init__(self, nz, r):
        # construct smallest list of unique planes
        base_planes = [
            SquarePlane(r, even=True, scale=1.0),
            SquarePlane(r, even=False, scale=1.0)
        ]

        # construct whole list of planes
        planes = []
        dz = np.zeros((nz, 3))
        dxy = np.zeros((nz, 3))
        for i in range(nz):
            planes.append(base_planes[i % 2])
            dz[i][2] = i * 0.5
        super().__init__(planes, dz, dxy)

    @property
    def type_name(self):
        return "FCCPristine100"

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None,
                        r=None, nz=None, z_periodic=True):
        if diameter is None and r is None:
            raise ValueError("must specify either `diameter` or `r`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        nz = 1 + round(2. * length / a0) if nz is None else nz
        if z_periodic:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))
        r = SquarePlane.get_index_for_diameter(a0, diameter) if r is None else r
        stk = cls(nz, r)
        stk._scale = a0
        return stk

    def write_map(self, file_path):
        raise NotImplementedError


class FCCTwin(AStackLattice):
    def __init__(self, nz, p, index):
        index = set([int(j) for j in index])

        # construct smallest list of unique planes
        scale = 1 / ROOT2
        unit_dxy = np.array([0.35355339, 0.20412415, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True
        # construct whole list of planes
        planes = []
        dz = np.zeros((nz, 3))
        dxy = np.zeros((nz, 3))

        j = 0
        for i in range(nz):
            if i in index:
                j += 1
            planes.append(base_planes[j % 3])
            dxy[i] += (j % 3) * unit_dxy
            dz[i][2] = i / ROOT3
            j += 1
        super().__init__(planes, dz, dxy)

    @property
    def type_name(self):
        return "FCCTwin"

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None, period=None,
                        index=None, p=None, nz=None, q_max=None,
                        z_periodic=True):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")
        if period is None and q_max is None and index is None:
            raise ValueError(
                "must specify either `period`, `q_max`, or `index`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p
        if q_max or period:
            q_max = round(ROOT3 * period / 2. / a0) if q_max is None else q_max

        if z_periodic and q_max:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz, q_max)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))

        if index is not None:
            pass
        elif period or q_max:
            index = []
            i_period = q_max
            include = True
            for i in range(nz):
                if i % i_period == 0:
                    include = not include
                if include:
                    index.append(i)
        else:
            index = []

        stk = cls(nz, p, index)
        stk._scale = a0
        if q_max:
            stk._P = 2 * a0 * q_max / ROOT3
        return stk

    @staticmethod
    def get_cyclic_nz(*args):
        nz, q_max = args
        k = 2 * q_max
        nlo = (nz // k) * k
        nhi = ((nz + k) // k) * k

        if nlo == 0:
            return nhi
        elif (nz - nlo) < (nhi - nz):
            return nlo
        else:
            return nhi

    def write_map(self, file_path):
        raise NotImplementedError


class FCCTwinFaceted(AStackLattice):
    def __init__(self, nz, p, q0, q_max):
        # obtain cycle of `q` indices for comprising TwinPlanes
        q_cycle = self.get_q_cycle(nz, q0, q_max)

        # construct whole list of planes
        scale = 1 / ROOT2
        planes = [TwinPlane(p, q, scale=scale) for q in q_cycle]
        dz = np.zeros((nz, 3))
        dxy = np.zeros((nz, 3))
        for i in range(nz):
            dz[i][2] = i / ROOT3
        super().__init__(planes, dz, dxy)

    @property
    def type_name(self):
        return "FCCTwinFaceted"

    @property
    def q0(self):
        return self.planes[0].q

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None, period=None,
                        p=None, nz=None, q_max=None, q0=0, q_max_auto=True,
                        z_periodic=True):

        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")
        if period is None and q_max is None:
            raise ValueError("must specify either `period` or `q_max`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p
        q_max = round(ROOT3 * period / 2 / a0) if q_max is None else q_max

        if z_periodic:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz, q_max)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))

        if q_max >= p:
            q_max = p - 1
            if q_max_auto:
                old_period = period
                period = 2. * a0 * q_max / ROOT3
                print("Period is too large, corrected: %f --> %f "
                      % (old_period, period))
            else:
                raise ValueError("period {:f} is too large for given "
                                 "diameter {:f}\n Maximum period is {:f} "
                                 "(or `q_max = {:d}`)"
                                 .format(period, diameter,
                                         2. * a0 * q_max / ROOT3, q_max))

        stk = cls(nz, p, q0, q_max)
        stk._scale = a0
        stk._P = 2 * a0 * q_max / ROOT3
        return stk

    @staticmethod
    def get_cyclic_nz(*args):
        """same as FCCTwin"""
        nz, q_max = args
        k = 2 * q_max
        nlo = (nz // k) * k
        nhi = ((nz + k) // k) * k

        if nlo == 0:
            return nhi
        elif (nz - nlo) < (nhi - nz):
            return nlo
        else:
            return nhi

    @staticmethod
    def get_q_cycle(nz, q0, q_max):
        q_cycle = [q0]
        step = 1
        count = 0
        while count < nz - 1:
            next_q = q_cycle[-1] + step
            q_cycle.append(next_q)
            if next_q == q_max or next_q == 0:
                step *= -1
            count += 1
        return q_cycle

    def write_map(self, file_path):
        raise NotImplementedError


class HexagonalPristine111(AStackLattice):
    def __init__(self, nz, p):
        # obtain cycle of `q` indices for comprising TwinPlanes
        q_cycle = FCCTwinFaceted.get_q_cycle(nz, 0, 1)

        # construct whole list of planes
        scale = 1 / ROOT2
        planes = [TwinPlane(p, q, scale=scale) for q in q_cycle]
        dz = np.zeros((nz, 3))
        dxy = np.zeros((nz, 3))
        for i in range(nz):
            dz[i][2] = i / ROOT3
        super().__init__(planes, dz, dxy)

    @property
    def type_name(self):
        return "HexagonalPristine111"

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None,
                        p=None, nz=None, z_periodic=True):

        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p

        if z_periodic:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))

        stk = cls(nz, p)
        stk._scale = a0
        return stk

    def write_map(self, file_path):
        raise NotImplementedError


class FCCHexagonalMixed(AStackLattice):
    def __init__(self, nz, p, index):
        index = set([int(j) for j in index])

        # construct smallest list of unique planes
        scale = 1 / ROOT2
        unit_dxy = np.array([0.35355339, 0.20412415, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True

        # construct whole list of planes
        planes = []
        dz = np.zeros((nz, 3))
        dxy = np.zeros((nz, 3))

        j = 0
        for i in range(nz):
            if i in index:
                j += -2 * (i % 2)
            planes.append(base_planes[j % 3])
            dxy[i] += (j % 3) * unit_dxy
            dz[i][2] = i / ROOT3
            j += 1
        super().__init__(planes, dz, dxy)
        self._fraction = len(index) / nz

    @property
    def type_name(self):
        return "FCCHexagonalMixed"

    @property
    def fraction(self):
        return self._fraction

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None, index=None,
                        fraction=None, p=None, nz=None):
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p

        if index is not None:
            index = []
        elif fraction is not None:
            index = []
            for i in range(nz):
                if np.random.uniform(0, 1) < fraction:
                    index.append(i)
        else:
            index = []
        stk = cls(nz, p, index)
        stk._scale = a0
        return stk

    def write_map(self, file_path):
        raise NotImplementedError
