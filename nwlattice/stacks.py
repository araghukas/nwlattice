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
    def from_dimensions(cls, a0, diameter, length):
        nz = round(ROOT3 * length / a0)
        p = HexPlane.get_index_for_diameter(a0, diameter)
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
    def from_dimensions(cls, a0, diameter, length):
        nz = 1 + round(2. * length / a0)
        r = SquarePlane.get_index_for_diameter(a0, diameter)
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
    def from_dimensions(cls, a0, diameter, length, period=None, index=None):
        nz = round(ROOT3 * length / a0)
        p = HexPlane.get_index_for_diameter(a0, diameter)
        if index is not None:
            index = index
        elif period is not None:
            index = []
            i_period = round(ROOT3 * period / 2 / a0)
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
        return stk

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

    @classmethod
    def from_dimensions(cls, a0, diameter, length, period, q0=0):
        nz = round(ROOT3 * length / a0)
        p = HexPlane.get_index_for_diameter(a0, diameter)
        q_max = round(ROOT3 * period / 2 / a0)
        stk = cls(nz, p, q0, q_max)
        stk._scale = a0
        return stk

    def write_map(self, file_path):
        raise NotImplementedError

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
    def from_dimensions(cls, a0, diameter, length):
        nz = round(ROOT3 * length / a0)
        p = HexPlane.get_index_for_diameter(a0, diameter)
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
    def from_dimensions(cls, a0, diameter, length, index=None, fraction=None):
        nz = round(ROOT3 * length / a0)
        p = HexPlane.get_index_for_diameter(a0, diameter)
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
