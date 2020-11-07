import numpy as np

from nwlattice.base import APointPlane, AStackLattice
from nwlattice.planes import HexPlane, TwinPlane, SquarePlane
from math import sqrt


class CustomStack(AStackLattice):
    def __init__(self, planes, dz, dxy):
        super().__init__()
        for plane in planes:
            if isinstance(plane, APointPlane):
                self._planes.append(plane)
            else:
                raise TypeError("all items in planes list must be PointPlanes")

        self._dz = np.reshape(dz, (self.nz, 3))
        self._dxy = np.reshape(dxy, (self.nz, 3))
        self._v_center_com = np.zeros(3)

    @classmethod
    def fcc_111_pristine(cls, nz, p):
        # construct smallest list of unique planes
        scale = 1 / sqrt(2)
        unit_dxy = np.array([0.35355339, 0.20412415, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True

        # construct whole list of planes
        ROOT3 = np.sqrt(3)
        planes = []
        dz = np.zeros((nz, 3))
        dxy = np.zeros((nz, 3))
        for i in range(nz):
            planes.append(base_planes[i % 3])
            dz[i][2] = i / ROOT3
            dxy[i] += (i % 3) * unit_dxy

        stk = cls(planes, dz, dxy)
        stk._v_center_com = -unit_dxy
        return stk

    @classmethod
    def fcc_100_pristine(cls, nz, r):
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

        return cls(planes, dz, dxy)

    @classmethod
    def fcc_111_twin(cls, nz, p, q0, q_max):
        # obtain cycle of `q` indices for comprising TwinPlanes
        q_cycle = CustomStack._get_twinstack_q_cycle(nz, q0, q_max)

        # construct whole list of planes
        scale = 1 / np.sqrt(2)
        ROOT3 = np.sqrt(3)
        planes = [TwinPlane(p, q, scale=scale) for q in q_cycle]
        dz = np.zeros((nz, 3))
        dxy = np.zeros((nz, 3))
        for i in range(nz):
            dz[i][2] = i / ROOT3

        return cls(planes, dz, dxy)

    @classmethod
    def fcc_111_smooth_twin(cls, nz, p, index):
        index = set([int(j) for j in index])

        # construct smallest list of unique planes
        scale = 1 / sqrt(2)
        unit_dxy = np.array([0.35355339, 0.20412415, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True

        # construct whole list of planes
        ROOT3 = np.sqrt(3)
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

        return cls(planes, dz, dxy)

    @classmethod
    def hexagonal_111_pristine(cls, nz, p):
        return CustomStack.fcc_111_twin(nz, p, q0=0, q_max=1)

    @classmethod
    def mixed_phase_fcc_hexagonal(cls, nz, p, index):
        index = set([int(j) for j in index])

        # construct smallest list of unique planes
        scale = 1 / sqrt(2)
        unit_dxy = np.array([0.35355339, 0.20412415, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True

        # construct whole list of planes
        ROOT3 = np.sqrt(3)
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

        return cls(planes, dz, dxy)

    @staticmethod
    def _get_twinstack_q_cycle(nz, q0, q_max):
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

    @property
    def dz(self):
        return self._dz

    @property
    def dxy(self):
        return self._dxy

    def get_points(self, t):
        # set up lattice points
        pts = np.zeros((self.N, 3))
        n = 0
        for i, plane in enumerate(self.planes):
            pts[n:(n + plane.N)] = (
                    plane.get_points(center=True)
                    + self.dxy[i]
                    + self.dz[i]
            )
            n += plane.N

        # populate lattice points with basis points of type `t`
        atom_pts = np.zeros((self.N * len(self.basis[t]), 3))
        n = 0
        for bpt in self.basis[t]:
            nb = 0
            for i in range(self.nz):
                plane = self.planes[i]
                atom_pts[n:(n + plane.N)] = pts[nb:(nb + plane.N)] + bpt
                nb += plane.N
                n += plane.N

        return atom_pts + self._v_center_com

    def write_map(self, file_path):
        raise NotImplementedError

    @property
    def D(self):
        """returns largest plane diameter"""
        if self._D is None:
            D = self.planes[0].D
            for plane in self.planes[1:]:
                if plane.D > D:
                    D = plane.D
            self._D = D
        return self._D

    @property
    def area(self):
        """returns average plane area"""
        if self._area is None:
            sum_area = 0
            n = 0
            for plane in self.planes:
                sum_area += plane.area
                n += 1
            self._area = sum_area / n
        return self._area
