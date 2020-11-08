import numpy as np

from nwlattice import ROOT2, ROOT3
from nwlattice.base import APointPlane
from nwlattice.quaternion import qrotate


class HexPlane(APointPlane):

    def __init__(self, p, scale, even=True, com_offset=None):
        super().__init__(scale)
        if p > 1:
            self._p = int(p)
        else:
            raise ValueError("HexPlane index `p` must be > 1")

        if com_offset is None:
            self._com_offset = np.zeros(3)
        else:
            self._com_offset = np.reshape(com_offset, (3,))

        self._even = even
        self._delta = None
        self._inverted = False

    @property
    def p(self):
        return self._p

    @property
    def even(self):
        return self._even

    @property
    def N(self):
        if self._N is None:
            if self.even:
                self._N = 1 + 3 * self.p * (self.p - 1)
            else:
                self._N = 3 * self.p**2
        return self._N

    @property
    def D(self):
        if self._D is None:
            if self.even:
                self._D = 2 * (self.p - 1) * ROOT3 / ROOT2 / 2
            else:
                self._D = (2 * self.p - 1) * ROOT3 / ROOT2 / 2
        return self._D

    @property
    def area(self):
        if self._area is None:
            if self.even:
                self._area = 0.5 * (3 * (self.p - 1)) * (self.p - 1) * ROOT3
            else:
                self._area = 0.25 * (
                        ((self.p - 1) + (2 * self.p - 1)) * self.p * ROOT3
                        + (self.p + (2 * self.p - 1)) * (self.p - 1) * ROOT3
                )
        return self._area

    @property
    def vectors(self):
        if self._vectors is None:
            self._vectors = super().hx_vectors
        return self._vectors

    @property
    def delta(self):
        if self._delta is None:
            if self._even:
                self._delta = super().ehex_delta
            else:
                self._delta = super().ohex_delta
        return self._delta

    @property
    def com(self):
        if self._com is None:
            self._com = (self.p - 1) * (self.vectors[0] + self.vectors[1])
            self._com += self._com_offset
        return self._com

    @property
    def inverted(self):
        return self._inverted

    @inverted.setter
    def inverted(self, b):
        if type(b) is bool:
            self._inverted = b
        else:
            raise TypeError("can't assign non boolean to `inverted` property")

    def get_points(self, center=True):
        if self.even:
            pts = self._get_points_even(center)
        else:
            pts = self._get_points_odd(center)

        if self.inverted:
            pts = qrotate(pts, [0, 0, 1], np.pi)
        return self.scale * pts

    def _get_points_even(self, center=True):
        pts = np.zeros((self.N, 3))
        i = 0

        # translation vectors
        v1, v2 = self.vectors

        n = 0
        m = self.p - 1  # start with this many +1 points in 0th row

        # loop from 1st row to second-widest row
        while n < self.p:
            m += 1
            for r in range(0, m):
                pts[i] = r * v1 + n * v2
                i += 1
            n += 1

        # loop from widest row to last row
        s = 1
        while n < (2 * self.p - 1):
            m -= 1
            for r in range(0, m):
                pts[i] = (r + s) * v1 + n * v2
                i += 1
            s += 1
            n += 1

        if center:
            pts = pts - self.com

        return pts

    def _get_points_odd(self, center=True):
        pts = np.zeros((self.N, 3))
        i = 0

        # translation vectors
        v1, v2 = self.vectors

        n = 0
        m = self.p

        # loop from 1st row to second-widest row
        while n < self.p:
            m += 1
            for r in range(0, m):
                pts[i] = r * v1 + n * v2
                i += 1
            n += 1

        # loop from widest row to last row
        s = 1
        while n < 2 * self.p:
            m -= 1
            for r in range(0, m):
                pts[i] = (r + s) * v1 + n * v2
                i += 1
            s += 1
            n += 1

        if center:
            pts = pts - self.com

        return pts

    @staticmethod
    def get_index_for_diameter(scale, D):
        p = round(1. + ROOT2 / ROOT3 * D / scale)
        return p


class SquarePlane(APointPlane):
    def __init__(self, r, scale, even=True):
        super().__init__(scale)
        if r > 1:
            self._r = int(r)
        else:
            raise ValueError("SquarePlane index `r` must be > 1")
        self._even = even

    @property
    def r(self):
        return self._r

    @property
    def even(self):
        return self._even

    @property
    def N(self):
        if self._N is None:
            if self.even:
                self._N = self.r**2 + (self.r - 1)**2
            else:
                self._N = 2 * self.r * (self.r - 1)
        return self._N

    @property
    def D(self):
        if self._D is None:
            self._D = self.r - 1
        return self._D

    @property
    def area(self):
        if self._area is None:
            self._area = (self.r - 1)**2
        return self._area

    @property
    def vectors(self):
        return super().sq_vectors

    @property
    def com(self):
        if self._com is None:
            self._com = (self.r - 1) * (self.vectors[0] + self.vectors[1]) / 2
        return self._com

    def get_points(self, center=True):
        if self.even:
            pts = self._get_points_even()
        else:
            pts = self._get_points_odd()

        if center:
            pts = pts - self.com

        return self.scale * pts

    def _get_points_even(self):
        pts = np.zeros((self.N, 3))

        # translation vectors
        v1, v2 = self.vectors
        u1, u2 = self.vectors / 2.

        i = 0
        for a in range(self.r):
            for b in range(self.r):
                pts[i] = a * v1 + b * v2
                i += 1
        for c in range(self.r - 1):
            for d in range(self.r - 1):
                pts[i] = (u1 + u2) + c * v1 + d * v2
                i += 1

        return pts

    def _get_points_odd(self):
        pts = np.zeros((self.N, 3))

        # translation vectors
        v1, v2 = self.vectors
        u1, u2 = self.vectors / 2.

        i = 0
        for a in range(self.r - 1):
            for b in range(self.r):
                pts[i] = u1 + a * v1 + b * v2
                i += 1
        for c in range(self.r):
            for d in range(self.r - 1):
                pts[i] = u2 + c * v1 + d * v2
                i += 1

        return pts

    @staticmethod
    def get_index_for_diameter(scale, D):
        r = 1 + round(D / scale)
        return r


class TwinPlane(APointPlane):
    def __init__(self, p, q, scale):
        super().__init__(scale)
        if p > 1:
            self._p = int(p)
        else:
            raise ValueError("TwinPlane index 'p' must be > 1")

        if 0 <= q < p:
            self._q = int(q)
        else:
            raise ValueError("TwinPlane index 'q' must be less than index `p`")

    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    @property
    def N(self):
        if self._N is None:
            self._N = 1 + 3 * self.p * (self.p - 1) - self.q**2
        return self._N

    @property
    def D(self):
        if self._D is None:
            self._D = 2 * (self.p - 1) * ROOT3 / ROOT2 / 2.
        return self._D

    @property
    def area(self):
        if self._area is None:
            self._area = ((3 * (self.p - 1) + self.q) * (self.p - 1 - self.q)
                          + (3 * (self.p - 1) - self.q) * (self.p - 1 + self.q)
                          ) * ROOT3 / 4
        return self._area

    @property
    def vectors(self):
        return super().hx_vectors

    @property
    def delta(self):
        return super().ehex_delta

    @property
    def com(self):
        if self._com is None:
            self._com = (2 * self.q * self.delta
                         + (self.p - 1) * (self.vectors[0] + self.vectors[1]))
        return self._com

    def get_points(self, center=True):
        pts = np.zeros((self.N, 3))
        i = 0

        # translation vectors
        v1, v2 = self.vectors

        n = 0
        m = self.p + self.q - 1  # start with this many +1 points in 0th row

        # loop from 1st row to second-widest row
        while n < (self.p - self.q):
            m += 1
            for r in range(0, m):
                pts[i] = r * v1 + n * v2
                i += 1
            n += 1

        # loop from widest row to last row
        s = 1
        while n < (2 * self.p - 1):
            m -= 1
            for r in range(0, m):
                pts[i] = (r + s) * v1 + n * v2
                i += 1
            s += 1
            n += 1

        if center:
            pts = pts - self.com

        return self.scale * pts

    @staticmethod
    def get_index_for_diameter(scale, D):
        p = round(1. + ROOT2 / ROOT3 * D / scale)
        return p
