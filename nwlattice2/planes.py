import numpy as np

from nwlattice2.utilities import ROOT3
from nwlattice2.sizes import PlaneSize
import nwlattice2.base as base


class FCCa(base.APointPlane):
    def __init__(self, scale, n_xy=None, width=None, theta=None):
        size = PlaneSize(scale, n_xy, width)
        size._n_xy_func = self.get_n_xy
        size._width_func = self.get_width
        size._area_func = self.get_area
        vectors = self.get_vectors()
        super().__init__(size, vectors, theta)

    @staticmethod
    def get_n_xy(scale, width):
        return round(width / ROOT3 / scale + 0.5)

    @staticmethod
    def get_width(scale, n_xy):
        return scale * ROOT3 / 2 * (2 * n_xy - 1)

    @staticmethod
    def get_area(scale, n_xy):
        return scale**2 * (ROOT3 / 4) * (1 + 6 * n_xy * (n_xy - 1))

    @staticmethod
    def get_vectors():
        return np.array([[1., 0., 0.], [-.5, ROOT3 / 2., 0.]])

    @property
    def N(self):
        if self._N is None:
            self._N = 3 * self.size.n_xy**2
        return self._N

    @property
    def com(self):
        if self._com is None:
            self._com = (self.size.n_xy - 1) * np.sum(self.vectors, axis=0)
        return self._com

    def get_points(self):
        pts = np.zeros((self.N, 3))
        i = 0

        # translation vectors
        v1, v2 = self.vectors

        n = 0
        m = self.size.n_xy

        # loop from 1st row to second-widest row
        while n < self.size.n_xy:
            m += 1
            for r in range(0, m):
                pts[i] = r * v1 + n * v2
                i += 1
            n += 1

        # loop from widest row to last row
        s = 1
        while n < 2 * self.size.n_xy:
            m -= 1
            for r in range(0, m):
                pts[i] = (r + s) * v1 + n * v2
                i += 1
            s += 1
            n += 1

        pts -= self.com
        return pts


class FCCb(FCCa):
    @staticmethod
    def get_n_xy(scale, width):
        return round(width / ROOT3 / scale + 1)

    @staticmethod
    def get_width(scale, n_xy):
        return scale * ROOT3 * (n_xy - 1)

    @staticmethod
    def get_area(scale, n_xy):
        return scale**2 * (ROOT3 / 4) * 6 * (n_xy - 1)**2

    @property
    def N(self):
        if self._N is None:
            n_xy = self.size.n_xy
            self._N = 1 + 3 * n_xy * (n_xy - 1)
        return self._N

    def get_points(self):
        pts = np.zeros((self.N, 3))
        i = 0

        # translation vectors
        v1, v2 = self.vectors

        n = 0
        m = self.size.n_xy - 1  # start with this many +1 points in 0th row

        # loop from 1st row to second-widest row
        while n < self.size.n_xy:
            m += 1
            for r in range(0, m):
                pts[i] = r * v1 + n * v2
                i += 1
            n += 1

        # loop from widest row to last row
        s = 1
        while n < (2 * self.size.n_xy - 1):
            m -= 1
            for r in range(0, m):
                pts[i] = (r + s) * v1 + n * v2
                i += 1
            s += 1
            n += 1

        pts -= self.com
        return pts


class FCCc(FCCa):
    def __init__(self, scale, n_xy=None, width=None):
        super().__init__(scale, n_xy, width, theta=np.pi)


class SqFCCa(base.APointPlane):

    def __init__(self, scale, n_xy=None, width=None):
        size = PlaneSize(scale, n_xy, width)
        size._n_xy_func = self.get_n_xy
        size._width_func = self.get_width
        size._area_func = self.get_area
        vectors = self.get_vectors()
        super().__init__(size, vectors)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return 1 + round(width / scale)

    @staticmethod
    def get_width(scale: float, n_xy: int) -> float:
        return scale * (n_xy - 1)

    @staticmethod
    def get_area(scale: float, n_xy: int) -> float:
        return (scale * (n_xy - 1))**2

    @staticmethod
    def get_vectors() -> np.ndarray:
        return np.array([[1., 0., 0.], [0., 1., 0.]])

    @property
    def N(self):
        if self._N is None:
            self._N = self.size.n_xy**2 + (self.size.n_xy - 1)**2
        return self._N

    @property
    def com(self):
        if self._com is None:
            v0, v1 = self.vectors
            self._com = (self.size.n_xy - 1) * (v0 + v1) / 2
        return self._com

    def get_points(self) -> np.ndarray:
        pts = np.zeros((self.N, 3))

        # translation vectors
        v0, v1 = self.vectors
        u0, u1 = self.vectors / 2.

        i = 0
        for a in range(self.size.n_xy):
            for b in range(self.size.n_xy):
                pts[i] = a * v0 + b * v1
                i += 1
        for c in range(self.size.n_xy - 1):
            for d in range(self.size.n_xy - 1):
                pts[i] = (u0 + u1) + c * v0 + d * v1
                i += 1

        pts -= self.com
        return pts


class SqFCCb(SqFCCa):
    @property
    def N(self):
        if self._N is None:
            self._N = self._N = 2 * self.size.n_xy * (self.size.n_xy - 1)
        return self._N

    def get_points(self) -> np.ndarray:
        pts = np.zeros((self.N, 3))

        # translation vectors
        v0, v1 = self.vectors
        u0, u1 = self.vectors / 2.

        i = 0
        for a in range(self.size.n_xy - 1):
            for b in range(self.size.n_xy):
                pts[i] = u0 + a * v0 + b * v1
                i += 1
        for c in range(self.size.n_xy):
            for d in range(self.size.n_xy - 1):
                pts[i] = u1 + c * v0 + d * v1
                i += 1

        pts -= self.com
        return pts
