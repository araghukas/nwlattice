from abc import ABC, abstractmethod
import numpy as np

"""
Objects that generate twinning patterns for `base.ANanowireLatticeArbitrary` subclass
"""


class ATwinPlanesIndex(ABC):
    """
    Useless base class just to clarify interface
    """

    def __init__(self):
        self.nz_prev = None

    @abstractmethod
    def __call__(self, **kwrgs) -> (int, list):
        """return new nz and index; wire size unaffected in bool(nz) is False"""
        return 0, []


class Empty(ATwinPlanesIndex):
    def __call__(self, *args, **kwargs) -> (int, list):
        return 0, []


class Manual(ATwinPlanesIndex):
    """
    Specify twin locations with iterable `index` and (optional) adjust `nz`
    """

    def __init__(self, index, nz=None):
        super().__init__()
        self.index = index
        self.nz = nz

    def __call__(self, *args):
        return self.nz, self.index


class LinearDecrease(ATwinPlanesIndex):
    """
    Consider a general case of the sum of integers 1 to n:

        1 + 2 + 3 + ... + n  =  n(n + 1)/2

    where we ascend in steps of m:

        1 + (1 + m) + (1 + 2(m)) + ... + (1 + (k - 1)m)  =  k + mk(k - 1)/2

    Twice the sum of the first sequence is the number of monolayers
    in a nanowire structure where the twinning period decreases by
    one monolayer after each cycle.

    Similarly, twice the sum--because there are two copies of each twin segment--
    of the second sequence is the number of monolayers (nz) in a nanowire
    structure where the twinning period decreases by m after each cycle.

    There are n terms in the first sum and k terms in the second

    Inverting the RHS of the second expression allows you to determine the
    starting (maximum) value of the twin length (q), such that a length decrease of m
    after each cycle results in exactly nz segments upon reaching the smallest
    possible twin segment length (1).

    Exactness is not possible at arbitrary nz_,
    but we can take the nearest working nz.
    """

    def __init__(self, m: int, r: int = None, q_min: int = None, q_max: int = None):
        super().__init__()
        if m < 1:
            raise ValueError("parameter `m` less than 1")
        self.m = m

        if q_min and q_min < 1:
            raise ValueError("parameter `q_min` less than 1")
        self.q_min = q_min

        if q_max and q_max < 1:
            raise ValueError("parameter `q_max` less than 1")
        self.q_max = q_max

        if r and r < 1:
            raise ValueError("parameter `r` less than 1; can't repeat less than once")
        self.r = r if r else 1

    def approximate(self, nz_) -> (int, int):
        # solve nz_ = 2 * (k + mk(k - 1)/2) for number of twin segments
        # determine q and nz
        if self.q_max:
            k_ = (self.q_max - 1) / self.m + 1
        else:
            a = self.m * self.r
            b = (2 - self.m) * self.r
            c = -nz_
            k_ = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        kf = int(k_)
        kr = round(k_)

        nzf = (2 * kf + self.m * kf * (kf - 1)) * self.r
        nzr = (2 * kr + self.m * kr * (kr - 1)) * self.r

        # choose better approx. and determine largest twin length
        if abs(nz_ - nzf) <= abs(nz_ - nzr):
            q = 1 + (kf - 1) * self.m
            nz = nzf
        else:
            q = 1 + (kr - 1) * self.m
            nz = nzr

        return q, nz

    def __call__(self, nz_):
        self.nz_prev = nz_
        q, nz = self.approximate(nz_)

        top = 0
        index = []
        while q > 0 and self._q_min_satisfied(q):
            for i in range(2 * self.r):
                index.append(top + q - 1)
                top += q
            q -= self.m

        if not self._q_min_satisfied(q):
            nz = top

        return nz, index

    def _q_min_satisfied(self, q):
        if self.q_min is None:
            return True
        else:
            return q >= self.q_min
