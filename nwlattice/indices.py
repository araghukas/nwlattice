from abc import ABC, abstractmethod
import numpy as np

"""
Objects that generate twinning patterns for `base.ANanowireLatticeArbitrary` subclass
"""


class APlanesIndex(ABC):
    """
    Useless base class just to clarify interface
    """
    @abstractmethod
    def __call__(self, *args) -> (int, list):
        """return new nz and index; wire size unaffected in bool(nz) is False"""
        return 0, []


class Manual(APlanesIndex):
    """
    Specify twin locations with iterable `index` and (optional) adjust `nz`
    """
    def __init__(self, index, nz=None):
        self.index = index
        self.nz = nz

    def __call__(self, *args):
        return self.nz, self.index


class LinearDecrease(APlanesIndex):
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
    starting value of the twin length (q), such that a length decrease of m
    after each cycle results in exactly nz segments upon reaching the smallest
    possible twin segment length (1).

    Exactness is not possible at arbitrary nz_,
    but we can take the nearest working nz.
    """

    def __init__(self, m: int, q_min: int = None):
        if m < 1:
            raise ValueError("parameter `m` less than 1")
        self.m = m

        if q_min and q_min < 1:
            raise ValueError("parameter `q_min` less than 1")

        self.q_min = q_min
        self.nz_prev = None

    def __call__(self, nz_):
        self.nz_prev = nz_

        # solve nz_ = 2 * (k + mk(k - 1)/2) for number of twin segments
        a = self.m
        b = 2 - self.m
        c = -nz_
        k_ = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        kf = int(k_)
        kr = round(k_)

        nzf = 2 * kf + self.m * kf * (kf - 1)
        nzr = 2 * kr + self.m * kr * (kr - 1)

        # choose better approx. and determine largest twin length
        if abs(nz_ - nzf) <= abs(nz_ - nzr):
            q = 1 + (kf - 1) * self.m
            nz = nzf
        else:
            q = 1 + (kr - 1) * self.m
            nz = nzr

        top = 0
        index = []
        while q > 0 and self._q_min_satisfied(q):
            for i in range(2):
                index.append(top + q - 1)
                top += q
            q -= self.m

        if self.q_min:
            nz = top
        return nz, index

    def _q_min_satisfied(self, q):
        if self.q_min is None:
            return True
        else:
            return q >= self.q_min
