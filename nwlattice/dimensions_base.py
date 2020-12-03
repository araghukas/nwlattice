from abc import abstractmethod

from nwlattice.printer import ABCPrinter


class AStackGeometry(ABCPrinter):
    """Handles conversions from real sizes to lattice indices"""

    @abstractmethod
    def z_index(self, length: float) -> int:
        raise NotImplementedError

    @abstractmethod
    def xy_index(self, xy_length: float) -> int:
        raise NotImplementedError

    @abstractmethod
    def validate_args(self, *args):
        raise NotImplementedError

    def __init__(self, a0: float):
        self.a0 = a0

    @staticmethod
    def get_cyclic_nz(z_index, k, nearest=True):
        nlo = (z_index // k) * k
        nhi = ((z_index + k) // k) * k

        if nearest:
            if nlo == 0:
                return nhi
            elif (z_index - nlo) < (nhi - z_index):
                return nlo
            else:
                return nhi
        else:
            return nlo, nhi


class APaddedStackGeometry(AStackGeometry):
    """
    stack geometry for padded lattice stacks
    """

    def z_index(self, length: float) -> int:
        return self.nz_bottom, self.nz_core, self.nz_top

    def xy_index(self, xy_length: float) -> int:
        return self.c_sg.xy_index(xy_length), self.p_sg.xy_index(xy_length)

    def validate_args(self, *args):
        # not necessary by construction
        pass

    @property
    @abstractmethod
    def nz_bottom(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def nz_core(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def nz_top(self):
        raise NotImplementedError

    def __init__(self, c_sg, p_sg):
        super().__init__(c_sg.a0)
        self.c_sg = c_sg
        self.p_sg = p_sg

        self._nz_bottom = None
        self._nz_core = None
        self._nz_top = None

    @property
    def nz(self):
        return self.nz_bottom + self.nz_core + self.nz_top
