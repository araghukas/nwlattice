from abc import ABC, abstractmethod
import numpy as np


class IPointLattice(ABC):
    """
    The interface for implementing logic that generates point information
    """
    @abstractmethod
    def get_points(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_types(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_map_rows(self) -> list:
        raise NotImplementedError

    @property
    @abstractmethod
    def N(self):
        raise NotImplementedError

    def __init__(self):
        self._supercell = self
        self._N = None
        self._basis = {1: [np.zeros(3)]}
        self._scale = 1.0

        self._v_center_com = np.zeros(3)

    @property
    def scale(self):
        return self._scale

    @property
    def basis(self):
        return self._basis

    @property
    def type_name(self):
        """string identifying each sub-class"""
        return str(self.__class__).split('.')[-1].strip("'>")

    def add_basis(self, t: int, pt):
        """
        Add a basis point of type `t` at 3-point `pt`

        :param t: integer atom type ID
        :param pt: 3-point indicating basis point relative to lattice point
        :return: None
        """
        t = int(t)
        if t <= 0:
            raise ValueError("only positive integers should be used for "
                             "atom type identifiers")

        pt = np.asarray(pt)
        if len(pt) != 3:
            raise ValueError("basis point must be 3-dimensional")

        if t in self._basis:
            self._basis[t].append(pt)
        else:
            self._basis[t] = [pt]
