from nwlattice import base
from nwlattice import sizes
from nwlattice import indices
from nwlattice.utilities import ROOT2, ROOT3
from nwlattice.planes import FCCb, TwFCC

import numpy as np


class FCCTwinFacetedA(base.ANanowireLatticeArbitrary):
    """
    Faceted twinning face-centered cubic nanowire with axis along [111].
    Each twin sister is a section of a non-primitive octahedral unit cell.
    Cross-section is constant-perimeter hexagonal.
    """
    DEFAULT_INDEXER = indices.LinearDecrease(1)

    def __init__(self, scale: float, width: float = None, length: float = None,
                 n_xy: int = None, nz: int = None, m0: int = 0,
                 indexer: callable = None):
        """
        :param scale: side length of cubic unit cell
        :param width: approximated width
        :param length: approximated length
        :param n_xy: (overrides `width`) number of atoms in radial direction
        :param nz: (overrides `length`) number of base planes stacked vertically
        :param indexer: a function int -> (int, list) that returns nz and twin point indices
        """
        if indexer is None:
            indexer = FCCTwinFacetedA.DEFAULT_INDEXER

        # TODO: maybe tell indexer about q in this... wanna avoid SPECIFIC indexer for each wire
        size = self.get_size(indexer, scale, width, length, n_xy, nz)
        plane_index = size.index

        q_max = size.n_xy - 1
        m_cycle = [m0]
        step = 1
        i = 0
        while i < size.nz - 1:
            next_q = m_cycle[-1] + step
            m_cycle.append(next_q)

            # twin if structure forces it or if `plane_index` says so
            if next_q == q_max or next_q == 0 or (i in plane_index):
                step *= -1
            i += 1

        planes = [TwFCC(1 / ROOT2, size.n_xy, m) for m in m_cycle]
        vr = np.zeros((size.nz, 3))
        super().__init__(size, planes, vr)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        return FCCb.get_n_xy(scale, width * ROOT2)

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        return FCCb.get_width(scale, n_xy) / ROOT2

    def get_size(self, indexer: callable, scale, width=None, length=None, n_xy=None, nz=None):
        size = sizes.NanowireSizeArbitrary(scale, 1 / ROOT3, n_xy, nz, width, length)
        size._indexer = indexer
        size._n_xy_func = self.get_n_xy
        size._nz_func = self.get_nz
        size._width_func = self.get_width
        size._length_func = self.get_length
        size._area_func = self.get_area
        return size
