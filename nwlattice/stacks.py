import numpy as np

import nwlattice.base as base
import nwlattice.dimensions as dims

from nwlattice.utilities import ROOT3, ROOT2
from nwlattice.planes import HexPlane, TwinPlane, SquarePlane


class FCCPristine111(base.AStackLattice):
    """A pristine FCC nanowire structure with axis along [111]"""
    sg = dims.FCCPristine111GP

    @property
    def supercell(self):
        if self._supercell is self:
            p = HexPlane.index_for_diameter(self.scale, self.D)
            self._supercell = self.get_supercell(self.scale, p)
        return self._supercell

    @classmethod
    def get_supercell(cls, a0, p):
        supercell = cls(3, p)
        supercell._scale = a0
        return supercell

    @classmethod
    def from_dimensions(cls, a0: float = 1.0,
                        diameter: float = None, length: float = None,
                        p: int = None, nz: int = None, z_periodic: bool = True):
        """
        Instantiate an FCCPristine111 nanowire lattice from actual dimensions

        :param a0: cubic cell lattice constant
        :param diameter: diameter of wire with lattice constant `a0`
        :param length: length of wire with lattice constant `a0`
        :param p: index of largest HexPlane in the stack (in lieu of `diameter`)
        :param nz: number of planes stacked (in lieu of `length`)
        :param z_periodic: enforce z-periodicity by adjusting `nz` and `length`
        :return: FCCPristine111 instance with given lattice dimensions
        """
        sg = FCCPristine111.sg(a0, diameter, length, p, nz, z_periodic)
        stk = cls(sg.nz, sg.p)
        stk._scale = sg.a0
        stk._supercell = cls.get_supercell(sg.a0, sg.p)
        return stk

    def __init__(self, nz, p):
        # construct smallest list of unique planes
        scale = 1 / ROOT2
        unit_vr = np.array([ROOT2 / 4, ROOT2 * ROOT3 / 12, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True

        # construct whole list of planes
        planes = []
        vr = np.zeros((nz, 3))
        vz_unit = 1 / ROOT3
        for i in range(nz):
            planes.append(base_planes[i % 3])
            vr[i] += (i % 3) * unit_vr
        self._v_center_com = -unit_vr
        super().__init__(planes, vz_unit, vr)


class FCCPristine100(base.AStackLattice):
    """A pristine FCC nanowire structure with axis along [100]"""
    sg = dims.FCCPristine100GP

    @property
    def supercell(self):
        if self._supercell is self:
            r = SquarePlane.index_for_diameter(self.scale, self.D)
            self._supercell = self.get_supercell(self.scale, self.nz, r)
        return self._supercell

    @classmethod
    def get_supercell(cls, a0, nz, r):
        supercell = cls(2, r)
        supercell._scale = a0
        return supercell

    @classmethod
    def from_dimensions(cls, a0=1.0, side_length=None, length=None,
                        r=None, nz=None, z_periodic=True):
        """
        Instantiate an FCCPristine100 nanowire from actual dimensions

        :param a0: cubic cell lattice constant
        :param side_length: side length of wire with lattice constant `a0`
        :param length: length of wire with lattice constant `a0`
        :param r: index of SquarePlane in stack (in lieu of `side_length`)
        :param nz: number of planes stacked (in lieu of `length`)
        :param z_periodic: enforce z-periodicity by adjusting `nz` and `length`
        :return: FCCPristine100 instance with given lattice dimensions
        """
        sg = FCCPristine100.sg(a0, side_length, length, r, nz, z_periodic)
        stk = cls(sg.nz, sg.r)
        stk._scale = sg.a0
        stk._supercell = cls.get_supercell(sg.a0, sg.nz, sg.r)
        return stk

    def __init__(self, nz, r):
        # construct smallest list of unique planes
        base_planes = [
            SquarePlane(r, even=True, scale=1.0),
            SquarePlane(r, even=False, scale=1.0)
        ]

        # construct whole list of planes
        planes = []
        vr = np.zeros((nz, 3))
        vz_unit = 0.5
        for i in range(nz):
            planes.append(base_planes[i % 2])
        super().__init__(planes, vz_unit, vr)


class FCCTwin(base.ATwinStackLattice):
    """A twinning FCC nanowire structure with smooth sidewalls"""
    sg = dims.FCCTwin111GP

    @property
    def supercell(self):
        if self._supercell is self:
            p = HexPlane.index_for_diameter(self.scale, self.D)
            self._supercell = self.get_supercell(self.scale, p, self.q_max)
        return self._supercell

    @classmethod
    def get_supercell(cls, a0, p, q_max):
        nz = base.AStackGeometry.get_cyclic_nz(0, 2 * q_max)
        index = dims.FCCTwin111GP.get_index(nz, q_max)
        supercell = cls(nz, p, index, q_max)
        supercell._scale = a0
        return supercell

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None, period=None,
                        index=None, p=None, nz=None, q_max=None,
                        z_periodic=True):
        """
        Instantiate an FCCTwin nanowire lattice from actual dimensions

        :param a0: cubic cell lattice constant
        :param diameter: diameter of wire with lattice constant `a0`
        :param length: length of wire with lattice constant `a0`
        :param period: period of twinning superlattice
        :param index: integer indices of twinned planes (in lieu of `period`)
        :param p: index of largest HexPlane in the stack (in lieu of `diameter`)
        :param nz: number of planes stacked (in lieu of `length`)
        :param q_max: number of planes in a half-period (in lieu of `period`)
        :param z_periodic: enforce z-periodicity by adjusting `nz` and `length`
        :return: FCCTwin instance with given lattice dimensions
        """
        sg = FCCTwin.sg(a0, diameter, length, period, index, p, nz, q_max,
                        z_periodic)
        stk = cls(sg.nz, sg.p, sg.index, sg.q_max)
        stk._scale = a0
        stk._P = sg.get_period()
        stk._supercell = cls.get_supercell(sg.a0, sg.p, sg.q_max)
        return stk

    def __init__(self, nz, p, index, q_max):
        index = set([int(j) for j in index])

        # construct smallest list of unique planes
        scale = 1 / ROOT2
        unit_vr = np.array([ROOT2 / 4, ROOT2 * ROOT3 / 12, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True
        # construct whole list of planes
        planes = []
        vr = np.zeros((nz, 3))
        vz_unit = 1 / ROOT3
        j = 0
        for i in range(nz):
            if i in index:
                j += 1
            planes.append(base_planes[j % 3])
            vr[i] += (j % 3) * unit_vr
            j += 1
        super().__init__(planes, vz_unit, vr, q_max, theta=np.pi / 3)


class FCCTwinFaceted(base.ATwinStackLattice):
    """A twinning FCC nanowire structure with faceted sidewalls"""
    sg = dims.FCCTwinFacetedGP

    @property
    def supercell(self):
        if self._supercell is self:
            p = HexPlane.index_for_diameter(self.scale, self.D)
            self._supercell = self.get_supercell(self.scale, p, self.q_max,
                                                 self.q0)
        return self._supercell

    @classmethod
    def get_supercell(cls, a0, p, q_max, q0=0):
        nz = base.AStackGeometry.get_cyclic_nz(0, 2 * q_max)
        supercell = cls(nz, p, q0, q_max)
        supercell._scale = a0
        return supercell

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None, period=None,
                        p=None, nz=None, q_max=None, q0=0, q_max_auto=True,
                        z_periodic=True):
        """
        Instantiate an FCCTwinFaceted nanowire lattice from actual dimensions

        :param a0: cubic cell lattice constant
        :param diameter: diameter of wire with lattice constant `a0`
        :param length: length of wire with lattice constant `a0`
        :param period: period of twinning superlattice
        :param p: index of largest HexPlane in the stack (in lieu of `diameter`)
        :param nz: number of planes stacked (in lieu of `length`)
        :param q_max: number of planes in a half-period (in lieu of `period`)
        :param q0: second index of first TwinPlane in the stack
        :param q_max_auto: auto-adjust `q_max` if resulting period too large
        :param z_periodic: enforce z-periodicity by adjusting `nz` and `length`
        :return: FCCTwinFaceted instance with given lattice dimensions
        """
        sg = FCCTwinFaceted.sg(a0, diameter, length, period, p, nz, q_max, q0,
                               z_periodic)
        stk = cls(sg.nz, sg.p, sg.q0, sg.q_max)
        stk._scale = sg.a0
        stk._P = sg.get_period()
        stk._supercell = cls.get_supercell(sg.a0, sg.p, sg.q_max, sg.q0)
        return stk

    def __init__(self, nz, p, q0, q_max):
        # obtain cycle of `q` indices for comprising TwinPlanes
        q_cycle = FCCTwinFaceted.get_q_cycle(nz, q0, q_max)

        # construct whole list of planes
        scale = 1 / ROOT2
        planes = [TwinPlane(p, q, scale=scale) for q in q_cycle]
        vr = np.zeros((nz, 3))
        vz_unit = 1 / ROOT3
        super().__init__(planes, vz_unit, vr, q_max, theta=np.pi / 3)

    @property
    def q0(self):
        return self.planes[0].q

    @staticmethod
    def get_q_cycle(nz, q0, q_max):
        """
        List of second indices for TwinPlanes in an FCCTwinFaceted instance

        :param nz: number of planes stacked
        :param q0: second index of first TwinPlane in the stack
        :param q_max: number of planes in a half-period
        :return: list of int `q` values
        """
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


class HexPristine0001(base.AStackLattice):
    """A pristine hexagonal nanowire structure oriented along [0001]"""
    sg = dims.HexPristine0001GP

    @property
    def supercell(self):
        if self._supercell is self:
            p = HexPlane.index_for_diameter(self.scale, self.D)
            self._supercell = self.get_supercell(self.scale, p)
        return self._supercell

    @classmethod
    def get_supercell(cls, a0, p):
        supercell = cls(2, p)
        supercell._scale = a0
        return supercell

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None,
                        p=None, nz=None, z_periodic=True):
        """
        Instantiate a HexPristine0001 nanowire lattice from actual dimensions

        :param a0: cubic cell lattice constant
        :param diameter: diameter of wire with lattice constant `a0`
        :param length: length of wire with lattice constant `a0`
        :param p: index of largest HexPlane in the stack (in lieu of `diameter`)
        :param nz: number of planes stacked (in lieu of `length`)
        :param z_periodic: enforce z-periodicity by adjusting `nz` and `length`
        :return: HexPristine0001 instance with given lattice dimensions
        """
        sg = HexPristine0001.sg(a0, diameter, length, p, nz, z_periodic)
        stk = cls(sg.nz, sg.p)
        stk._scale = sg.a0
        stk._supercell = cls.get_supercell(sg.a0, sg.p)
        return stk

    def __init__(self, nz, p):
        # obtain cycle of `q` indices for comprising TwinPlanes
        q_cycle = FCCTwinFaceted.get_q_cycle(nz, 0, 1)

        # construct whole list of planes
        scale = 1 / ROOT2
        planes = [TwinPlane(p, q, scale=scale) for q in q_cycle]
        vr = np.zeros((nz, 3))
        vz_unit = 1 / ROOT3
        super().__init__(planes, vz_unit, vr)


class FCCHexMixed(base.AStackLattice):
    """A mixed phase nanowire structure with FCC and hexagonal segments"""
    sg = dims.FCCPristine111GP

    @property
    def supercell(self):
        return self

    @classmethod
    def get_supercell(cls, nz, p, index):
        raise cls(nz, p, index)

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None, index=None,
                        frac=None, p=None, nz=None):
        """
        Instantiate a HexPristine0001 nanowire lattice from actual dimensions

        :param a0: cubic cell lattice constant
        :param diameter: diameter of wire with lattice constant `a0`
        :param length: length of wire with lattice constant `a0`
        :param index: integer indices of Hexagonal planes
        :param frac: randomly insert approx. this fraction of Hexagonal planes
        :param p: index of largest HexPlane in the stack (in lieu of `diameter`)
        :param nz: number of planes stacked (in lieu of `length`)
        :return: FCCHexMixed instance with given lattice dimensions
        """
        sg = FCCHexMixed.sg(a0, diameter, length, p, nz, z_periodic=False)
        if index:
            pass
        elif frac:
            index = []
            for i in range(sg.nz):
                if np.random.uniform(0, 1) < frac:
                    index.append(i)
        else:
            index = []
        stk = cls(sg.nz, sg.p, index)
        stk._scale = a0
        return stk

    def __init__(self, nz, p, index):
        index = set([int(j) for j in index])

        # construct smallest list of unique planes
        scale = 1 / ROOT2
        unit_vr = np.array([ROOT2 / 4, ROOT2 * ROOT3 / 12, 0.])
        base_planes = [
            HexPlane(p - 1, even=False, scale=scale),
            HexPlane(p, even=True, scale=scale),
            HexPlane(p - 1, even=False, scale=scale)
        ]
        base_planes[2].inverted = True

        # construct whole list of planes
        planes = []
        vr = np.zeros((nz, 3))
        vz_unit = 1 / ROOT3
        j = 0
        for i in range(nz):
            if i in index:
                j += -2 * (i % 2)
            planes.append(base_planes[j % 3])
            vr[i] += (j % 3) * unit_vr
            j += 1
        super().__init__(planes, vz_unit, vr)
        self._fraction = len(index) / nz

    @property
    def fraction(self):
        return self._fraction


class FCCTwinPadded(base.APaddedStackLattice):
    sg = dims.FCCTwinPaddedStackGeometry

    @classmethod
    def from_dimensions(cls, a0=1.0, diameter=None, length=None, period=None):
        sg = FCCTwinPadded.sg(a0, diameter, length, period)

        stk_core = FCCTwin.from_dimensions(a0, p=sg.p, q_max=sg.q_max, nz=sg.nz)
        stk_top = FCCPristine111.from_dimensions(a0, p=sg.p, nz=sg.nz)
        stk_bottom = stk_top.inverted()

        stk = cls(stk_bottom, stk_core, stk_top,
                  sg.nz_bottom, sg.nz_core, sg.nz_top,
                  vz_unit=1 / ROOT3)
        stk._scale = a0
        return stk


class ZBTwinPadded(FCCTwinPadded):
    def __init__(self, stk_bottom, stk_core, stk_top,
                 nz_bottom, nz_core, nz_top, vz_unit):
        super().__init__(stk_bottom, stk_core, stk_top,
                         nz_bottom, nz_core, nz_top, vz_unit)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBPristine111(FCCPristine111):
    """A pristine zincblende nanowire structure with axis along [111]"""

    def __init__(self, nz, p):
        super().__init__(nz, p)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBPristine100(FCCPristine100):
    """A pristine zincblende nanowire structure with axis along [100]"""

    def __init__(self, nz, r):
        super().__init__(nz, r)
        self.add_basis(2, np.array([0., -0.40824829, -0.14433757]))


class ZBTwin(FCCTwin):
    """A twinning zincblende nanowire structure with smooth sidewalls"""

    def __init__(self, nz, p, index, q_max=None):
        super().__init__(nz, p, index, q_max)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBTwinFaceted(FCCTwinFaceted):
    """A twinning zincblende nanowire structure with faceted sidewalls"""

    def __init__(self, nz, p, q0, q_max=None):
        super().__init__(nz, p, q0, q_max)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class WZPristine0001(HexPristine0001):
    """A pristine wurtzite nanowire structure oriented along [0001]"""

    def __init__(self, nz, p):
        super().__init__(nz, p)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBWZMixed(FCCHexMixed):
    """A mixed phase nanowire structure with ZB and WZ segments"""

    def __init__(self, nz, p, index):
        super().__init__(nz, p, index)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))
