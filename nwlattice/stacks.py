import numpy as np

from nwlattice.utilities import ROOT2, ROOT3
from nwlattice.base import AStackLattice, ATwinStackLattice
from nwlattice.planes import HexPlane, TwinPlane, SquarePlane


class FCCPristine111(AStackLattice):
    """A pristine FCC nanowire structure with axis along [111]"""

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
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        if z_periodic:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p
        stk = cls(nz, p)
        stk._scale = a0
        stk._supercell = cls.get_supercell(a0, p)
        return stk

    def write_map(self, file_path):
        raise NotImplementedError

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


class FCCPristine100(AStackLattice):
    """A pristine FCC nanowire structure with axis along [100]"""

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
        if side_length is None and r is None:
            raise ValueError("must specify either `diameter` or `r`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        nz = 1 + round(2. * length / a0) if nz is None else nz
        if z_periodic:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))
        r = (SquarePlane.get_index_for_diameter(a0, side_length)
             if r is None else r)
        stk = cls(nz, r)
        stk._scale = a0
        stk._supercell = cls.get_supercell(a0, nz, r)
        return stk

    def write_map(self, file_path):
        raise NotImplementedError

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


class FCCTwin(ATwinStackLattice):
    """A twinning FCC nanowire structure with smooth sidewalls"""

    @classmethod
    def get_supercell(cls, a0, p, q_max):
        nz = cls.get_cyclic_nz(0, q_max)
        index = cls.get_index(nz, q_max)
        supercell = cls(nz, p, index)
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
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")
        if period is None and q_max is None and index is None:
            raise ValueError(
                "must specify either `period`, `q_max`, or `index`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p
        if q_max or period:
            q_max = round(ROOT3 * period / 2. / a0) if q_max is None else q_max

        if z_periodic and q_max:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz, q_max)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))

        if index is not None:
            pass
        elif q_max or period:
            index = cls.get_index(nz, q_max)
        else:
            index = []

        stk = cls(nz, p, index, q_max)
        stk._scale = a0
        if q_max:
            stk._P = 2 * a0 * q_max / ROOT3
        stk._supercell = cls.get_supercell(a0, p, q_max)
        return stk

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, p, index, q_max: int = None):
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

    @staticmethod
    def get_cyclic_nz(*args):
        """
        Overridden base method: returns `nz` for for integer number of twins

        :param args: first argument should be `nz`, and the second `q_max`
        :return: an int `nhi` or `nlo`, the nearest `2 * q_max` multiple of `nz`
        """
        nz, q_max = args
        k = 2 * q_max
        nlo = (nz // k) * k
        nhi = ((nz + k) // k) * k

        if nlo == 0:
            return nhi
        elif (nz - nlo) < (nhi - nz):
            return nlo
        else:
            return nhi

    @staticmethod
    def get_index(nz, q_max):
        index = []
        i_period = q_max
        include = True
        for i in range(nz):
            if i % i_period == 0:
                include = not include
            if include:
                index.append(i)
        return index


class FCCTwinFaceted(ATwinStackLattice):
    """A twinning FCC nanowire structure with faceted sidewalls"""

    @classmethod
    def get_supercell(cls, a0, p, q_max, q0=0):
        nz = cls.get_cyclic_nz(0, q_max)
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

        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")
        if period is None and q_max is None:
            raise ValueError("must specify either `period` or `q_max`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p
        q_max = round(ROOT3 * period / 2 / a0) if q_max is None else q_max

        if q_max >= p:
            q_max = p - 1
            if q_max_auto:
                old_period = period
                period = 2. * a0 * q_max / ROOT3
                print("Period is too large, corrected: %f --> %f "
                      % (old_period, period))
            else:
                raise ValueError("period {:f} is too large for given "
                                 "diameter {:f}\n Maximum period is {:f} "
                                 "(or `q_max = {:d}`)"
                                 .format(period, diameter,
                                         2. * a0 * q_max / ROOT3, q_max))

        if z_periodic:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz, q_max)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))

        stk = cls(nz, p, q0, q_max)
        stk._scale = a0
        stk._P = 2 * a0 * q_max / ROOT3
        stk._supercell = cls.get_supercell(a0, p, q_max, q0)
        return stk

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, p, q0, q_max):
        # obtain cycle of `q` indices for comprising TwinPlanes
        q_cycle = self.get_q_cycle(nz, q0, q_max)

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
    def get_cyclic_nz(*args):
        """
        Overridden base method: returns `nz` for for integer number of twins

        :param args: first argument should be `nz`, and the second `q_max`
        :return: an int `nhi` or `nlo`, the nearest `2 * q_max` multiple of `nz`
        """
        nz, q_max = args
        k = 2 * q_max
        nlo = (nz // k) * k
        nhi = ((nz + k) // k) * k

        if nlo == 0:
            return nhi
        elif (nz - nlo) < (nhi - nz):
            return nlo
        else:
            return nhi

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


class HexPristine0001(AStackLattice):
    """A pristine hexagonal nanowire structure oriented along [0001]"""
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

        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p

        if z_periodic:
            old_nz = nz
            nz = cls.get_cyclic_nz(nz)
            print("forced z periodicity, adjusted nz: %d --> %d"
                  % (old_nz, nz))

        stk = cls(nz, p)
        stk._scale = a0
        stk._supercell = cls.get_supercell(a0, p)
        return stk

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, p):
        # obtain cycle of `q` indices for comprising TwinPlanes
        q_cycle = FCCTwinFaceted.get_q_cycle(nz, 0, 1)

        # construct whole list of planes
        scale = 1 / ROOT2
        planes = [TwinPlane(p, q, scale=scale) for q in q_cycle]
        vr = np.zeros((nz, 3))
        vz_unit = 1 / ROOT3
        super().__init__(planes, vz_unit, vr)


class FCCHexMixed(AStackLattice):
    """A mixed phase nanowire structure with FCC and hexagonal segments"""
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
        if diameter is None and p is None:
            raise ValueError("must specify either `diameter` or `p`")
        if length is None and nz is None:
            raise ValueError("must specify either `length` or `nz`")

        nz = round(ROOT3 * length / a0) if nz is None else nz
        p = HexPlane.get_index_for_diameter(a0, diameter) if p is None else p

        if index:
            pass
        elif frac:
            index = []
            for i in range(nz):
                if np.random.uniform(0, 1) < frac:
                    index.append(i)
        else:
            index = []
        stk = cls(nz, p, index)
        stk._scale = a0
        return stk

    def write_map(self, file_path):
        raise NotImplementedError

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


class ZBPristine111(FCCPristine111):
    """A pristine zincblende nanowire structure with axis along [111]"""

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, p):
        super().__init__(nz, p)
        super().__init__(nz, p)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBPristine100(FCCPristine100):
    """A pristine zincblende nanowire structure with axis along [100]"""

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, r):
        super().__init__(nz, r)
        self.add_basis(2, np.array([0., -0.40824829, -0.14433757]))


class ZBTwin(FCCTwin):
    """A twinning zincblende nanowire structure with smooth sidewalls"""

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, p, index, q_max=None):
        super().__init__(nz, p, index, q_max)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBTwinFaceted(FCCTwinFaceted):
    """A twinning zincblende nanowire structure with faceted sidewalls"""

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, p, q0, q_max=None):
        super().__init__(nz, p, q0, q_max)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class WZPristine0001(HexPristine0001):
    """A pristine wurtzite nanowire structure oriented along [0001]"""

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, p):
        super().__init__(nz, p)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))


class ZBWZMixed(FCCHexMixed):
    """A mixed phase nanowire structure with ZB and WZ segments"""

    def write_map(self, file_path):
        raise NotImplementedError

    def __init__(self, nz, p, index):
        super().__init__(nz, p, index)
        self.add_basis(2, np.array([0., 0., ROOT3 / 4]))
