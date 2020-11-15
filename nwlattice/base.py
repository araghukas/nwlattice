from abc import ABC, abstractmethod
import numpy as np
from time import time
from os.path import expanduser

from nwlattice.utilities import ROOT3


class AStackLattice(ABC):
    """abstract base class for wire lattices made of stacked planes"""

    def __init__(self, planes, dz, dxy):
        super().__init__()
        self._planes = []  # list of PointPlane objects to be stacked
        self._N = None  # number of lattice points
        self._nz = None  # number of planes in wire lattice
        self._dz = None  # spacing between planes in a0=1 units
        self._dxy = None  # xy-plane offset between planes in a0=1 units
        self._D = None  # actual diameter (scaled)
        self._L = None  # actual length (scaled)
        self._P = None  # actual period (scaled)
        self._basis = {}  # atom types attached to lattice
        self._area = None
        self._v_center_com = np.zeros(3)
        self._scale = 1.0

        for plane in planes:
            if isinstance(plane, APointPlane):
                self._planes.append(plane)
            else:
                raise TypeError("all items in planes list must be PointPlanes")

        self._dz = np.reshape(dz, (self.nz, 3))
        self._dxy = np.reshape(dxy, (self.nz, 3))

    @classmethod
    @abstractmethod
    def from_dimensions(cls, *args):
        """create stack lattice from measurements instead of indices"""
        raise NotImplementedError

    @abstractmethod
    def write_map(self, file_path):
        """write phana compatible map file for smallest unit cell"""
        raise NotImplementedError

    @property
    @abstractmethod
    def type_name(self):
        """string identifying each sub-class"""
        raise NotImplementedError

    # --------------------------------------------------------------------------
    # concrete properties
    # --------------------------------------------------------------------------

    @property
    def scale(self):
        return self._scale

    @property
    def D(self):
        """returns largest plane diameter"""
        if self._D is None:
            D = self.planes[0].D
            for plane in self.planes[1:]:
                if plane.D > D:
                    D = plane.D
            self._D = D * self._scale  # NOTE: scaled by planes scales
        return self._D

    @property
    def L(self):
        """real length; returns scaled sum of z displacements"""
        if self._L is None:
            L = 0.
            ddz = np.diff(self._dz, axis=0)
            ddz = np.linalg.norm(ddz, axis=1)
            for d in ddz:
                L += d * self._scale
            self._L = L
        return self._L

    @property
    def P(self):
        """real twinning period"""
        return self._P

    @property
    def area(self):
        """returns average plane area"""
        if self._area is None:
            sum_area = 0
            n = 0
            for plane in self.planes:
                sum_area += plane.area
                n += 1
            self._area = sum_area / n * self._scale**2
        return self._area

    @property
    def planes(self):
        return self._planes

    @property
    def N(self):
        if self._N is None:
            N = 0
            for plane in self.planes:
                N += plane.N
            self._N = N
        return self._N

    @property
    def nz(self):
        if self._nz is None:
            self._nz = len(self.planes)
        return self._nz

    @property
    def basis(self):
        return self._basis

    @property
    def dz(self):
        return self._dz

    @property
    def dxy(self):
        return self._dxy

    # --------------------------------------------------------------------------
    # concrete methods
    # --------------------------------------------------------------------------

    def add_basis(self, t, pt):
        """add an atom of type `t` at point `pt`"""
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

    def get_points(self, t):
        """return an array of all atom points of type `t`"""
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

    def write_points(self, file_path):
        """write LAMMPS/OVITO compatible data file of all atom points"""
        # create dict of atom types and arrays of corresponding points
        N_atoms = 0  # total number of atoms
        points_dict = {}
        for b in self.basis:
            points_dict[b] = self.get_points(b)
            N_atoms += self.N * len(self.basis[b])

        t1 = time()
        path_ = expanduser(file_path)
        with open(path_, "w") as file_:
            # header (ignored)
            file_.write("atom coordinates generated by 'nwlattice' package\n")
            file_.write("\n")

            # number of atoms and number of atom types
            file_.write("%d atoms\n" % N_atoms)
            file_.write("%d atom types\n" % len(self.basis))
            file_.write("\n")

            # simulation box
            x = 2 * self.D  # keep atoms' (x,y) in box
            y = x
            z = self.L
            basis_z_min = basis_z_max = 0.
            for b in self._basis:
                for bpt in self._basis[b]:
                    if bpt[2] < basis_z_min:
                        basis_z_min = bpt[2] * self._scale
                    elif bpt[2] > basis_z_max:
                        basis_z_max = bpt[2] * self._scale

            file_.write("{} {} xlo xhi\n".format(-x / 2., x / 2.))
            file_.write("{} {} ylo yhi\n".format(-y / 2., y / 2.))
            file_.write("{} {} zlo zhi\n"
                        .format(basis_z_min, z + basis_z_max))
            file_.write("\n")

            # Atoms section
            file_.write("Atoms # atomic\n")
            file_.write("\n")
            id_ = 1
            for typ, points in points_dict.items():
                for pt in points:
                    pt *= self._scale
                    file_.write("{} {} {} {} {} 0 0 0\n"
                                .format(id_, typ, pt[0], pt[1], pt[2]))
                    id_ += 1
            t2 = time()
            print("wrote %d atoms to data file '%s' in %f seconds"
                  % (N_atoms, path_, t2 - t1))

    @staticmethod
    def get_cyclic_nz(*args):
        nz = args[0]
        k = 3
        nlo = (nz // k) * k
        nhi = ((nz + k) // k) * k

        if nlo == 0:
            return nhi
        elif (nz - nlo) < (nhi - nz):
            return nlo
        else:
            return nhi


class APointPlane(ABC):
    hex_vectors = np.array([[1., 0., 0.], [-.5, ROOT3 / 2., 0.]])
    sq_vectors = np.array([[1., 0., 0.], [0., 1., 0.]])

    ohex_delta = .5 * np.array([1., 1. / ROOT3, 0.])
    ehex_delta = .25 * np.array([1, -1. / ROOT3, 0.])

    def __init__(self, scale):
        super().__init__()
        self._N = None
        self._D = None
        self._nz = None
        self._vectors = None
        self._area = None
        self._com = None
        self._scale = scale
        self._points = None

    @abstractmethod
    def get_points(self, center=True):
        """return an array of all atom points"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_index_for_diameter(scale, D):
        """return plane index for given diameter"""
        raise NotImplementedError

    @property
    @abstractmethod
    def N(self):
        return self._N

    @property
    @abstractmethod
    def D(self):
        # implement diameter formula for each subclass
        raise NotImplementedError

    @property
    @abstractmethod
    def area(self):
        # implement area formula for each subclass
        raise NotImplementedError

    @property
    @abstractmethod
    def vectors(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def com(self):
        # implement com formula for each subclass
        raise NotImplementedError

    @property
    def scale(self):
        # global scaling of all point in plane
        return self._scale
