from abc import ABC, abstractmethod
from os.path import expanduser
from time import time
import warnings
import numpy as np
from typing import List

from nwlattice.utilities import Quaternion as Qtr
from nwlattice.sizes import NanowireSizeCompound, NanowireSizeArbitrary


# TODO: reduce data file size and write times; compressed output?
class IDataWriter(ABC):
    """
    The interface for writing the LAMMPS atom data
    """

    # globally toggles runtime printing of feedback
    WILL_PRINT = False

    # globally toggles runtime warning
    WILL_WARN = False

    @property
    def type_name(self) -> str:
        """return class name as a string"""
        return str(self.__class__).split('.')[-1].strip("'>")

    @abstractmethod
    def get_points(self) -> np.ndarray:
        """returns an N x 3 array containing all atoms points"""
        raise NotImplementedError

    @abstractmethod
    def get_types(self) -> np.ndarray:
        """returns an N x 1 array containing all atom types"""
        raise NotImplementedError

    @abstractmethod
    def write_points(self, *args):
        """writes the LAMMPS data file readable via the `read_data` command"""
        raise NotImplementedError

    @staticmethod
    def print(s: str):
        """prints the string `s` if `WILL_PRINT` is True"""
        if IDataWriter.WILL_PRINT:
            print(s)

    @staticmethod
    def warn(s: str):
        """warns with message `s` if `WILL_WARN` is True"""
        if IDataWriter.WILL_WARN:
            warnings.warn(s)


class APointPlane(IDataWriter):
    """
    Base class for single-basis-atom planar lattices.
    """

    def __init__(self, size_obj, vectors, theta=None):
        super().__init__()
        self._vectors = vectors  # translation vectors
        self._size_obj = size_obj  # dimensions/geometry handler
        self._theta = theta  # in-plane rotation angle

        self._N = None  # number of lattice points
        self._com = None  # points centre of mass
        self._points = None  # array of points in this PlaneLattice
        self._v_offset = np.zeros(3)  # translation vector

    @staticmethod
    @abstractmethod
    def get_n_xy(scale: float, width: float) -> int:
        """returns nearest integer lattice width from continuous width"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_width(scale: float, n_xy: int) -> float:
        """returns continuous width from integer lattice width"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_area(*args) -> float:
        """returns area of plane"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_vectors() -> np.ndarray:
        """returns the translation vectors for the given plane class"""
        raise NotImplementedError

    @abstractmethod
    def fits(self, other) -> bool:
        raise NotImplementedError

    @property
    def N(self):
        # subclass-specific method
        raise NotImplementedError

    @property
    def com(self):
        # subclass-specific method
        raise NotImplementedError

    @property
    def size(self):
        return self._size_obj

    @property
    def vectors(self):
        return self._vectors

    @property
    def points(self):
        if self.theta is None:
            points = self.get_points()
        else:
            axis = [0, 0, 1]
            points = Qtr.qrotate(self.get_points(), axis, self.theta)
        return points * self.size.scale

    @property
    def theta(self):
        return self._theta

    def write_points(self, file_path: str = None, first_quad: bool = False):
        """
        Write LAMMPS/OVITO compatible data file of all atom points
        :param file_path: string indicating target file (created/overwritten)
        :param first_quad: translate points into first quadrant
        """
        if file_path is None:
            file_path = "{}_structure.data".format(self.type_name)

        N_atoms = self.N  # total number of atoms in plane

        t1 = time()
        file_path = expanduser(file_path)
        with open(file_path, "w") as file_:
            # header (ignored)
            file_.write("atom coordinates generated by 'nwlattice' package\n")
            file_.write("\n")

            # number of atoms and number of atom types
            file_.write("%d atoms\n" % N_atoms)
            file_.write("1 atom types\n")
            file_.write("\n")

            # simulation box
            xlo, xhi, ylo, yhi = self._get_points_box_dims()
            dx = xlo if first_quad else 0.0
            dy = ylo if first_quad else 0.0

            # write simulation box
            file_.write("{:.6f} {:.6f} xlo xhi\n".format(xlo - dx, xhi - dx))
            file_.write("{:.6f} {:.6f} ylo yhi\n".format(ylo - dy, yhi - dy))
            file_.write("\n")

            # Atoms section
            file_.write("Atoms # atomic\n")
            file_.write("\n")
            id_ = 1

            for pt in self.points:
                file_.write("{:d} {:d} {:.6f} {:.6f} {:.6f} 0 0 0\n"
                            .format(id_, 1, pt[0] - dx, pt[1] - dy, pt[2]))
                id_ += 1
            t2 = time()
            self.print("wrote %d atoms in %f seconds" % (N_atoms, t2 - t1))

    def get_points(self) -> np.ndarray:
        # subclass-specific method
        raise NotImplementedError

    def get_types(self) -> np.ndarray:
        """
        Return the integer type identifier for atoms identifier `ID`
        :return: array of atom types
        """
        return np.ones(self.N)

    def _get_points_box_dims(self):
        """return simulation box dimensions for write_points() method"""
        x = y = 2 * self.size.width  # keep atoms' (x,y) in box
        return -x / 2, x / 2, -y / 2, y / 2


class PlaneZStack(IDataWriter):
    """A basic stack of planes along the z-direction"""

    _scale: float  # scale factor applied for point planes
    _planes: List[APointPlane]  # planes to be stacked
    _basis: dict  # dictionary of the form {atom_type: [locations]}
    _vz: np.ndarray  # positions of the planes along z-axis
    _vr: np.ndarray  # offsets of planes in the xy-plane
    _points: np.ndarray

    def __init__(self, scale, planes, vz=None, vr=None, basis=None):
        self._scale = scale
        if vz is not None and vr is not None:
            if not (len(planes) == len(vz) == len(vr)):
                raise ValueError("planes, vz, and vr arrays don't have the same length")
        self._planes = planes
        self._vz = np.zeros((len(planes), 3)) if vz is None else vz
        self._vr = np.zeros((len(planes), 3)) if vr is None else vr
        self._basis = {1: [np.zeros(3)]} if basis is None else basis

    @property
    def scale(self):
        return self._scale

    @property
    def basis(self):
        return self._basis

    @property
    def planes(self):
        return self._planes.copy()

    @property
    def points(self):
        if self._points is None:
            self._points = self.get_points()
        return self._points.copy()

    @points.setter
    def points(self, _points: np.ndarray):
        if not _points.shape == self._points.shape:
            raise ValueError("shape of new points array does not match the original")
        self._points = _points

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

    def get_points(self) -> np.ndarray:
        """produce and return an array of points based on planes, basis, and offsets"""
        n_basis_atoms = sum(len(self.basis[t]) for t in self.basis)
        N_total = sum(plane.N for plane in self._planes)
        atom_pts = np.zeros((N_total * n_basis_atoms, 3), dtype=float)

        n = 0
        nz = len(self._planes)
        for t in self._basis:
            for bpt in self._basis[t]:
                for i in range(nz):
                    plane = self._planes[i]
                    atom_pts[n:(n + plane.N)] = (
                            plane.points + bpt + self._vr[i] + self._vz[i]
                    )
                    n += plane.N

        return self._scale * atom_pts

    def get_types(self) -> np.ndarray:
        n_basis_atoms = sum(len(self.basis[t]) for t in self._basis)
        N_total = sum(plane.N for plane in self._planes)
        types = np.zeros(N_total * n_basis_atoms, dtype=int)

        ID = 1
        nz = len(self._planes)
        for t in self._basis:
            for i in range(nz):
                plane = self._planes[i]
                types[(ID - 1):(ID - 1) + plane.N] = t
                ID += plane.N

        return types

    def get_layer_map(self, planes_per_layer: int = 1) -> List[list]:
        """

        :param planes_per_layer: (int) number of planes (including all basis atoms) in a layer
        :return: a list of lists of the form [layer number][atom id in layer]
        """
        N_planes = len(self.planes)
        if len(self.planes) % planes_per_layer != 0:
            raise ValueError("number of planes in structure ({:d}) is not divisible by {:d}"
                             .format(N_planes, planes_per_layer))
        n_layers = N_planes // planes_per_layer
        n_basis_atoms = sum(len(self._basis[t]) for t in self._basis)
        N_total = sum(plane.N for plane in self._planes)
        nz = len(self._planes)

        layer_map = [[] for _ in range(n_layers)]
        atom_ids = np.arange(N_total * n_basis_atoms) + 1
        id_counter = 0

        for t in self.basis:
            for _ in self.basis[t]:
                for i in range(nz):
                    layer_index = i // planes_per_layer
                    for j in range(self.planes[i].N):
                        layer_map[layer_index].append(atom_ids[id_counter])
                        id_counter += 1

        return layer_map

    def write_points(self,
                     file_path: str,
                     simbox_matrix: np.ndarray = None,
                     center_points: bool = True,
                     wrap_points: bool = False,
                     origin=None):
        """
        Write LAMMPS/OVITO compatible data file of all atom points

        :param file_path: string indicating target file (created/overwritten)
        :param simbox_matrix: 2D matrix [[xx, xy, xz], [_, yy, yz], [_, _, zz]] of simulation box
        :param center_points: translate points to the centre of the cell
        :param wrap_points: wrap points at periodic boundaries
        :param origin: origin of the simulation cell; default [0, 0, 0]
        :return: None
        """

        if simbox_matrix is None:
            xx = yy = 2. * self._scale * max(plane.size.width for plane in self._planes)
            xy = xz = yz = 0.
            dz = self._vz[1][2] - self._vz[0][2]
            zz = self._scale * dz * len(self._planes)
        else:
            xx = simbox_matrix[0][0]
            xy = simbox_matrix[0][1]
            xz = simbox_matrix[0][2]
            yy = simbox_matrix[1][1]
            yz = simbox_matrix[1][2]
            zz = simbox_matrix[2][2]

        if origin is None:
            origin = np.array([0., 0., 0.])

        t1 = time()

        file_path = expanduser(file_path)
        atom_types = self.get_types()
        atom_points = self.get_points()

        N_total = sum(plane.N for plane in self._planes)
        n_basis_atoms = sum(len(self._basis[t]) for t in self._basis)
        atom_ids = np.arange(N_total * n_basis_atoms) + 1

        if center_points:
            com = np.sum(atom_points, axis=0) / len(atom_points)
            atom_points -= com
            atom_points += .5 * np.array([xx + xy, yy, zz])

        if wrap_points:
            M = np.array([[xx, xy, xz],
                          [0., yy, yz],
                          [0., 0., zz]])
            M_inv = np.linalg.inv(M)
            x = np.dot(atom_points, M_inv.T)
            x %= 1
            atom_points = np.dot(x, M.T)

        atom_points += origin
        N_atoms = len(atom_points)
        with open(file_path, "w") as file_:
            # header (ignored)
            file_.write("# atom coordinates generated by 'nwlattice' package\n")
            file_.write("\n")

            # number of atoms and number of atom types
            file_.write("%d atoms\n" % N_atoms)
            file_.write("%d atom types\n" % len(self._basis))
            file_.write("\n")

            # write simulation box
            file_.write("{:.6f} {:.6f} xlo xhi\n".format(origin[0], xx + origin[0]))
            file_.write("{:.6f} {:.6f} ylo yhi\n".format(origin[1], yy + origin[1]))
            file_.write("{:.6f} {:.6f} zlo zhi\n".format(origin[2], zz + origin[2]))
            file_.write("{:.6f} {:.6f} {:.6f} xy xz yz\n".format(xy, xz, yz))
            file_.write("\n")

            # Atoms section
            file_.write("Atoms # atomic\n")
            file_.write("\n")
            for pt, typ, _id in zip(atom_points, atom_types, atom_ids):
                file_.write("{:d} {:d} {:.6f} {:.6f} {:.6f} 0 0 0\n"
                            .format(_id, typ, pt[0], pt[1], pt[2]))
            t2 = time()
            self.print("wrote %d atoms to data file '%s' in %f seconds"
                       % (N_atoms, file_path, t2 - t1))

    def write_layer_map(self, file_path: str = None, planes_per_layer: int = 1) -> None:
        """
        Write a file specifying which atom ID's belong to which layer.
        The file will look like this:

            # comment line
            1 N1
                type11 id11
                type12 id12
                type13 id13
                ....
                type1(N1) id1(N1)
            2 N2
                type21 id21
                type22 id22
                ...
                type2(N2) id2(N2)
            ...

        :param planes_per_layer: (int) number of planes (including all basis atoms) in a layer
        :param file_path: (str) output file path
        """
        if file_path is None:
            file_path = f"{self.type_name}_layers.map"
        file_path = expanduser(file_path)

        layer_map = self.get_layer_map(planes_per_layer)
        atom_types = self.get_types()

        with open(file_path, 'w') as file_:
            # header (ignored)
            file_.write("# layer map generated by 'nwlattice' package\n")

            for layer_number, atom_ids in enumerate(layer_map):
                file_.write("layer {:d} {:d}\n".format(layer_number, len(atom_ids)))
                for atom_id in atom_ids:
                    file_.write("\t{:d} {:d}\n".format(atom_id, atom_types[atom_id - 1], ))


class NanowireLattice(IDataWriter):
    """
    Base class for nanowire lattice objects (planes stacked along z-axis)
    """

    def get_size(self, *args):
        raise NotImplementedError

    @classmethod
    def get_supercell(cls, *args, **kwargs):
        # subclass-specific method
        raise NotImplementedError

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        """returns nearest integer lattice width from continuous width"""
        raise NotImplementedError

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        """returns continuous width from integer lattice width"""
        raise NotImplementedError

    def __init__(self, size_obj, planes, vr):
        super().__init__()
        self._size_obj = size_obj  # dimensions/geometry handler
        self._planes = []  # list ref's to every comprising plane
        for plane in planes:
            self._planes.append(plane)

        self._vz = np.zeros((self.size.nz, 3))  # plane z positions
        for i in range(self.size.nz):
            self._vz[i][2] = i * self.size.unit_dz
        self._vr = np.reshape(vr, (self.size.nz, 3))  # plane xy positions

        self._N = None  # number of lattice points
        self._supercell = self  # a minimum-length instance of the same class
        self._basis = {1: [np.zeros(3)]}  # points tacked onto lattice points
        self._area = None  # average cross sectional area among planes
        self._v_center_com = np.zeros(3)  # vector to center the structure
        self._v_offset = np.zeros(3)

    def __add__(self, other):
        types = (type(self), type(other))

        if types[1] is not types[0]:
            if not (CompoundNanowire in types or NanowireLattice in types):
                raise TypeError("can not add incompatible types %s and %s"
                                % (types[0], types[1]))

        s1 = self.size
        s2 = other.size

        if s1.scale != s2.scale:
            raise ValueError("nanowire sizes have unequal scales %f and %f"
                             % (s1.scale, s2.scale))

        if s1.n_xy != s2.n_xy:
            raise ValueError("nanowire sizes have unequal n_xy %d and %d"
                             % (s1.n_xy, s2.n_xy))

        if s1.unit_dz != s2.unit_dz:
            raise ValueError("nanowire sizes have unequal unit_dz %f and %f"
                             % (s1.unit_dz, s2.unit_dz))

        ps1 = s1.props()
        ps2 = s2.props()

        for p in ps1:
            if p not in ps2:
                raise ValueError("nanowire size property sets are unequal")
        if len(ps1) != len(ps2):
            raise ValueError("nanowire size property sets are unequal")

        ps = ps1.copy()
        for p in ps2:
            if ps[p] != ps2[p]:
                ps[p] = (ps[p], ps2[p])

        ps["nz"] = s1.nz + s2.nz

        # make sure planes fit at point of joining
        if self.planes[-1].fits(other.planes[0]):
            planes = self.planes + other.planes
            vr = np.concatenate((self.vr, other.vr))
        elif self.planes[-1].fits(other.planes[1]):
            # truncate by 1 plane if that'll make it fit
            planes = self.planes + other.planes[1:]
            vr = np.concatenate((self.vr, other.vr[1:]))
            ps["nz"] -= 1
            self.warn("truncated top planes by 1 to force fit")
        else:
            raise ValueError("could not fit constituent planes")

        # try to establish z-periodicity
        if planes[0].fits(planes[-1]):
            pass
        elif planes[0].fits(planes[-2]):
            self.warn("truncating top planes by 1 to force z-continuity")
            planes = planes[:-1]
            vr = vr[:-1]
            ps["nz"] -= 1
        else:
            raise ValueError("could not establish z-periodicity")

        ps["length"] = s1.scale * s1.unit_dz * (ps["nz"] - 1)
        size = NanowireSizeCompound(**ps)

        sum_wire = CompoundNanowire(size, planes, vr)
        sum_wire._basis = self.basis.copy()
        return sum_wire

    def __str__(self):
        s = "<{} nanowire: ".format(self.type_name)
        s_args = []
        for arg in ['scale', 'width', 'length', 'period', 'fraction']:
            if hasattr(self.size, arg):
                s_args.append("{:.2f}".format(self.size.__getattribute__(arg)))

        s += "(" + ", ".join(s_args) + ") "
        s += ">"
        return s

    @staticmethod
    def get_length(scale: float, nz, unit_dz) -> float:
        """returns continuous length from number of planes"""
        return scale * (nz - 1) * unit_dz

    @staticmethod
    def get_nz(scale: float, length: float, unit_dz) -> int:
        """returns number of planes from continuous length"""
        return int(length / scale / unit_dz)

    @staticmethod
    def get_cyclic_nz(nz, k, nearest=True):
        """
        returns int(s) `nlo` or(and) `nhi` multiple(s) of `k` nearest to `nz`
        """
        nlo = (nz // k) * k
        nhi = ((nz + k) // k) * k
        if nearest:
            if nlo == 0:
                return nhi
            elif (nz - nlo) < (nhi - nz):
                return nlo
            else:
                return nhi
        else:
            return nlo, nhi

    @property
    def basis(self):
        return self._basis

    @property
    def N(self):
        if self._N is None:
            N = 0
            for plane in self.planes:
                N += plane.N
            self._N = N
        return self._N

    @property
    def size(self):
        return self._size_obj

    @property
    def planes(self):
        return self._planes

    @property
    def vr(self):
        return self._vr

    @property
    def vz(self):
        return self._vz

    @property
    def supercell(self):
        # the supercell is updated from `self` when this property is accessed
        # NOTE: this should be overridden for non-periodic nanowire lattices
        if self._supercell is self:
            scale = self.size.scale
            n_xy = self.size.n_xy
            self._supercell = self.get_supercell(scale, n_xy=n_xy)
        return self._supercell

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

    def invert(self) -> None:
        self._planes = self._planes[::-1]
        self._vr = self._vr[::-1]

    def inverted(self):
        """returns a new instance with the planes stacked backwards"""
        inv_wire = NanowireLattice(self.size, self._planes[::-1], self._vr[::-1])
        for t, pts in self.basis.items():
            for pt in pts:
                inv_wire.add_basis(t, pt)

        if isinstance(self.size, NanowireSizeArbitrary):
            inv_wire.size.invert_index()

        return inv_wire

    def mirrored(self):
        """
        Returns a new instance made of this wire plus an inverted copy appended on top.
        Note that at least 2 planes must/will be cut away: to ensure continuity at the junction,
        and to force lattice periodicity in z.
        """
        return self + self.inverted()

    def rotate_vz(self, n):
        self._planes = self._planes[-n:] + self._planes[:-n]
        self._vr = np.concatenate((self._vr[-n:], self._vr[:-n]))

    def add_offset(self, v):
        """
        add offset vector to be applied to get_points() output
        """
        self._v_offset += v

    def get_ids(self) -> np.ndarray:
        """
        Return an array of atom ids belonging to given plane index.
        Simply counts from 1 to `self.N`.
        """
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        return np.arange(self.N * n_basis_atoms) + 1

    def get_points(self) -> np.ndarray:
        """
        Return an array of all atom points of  all types
        """
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        atom_pts = np.zeros((self.N * n_basis_atoms, 3), dtype=float)
        n = 0

        for t in self.basis:
            for bpt in self.basis[t]:
                for i in range(self.size.nz):
                    plane = self.planes[i]
                    atom_pts[n:(n + plane.N)] = (
                            plane.points + bpt + self.vr[i] + self.vz[i]
                    )
                    n += plane.N

        scale = self.size.scale
        return scale * (atom_pts + self._v_center_com) + self._v_offset

    def get_types(self) -> np.ndarray:
        """
        Return the integer type identifier for atoms identifier `ID`
        """
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        types = np.zeros(self.N * n_basis_atoms, dtype=int)
        ID = 1

        for t in self.basis:
            for _ in self.basis[t]:
                for i in range(self.size.nz):
                    plane = self.planes[i]
                    types[(ID - 1):(ID - 1) + plane.N] = t
                    ID += plane.N

        return types

    def get_layer_map(self, planes_per_layer: int = 1) -> List[list]:
        """

        :param planes_per_layer: (int) number of planes (including all basis atoms) in a layer
        :return: a list of lists of the form [layer number][atom id in layer]
        """
        N_planes = len(self.planes)
        if len(self.planes) % planes_per_layer != 0:
            raise ValueError("number of planes in structure ({:d}) is not divisible by {:d}"
                             .format(N_planes, planes_per_layer))
        n_layers = N_planes // planes_per_layer

        layer_map = [[] for _ in range(n_layers)]
        ids = self.get_ids()
        id_counter = 0

        for t in self.basis:
            for _ in self.basis[t]:
                for i in range(self.size.nz):
                    layer_index = i // planes_per_layer
                    for j in range(self.planes[i].N):
                        layer_map[layer_index].append(ids[id_counter])
                        id_counter += 1

        return layer_map

    def write_points(self,
                     file_path: str = None,
                     xy_space: float = None,
                     cell_type: str = "vector",
                     center_points: bool = True,
                     wrap_points: bool = False,
                     origin=None):
        """
        Write LAMMPS/OVITO compatible data file of all atom points

        :param file_path: string indicating target file (created/overwritten)
        :param xy_space: minimum perpendicular distance from nanowire edge to box wall
        :param cell_type: simulation cell shape, either 'ortho', or parallel to 'vector's
        :param origin: origin of the simulation cell; default [0, 0, 0]
        :param center_points: translate points to the centre of the cell
        :param wrap_points: wrap points at periodic boundaries
        :return: None
        """
        if xy_space is None:
            xy_space = self.size.width

        if cell_type == "vector":
            (xx, yy, zz, xy, xz, yz) = self._get_vector_box_dims(xy_space)
        elif cell_type == "ortho":
            (xx, yy, zz, xy, xz, yz) = self._get_ortho_box_dims(xy_space)
        else:
            raise ValueError("invalid cell type; must be one of 'ortho' or 'vectors'")

        if origin is None:
            origin = np.array([0., 0., 0.])

        if file_path is None:
            file_path = f"{self.type_name}_structure.data"

        t1 = time()

        file_path = expanduser(file_path)
        atom_types = self.get_types()
        atom_points = self.get_points()
        atom_ids = self.get_ids()
        if center_points:
            atom_points += .5 * np.array([xx + xy, yy, (zz - self.size.length) / 4.])

        if wrap_points:
            M = np.array([[xx, xy, xz],
                          [0., yy, yz],
                          [0., 0., zz]])
            M_inv = np.linalg.inv(M)
            x = np.dot(atom_points, M_inv.T)
            x %= 1
            atom_points = np.dot(x, M.T)

        atom_points += origin
        N_atoms = len(atom_points)
        with open(file_path, "w") as file_:
            # header (ignored)
            file_.write("# atom coordinates generated by 'nwlattice' package\n")
            file_.write("\n")

            # number of atoms and number of atom types
            file_.write("%d atoms\n" % N_atoms)
            file_.write("%d atom types\n" % len(self.basis))
            file_.write("\n")

            # write simulation box
            file_.write("{:.6f} {:.6f} xlo xhi\n".format(origin[0], xx + origin[0]))
            file_.write("{:.6f} {:.6f} ylo yhi\n".format(origin[1], yy + origin[1]))
            file_.write("{:.6f} {:.6f} zlo zhi\n".format(origin[2], zz + origin[2]))
            file_.write("{:.6f} {:.6f} {:.6f} xy xz yz\n".format(xy, xz, yz))
            file_.write("\n")

            # Atoms section
            file_.write("Atoms # atomic\n")
            file_.write("\n")
            for pt, typ, _id in zip(atom_points, atom_types, atom_ids):
                file_.write("{:d} {:d} {:.6f} {:.6f} {:.6f} 0 0 0\n"
                            .format(_id, typ, pt[0], pt[1], pt[2]))
            t2 = time()
            self.print("wrote %d atoms to data file '%s' in %f seconds"
                       % (N_atoms, file_path, t2 - t1))

    def write_layer_map(self, file_path: str = None, planes_per_layer: int = 1) -> None:
        """
        Write a file specifying which atom ID's belong to which layer.
        The file will look like this:

            # comment line
            1 N1
                type11 id11
                type12 id12
                type13 id13
                ....
                type1(N1) id1(N1)
            2 N2
                type21 id21
                type22 id22
                ...
                type2(N2) id2(N2)
            ...

        :param planes_per_layer: (int) number of planes (including all basis atoms) in a layer
        :param file_path: (str) output file path
        """
        if file_path is None:
            file_path = f"{self.type_name}_layers.map"
        file_path = expanduser(file_path)

        layer_map = self.get_layer_map(planes_per_layer)
        atom_types = self.get_types()

        with open(file_path, 'w') as file_:
            # header (ignored)
            file_.write("# layer map generated by 'nwlattice' package\n")

            for layer_number, atom_ids in enumerate(layer_map):
                file_.write("layer {:d} {:d}\n".format(layer_number, len(atom_ids)))
                for atom_id in atom_ids:
                    file_.write("\t{:d} {:d}\n".format(atom_types[atom_id - 1], atom_id))

    def get_area(self) -> float:
        """returns average cross-sectional area of comprising planes"""
        if self._area is None:
            sum_area = 0
            n = 0
            for plane in self.planes:
                sum_area += plane.size.area
                n += 1
            self._area = sum_area / n * self.size.scale**2
        return self._area

    def _get_ortho_box_dims(self, xy_space: float) -> tuple:
        """returns simulation box dimensions for write_points() method"""
        xx = yy = self.size.width + 2 * xy_space  # x and y sizes of the box
        zz = self.size.length + self.size.scale * self.size.unit_dz
        return xx, yy, zz, 0., 0., 0.

    def _get_vector_box_dims(self, xy_space: float) -> tuple:
        u1, u2 = self.planes[0].get_vectors()
        theta = np.arccos(np.dot(u1, u2)
                          / (np.linalg.norm(u1) * np.linalg.norm(u2)))
        v_mag = (self.size.width + 2 * xy_space) / np.sin(theta)
        v1 = v_mag * u1
        v2 = v_mag * u2

        xx = v1[0]
        yy = v2[1]
        zz = self.size.length + self.size.scale * self.size.unit_dz
        xy = v2[0]
        xz = 0.0
        yz = 0.0

        return xx, yy, zz, xy, xz, yz


class CompoundNanowire(NanowireLattice):
    def __init__(self, size_obj, planes, vr):
        super().__init__(size_obj, planes, vr)

    @classmethod
    def get_supercell(cls, *args, **kwargs):
        """dummy method: not applicable in general"""
        return cls(*args, **kwargs)

    @staticmethod
    def get_n_xy(scale: float, width: float) -> int:
        raise NotImplementedError

    @staticmethod
    def get_width(scale: float, n_xy) -> float:
        raise NotImplementedError

    def get_size(self, *args):
        raise NotImplementedError


class NanowireLatticePeriodic(NanowireLattice):
    """
    Base class for periodic twinning nanowire lattice objects
    """

    @classmethod
    @abstractmethod
    def get_supercell(cls, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_n_xy(scale: float, width: float) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_width(scale: float, n_xy) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_q(scale: float, period: float) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_period(scale: float, p: int) -> float:
        raise NotImplementedError

    @property
    def supercell(self):
        # the supercell is updated from `self` when this property is accessed
        # NOTE: this should be overridden for non-periodic nanowire lattices
        if self._supercell is self:
            scale = self.size.scale
            n_xy = self.size.n_xy
            q = self.size.q
            self._supercell = self.get_supercell(scale, n_xy=n_xy, q=q)
        return self._supercell


class NanowireLatticeArbitrary(NanowireLattice):
    """
    Base class for arbitrarily twinning nanowire lattice objects
    """

    @property
    def supercell(self):
        return self

    @classmethod
    def get_supercell(cls, *args, **kwargs):
        """dummy method: not applicable in general"""
        return cls(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_n_xy(scale: float, width: float) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_width(scale: float, n_xy) -> float:
        raise NotImplementedError
