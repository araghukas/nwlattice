from abc import ABC, abstractmethod
from os.path import expanduser
from time import time
import numpy as np

from nwlattice2.utilities import Quaternion as Qtr


class IDataWriter(ABC):
    """
    The interface that writes the LAMMPS/phana atom data and map files
    """
    WILL_PRINT = False

    @abstractmethod
    def write_points(self, file_path: str):
        raise NotImplementedError

    @abstractmethod
    def write_map(self, file_path: str):
        raise NotImplementedError

    @staticmethod
    def print(s):
        if IDataWriter.WILL_PRINT:
            print(s)


class IPointPlane(ABC):
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

    def __init__(self):
        self._N = None
        self._v_center_com = np.zeros(3)

    @property
    @abstractmethod
    def N(self):
        raise NotImplementedError

    @property
    def type_name(self):
        """string identifying each sub-class"""
        return str(self.__class__).split('.')[-1].strip("'>")


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

    def __init__(self):
        self._supercell = self
        self._N = None
        self._basis = {1: [np.zeros(3)]}

        self._v_center_com = np.zeros(3)

    @classmethod
    @abstractmethod
    def get_supercell(cls, *args):
        raise NotImplementedError

    @property
    @abstractmethod
    def N(self):
        raise NotImplementedError

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


class APointPlane(IDataWriter, IPointPlane):
    """
    Base class for planar base lattices. These do not support multi-atom bases.
    """

    def __init__(self, size_obj, vectors, theta=None):
        super().__init__()
        self._vectors = vectors  # translation vectors
        self._size_obj = size_obj
        self._theta = theta

        self._N = None  # number of lattice points
        self._com = None  # points centre of mass
        self._points = None  # array of points in this PlaneLattice

    @staticmethod
    @abstractmethod
    def get_n_xy(scale, width):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_width(scale, n_xy):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_area(scale, n_xy):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_vectors():
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
        if self._points is None:
            self._points = self.get_points()
            if self.theta is not None:
                axis = [0, 0, 1]
                self._points = Qtr.qrotate(self._points, axis, self.theta)
        return self.size.scale * self._points

    @property
    def theta(self):
        return self._theta

    def write_points(self, file_path: str):
        """
        Write LAMMPS/OVITO compatible data file of all atom points

        :param file_path: string indicating target file (created/overwritten)
        :return: None
        """
        N_atoms = self.N  # total number of atoms

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

            # Atoms section
            file_.write("Atoms # atomic\n")
            file_.write("\n")
            id_ = 1
            for pt in self.points:
                file_.write("{} {} {} {} {} 0 0 0\n"
                            .format(id_, 1, pt[0], pt[1], pt[2]))
                id_ += 1
            t2 = time()
            self.print("wrote %d atoms to map file '%s' in %f seconds"
                       % (N_atoms, file_path, t2 - t1))

    def write_map(self, file_path: str):
        """
        Write a map file for the LAMMPS command `fix phonon`

        The map file conforms to the specification found at:
        <https://code.google.com/archive/p/fix-phonon/wikis/MapFile.wiki>

        Basically,

            The map file simply shows the map between the lattice indices of an
            atom and its unique id in the simulation box. The format of the map
            file read:

                nx ny nz n
                comment line
                l1 l2 l3 k id
                ...

            The first line: nx, ny, and nz are the number of extensions of the
            unit cell in x, y, and z directions, respectively. That is to say,
            your simulation box contains nx x ny x nz unit cells. And n is the
            number of atoms in each unit cell.

            The second line: comment line, put whatever you want. From line 3
            to line (nx*ny*nz*n+2): l1, l2, l3 are the indices of the unit cell
            which atom id belongs to, and the atom corresponding to atom id is
            the k_th atom in this unit cell, starting at k = 0.

        For a stack of twins, the smallest repeating unit is the set of
        consecutive planes in a full q-cycle. Use this as the supercell.

        NOTE: atom ID's must be consistent with `write_points()` method output
        """
        file_path = expanduser(file_path)
        n_basis_atoms = 1
        n_atoms_cell = n_basis_atoms * self.N

        t1 = time()
        with open(file_path, 'w+') as file_:
            # first line
            file_.write("1 1 %d %d\n" % (1, n_atoms_cell))

            # second line (ignored)
            file_.write("# l1 l2 l3 k id (generated by 'nwlattice' package)\n")

            # data lines
            for row in self.get_map_rows():
                file_.write("%d %d %d %d %d\n" % row)
        t2 = time()
        self.print("wrote %d atoms to data file '%s' in %f seconds"
                   % (n_atoms_cell, file_path, t2 - t1))

    def get_points(self) -> np.ndarray:
        # subclass-specific method
        raise NotImplementedError

    def get_types(self) -> np.ndarray:
        """
        Return the integer type identifier for atoms identifier `ID`
        :return: array of atom types
        """
        return np.ones(self.N)

    def get_map_rows(self) -> list:
        """return `phana` map row for every atom"""
        rows = []
        l1 = l2 = l3 = 0
        k = 0
        ID = 1

        for i in range(self.N):
            rows.append((l1, l2, l3, k, ID))
            k += 1
            ID += 1

        return rows


class ANanowireLattice(IDataWriter, IPointLattice):
    """
    Base class for nanowire lattice (planes stacked along z-axis) objects
    """

    def __init__(self, size_obj, planes, vr):
        super().__init__()
        self._size_obj = size_obj
        self._planes = []
        for plane in planes:
            self._planes.append(plane)

        self._vz = np.zeros((self.size.nz, 3))
        for i in range(self.size.nz):
            self._vz[i][2] = i * size_obj.unit_dz
        self._vr = np.reshape(vr, (self.size.nz, 3))

        self._N = None
        self._supercell = self
        self._area = None

    @classmethod
    @abstractmethod
    def get_supercell(cls, *args):
        # subclass-specific method
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_n_xy(scale, width):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_width(scale, n_xy):
        raise NotImplementedError

    @staticmethod
    def get_length(scale, nz, unit_dz) -> float:
        return scale * (nz - 1) * unit_dz

    @staticmethod
    def get_nz(scale, length, unit_dz) -> int:
        return int(length / scale / unit_dz)

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
        return self._supercell

    def write_points(self, file_path: str, wrap=True):
        """
        Write LAMMPS/OVITO compatible data file of all atom points

        :param file_path: string indicating target file (created/overwritten)
        :param wrap: toggle taking (z_coord % zhi) when writing atoms to file
        :return: None
        """
        t1 = time()
        file_path = expanduser(file_path)
        atom_points = self.get_points()
        atom_types = self.get_types()
        N_atoms = len(atom_points)
        with open(file_path, "w") as file_:
            # header (ignored)
            file_.write("atom coordinates generated by 'nwlattice' package\n")
            file_.write("\n")

            # number of atoms and number of atom types
            file_.write("%d atoms\n" % N_atoms)
            file_.write("%d atom types\n" % len(self.basis))
            file_.write("\n")

            # decide simulation box dimensions
            xlo, xhi, ylo, yhi, zlo, zhi = self._get_points_box_dims(wrap)

            # write simulation box
            file_.write("{} {} xlo xhi\n".format(xlo, xhi))
            file_.write("{} {} ylo yhi\n".format(ylo, yhi))
            file_.write("{} {} zlo zhi\n".format(zlo, zhi))
            file_.write("\n")

            # Atoms section
            file_.write("Atoms # atomic\n")
            file_.write("\n")
            id_ = 1
            for pt, typ in zip(atom_points, atom_types):
                ptz = pt[2] % zhi if wrap else pt[2]
                file_.write("{} {} {} {} {} 0 0 0\n"
                            .format(id_, typ, pt[0], pt[1], ptz))
                id_ += 1
            t2 = time()
            self.print("wrote %d atoms to data file '%s' in %f seconds"
                       % (N_atoms, file_path, t2 - t1))

    def write_map(self, file_path: str):
        """
        Write a map file for the LAMMPS command `fix phonon`

        The map file conforms to the specification found at:
        <https://code.google.com/archive/p/fix-phonon/wikis/MapFile.wiki>

        Basically,

            The map file simply shows the map between the lattice indices of an
            atom and its unique id in the simulation box. The format of the map
            file read:

                nx ny nz n
                comment line
                l1 l2 l3 k id
                ...

            The first line: nx, ny, and nz are the number of extensions of the
            unit cell in x, y, and z directions, respectively. That is to say,
            your simulation box contains nx x ny x nz unit cells. And n is the
            number of atoms in each unit cell.

            The second line: comment line, put whatever you want. From line 3
            to line (nx*ny*nz*n+2): l1, l2, l3 are the indices of the unit cell
            which atom id belongs to, and the atom corresponding to atom id is
            the k_th atom in this unit cell, starting at k = 0.

        For a stack of twins, the smallest repeating unit is the set of
        consecutive planes in a full q-cycle. Use this as the supercell.

        NOTE: atom ID's must be consistent with `write_points()` method output
        """
        file_path = expanduser(file_path)
        cell_nz = self.supercell.size.nz
        cell_N = self.supercell.N
        if self.size.nz % cell_nz != 0:
            raise ArithmeticError("could not divide twinstack into integer "
                                  "number of cells; stack is not z-periodic")
        n_cells = self.size.nz // cell_nz
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        n_atoms_cell = n_basis_atoms * cell_N

        t1 = time()
        with open(file_path, 'w+') as file_:
            # first line
            file_.write("1 1 %d %d\n" % (n_cells, n_atoms_cell))

            # second line (ignored)
            file_.write("# l1 l2 l3 k id (generated by 'nwlattice' package)\n")

            # data lines
            for row in self.get_map_rows():
                file_.write("%d %d %d %d %d\n" % row)
        t2 = time()
        self.print("wrote %d atoms to data file '%s' in %f seconds"
                   % (n_atoms_cell * n_cells, file_path, t2 - t1))

    def get_points(self) -> np.ndarray:
        """
        Return an array of all atom points of type `t`

        :return: array of 3-points indicating all locations of type `t` atoms
        """
        n_supercell_planes = len(self.supercell.planes)
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        atom_pts = np.zeros((self.N * n_basis_atoms, 3))
        n_ID = 0
        for i in range(self.size.nz):
            plane = self.supercell.planes[i % n_supercell_planes]
            for t in self.basis:
                for bpt in self.basis[t]:
                    atom_pts[n_ID:(n_ID + plane.N)] = (
                            plane.points
                            + self.vr[i]
                            + self.vz[i]
                            + bpt
                    )
                    n_ID += plane.N
        return self.size.scale * (atom_pts + self._v_center_com)

    def get_types(self) -> np.ndarray:
        """
        Return the integer type identifier for atoms identifier `ID`

        :return: atom type
        """
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        n_supercell_planes = len(self.supercell.planes)
        types = np.zeros(self.N * n_basis_atoms)
        ID = 1
        for i in range(self.size.nz):
            plane = self.supercell.planes[i % n_supercell_planes]
            for t in self.basis:
                for _ in self.basis[t]:
                    types[(ID - 1):(ID - 1) + plane.N] = t
                    ID += plane.N
        return types

    def get_map_rows(self) -> list:
        """return `phana` map row for every atom"""
        # rows = [l1, l2, l3, k, ID]
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        rows = []
        l1 = l2 = l3 = 0
        k = 0
        ID = 1
        n = 0
        for i in range(self.size.nz):
            plane = self.supercell.planes[i % self.supercell.size.nz]
            for t in self.basis:
                for _ in plane.get_points():
                    for _ in self.basis[t]:
                        rows.append((l1, l2, l3, k, ID))
                        n += 1
                        ID += 1
                        if n % (self.supercell.N * n_basis_atoms) == 0:
                            l3 += 1
                            k = 0
                        else:
                            k += 1
        return rows

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

    def _get_points_box_dims(self, wrap: bool) -> tuple:
        """returns simulation box dimensions for write_points() method"""
        x = 2 * self.size.width  # keep atoms' (x,y) in box
        y = x
        n_atom_types = len(self._basis)
        basis_z_min = basis_z_max = 0.
        if n_atom_types > 1:
            for t in self._basis:
                for bpt in self._basis[t]:
                    if bpt[2] < basis_z_min:
                        basis_z_min = bpt[2] * self.size.scale
                    elif bpt[2] > basis_z_max:
                        basis_z_max = bpt[2] * self.size.scale
        zlo = 0. if wrap else basis_z_min
        zhi = self.size.length
        if wrap:
            zhi += self.size.scale* self.size.unit_dz
        else:
            zhi += basis_z_max
        return -x / 2, x / 2, -y / 2, y / 2, zlo, zhi