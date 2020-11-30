from abc import ABC, abstractmethod
import numpy as np
from time import time
from os.path import expanduser

from nwlattice.utilities import ROOT3, Quaternion


class APointPlane(ABC):
    """Abstract base class for planes of points stacked by AStackLattice"""
    PRINT = False

    # translation vectors for planar lattices
    hex_vectors = np.array([[1., 0., 0.], [-.5, ROOT3 / 2., 0.]])
    sq_vectors = np.array([[1., 0., 0.], [0., 1., 0.]])

    # centering offset vectors for planar lattices
    ohex_delta = .5 * np.array([1., 1. / ROOT3, 0.])
    ehex_delta = .25 * np.array([1, -1. / ROOT3, 0.])

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
        # implement N formula for each subclass
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
        # set translation vectors for each subclass
        raise NotImplementedError

    @property
    @abstractmethod
    def com(self):
        # implement com formula for each subclass
        raise NotImplementedError

    def __init__(self, scale):
        super().__init__()
        self._N = None  # number of lattice points
        self._D = None  # diameter
        self._vectors = None  # translation vectors
        self._area = None  # cross sectional area
        self._com = None  # points centre of mass
        self._scale = scale  # scaling factor
        self._points = None  # array of points in this PointPlane

    @property
    def scale(self):
        # global scaling of all point in plane
        return self._scale

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
            for pt in self.get_points(center=True):
                pt *= self._scale
                file_.write("{} {} {} {} {} 0 0 0\n"
                            .format(id_, 1, pt[0], pt[1], pt[2]))
                id_ += 1
            t2 = time()
            self.print("wrote %d atoms to map file '%s' in %f seconds"
                       % (N_atoms, file_path, t2 - t1))

    @staticmethod
    def print(s):
        if APointPlane.PRINT:
            print(s)


class AStackLattice(ABC):
    PRINT = False

    @classmethod
    @abstractmethod
    def get_supercell(cls, *args):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dimensions(cls, *args):
        """create stack lattice from measurements instead of indices"""
        raise NotImplementedError

    @property
    @abstractmethod
    def supercell(self):
        raise NotImplementedError

    def __init__(self, planes, vz_unit, vr):
        super().__init__()
        self._supercell = self  # an minimal-nz instance of the same class
        self._planes = []  # list of PointPlane objects to be stacked
        self._N = None  # number of lattice points
        self._nz = None  # number of planes in wire lattice
        self._vz_unit = float(vz_unit)  # unit inter-planar dist. in z direction
        self._vz = None  # array of offsets from 0. to plane z in a0=1 units
        self._vr = None  # xy-plane offset between planes in a0=1 units
        self._D = None  # actual diameter (scaled)
        self._L = None  # actual length (scaled)
        self._P = None  # actual period (scaled)
        self._area = None  # cross sectional area (scaled)
        self._basis = {1: [np.zeros(3)]}  # atom types attached to lattice
        self._v_center_com = np.zeros(3)  # vector that centers lattice in box
        self._scale = 1.0  # scaling factor; effectively lattice constant

        for plane in planes:
            if isinstance(plane, APointPlane):
                self._planes.append(plane)
            else:
                raise TypeError("all items in planes list must be PointPlanes")

        self._vz = np.zeros((self.nz, 3))
        for i in range(self.nz):
            self._vz[i][2] = i * self._vz_unit
        self._vr = np.reshape(vr, (self.nz, 3))  # xy offset for each plane

    @property
    def scale(self):
        return self._scale

    @property
    def vz_unit(self):
        """unit offset between planes in scaled units"""
        return self._vz_unit * self._scale

    @property
    def L(self):
        """real length in scaled units"""
        if self._L is None and self._vz_unit:
            self._L = self._scale * (self._nz - 1) * self._vz_unit
        return self._L

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
        """list of unique planes in the stack"""
        return self._planes

    @property
    def N(self):
        """number of lattice points in the stack"""
        if self._N is None:
            N = 0
            for plane in self.planes:
                N += plane.N
            self._N = N
        return self._N

    @property
    def nz(self):
        """number of vertically stacked planes"""
        if self._nz is None:
            self._nz = len(self.planes)
        return self._nz

    @property
    def basis(self):
        """dictionary of atom points added to each lattice point"""
        return self._basis

    @property
    def vz(self):
        """z offset for each plane"""
        return self._vz

    @property
    def vr(self):
        """xy offset for each plane"""
        return self._vr

    @property
    def type_name(self):
        """string identifying each sub-class"""
        return str(self.__class__).split('.')[-1].strip("'>")

    def cycle_z(self, n):
        self._planes = self._planes[-n:] + self._planes[:-n]
        self._vr = np.concatenate((self._vr[-n:], self._vr[:-n]))

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

    def get_types(self) -> np.ndarray:
        """
        Return the integer type identifier for atoms identifier `ID`

        :return: atom type
        """
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        n_supercell_planes = len(self.supercell.planes)
        types = np.zeros(self.N * n_basis_atoms)
        ID = 1
        for i in range(self._nz):
            plane = self.supercell.planes[i % n_supercell_planes]
            for t in self.basis:
                for _ in self.basis[t]:
                    types[(ID - 1):(ID - 1) + plane.N] = t
                    ID += plane.N

        return types

    def get_points(self, **kwargs) -> np.ndarray:
        """
        Return an array of all atom points of type `t`

        :return: array of 3-points indicating all locations of type `t` atoms
        """
        n_supercell_planes = len(self.supercell.planes)
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        atom_pts = np.zeros((self.N * n_basis_atoms, 3))
        n_ID = 0
        for i in range(self._nz):
            plane = self.supercell.planes[i % n_supercell_planes]
            for t in self.basis:
                for bpt in self.basis[t]:
                    atom_pts[n_ID:(n_ID + plane.N)] = (
                            plane.get_points(center=True)
                            + self.vr[i]
                            + self.vz[i]
                            + bpt
                    )
                    n_ID += plane.N

        return atom_pts + self._v_center_com

    def get_map(self):
        """return `phana` map row for every atom"""
        # rows = [l1, l2, l3, k, ID]
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        rows = []
        l1 = l2 = l3 = 0
        k = 0
        ID = 1
        n = 0
        for i in range(self._nz):
            plane = self.supercell.planes[i % self.supercell.nz]
            for t in self.basis:
                for _ in plane.get_points(t):
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

    def _get_points_box_dims(self, wrap: bool) -> tuple:
        """returns simulation box dimensions for write_points() method"""
        x = 2 * self.D  # keep atoms' (x,y) in box
        y = x
        n_atom_types = len(self._basis)
        basis_z_min = basis_z_max = 0.
        if n_atom_types > 1:
            for t in self._basis:
                for bpt in self._basis[t]:
                    if bpt[2] < basis_z_min:
                        basis_z_min = bpt[2] * self._scale
                    elif bpt[2] > basis_z_max:
                        basis_z_max = bpt[2] * self._scale
        zlo = 0. if wrap else basis_z_min
        zhi = self.L
        if wrap:
            zhi += self.vz_unit
        else:
            zhi += basis_z_max
        return -x / 2, x / 2, -y / 2, y / 2, zlo, zhi

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
                pt *= self._scale
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
        cell_nz = self.supercell.nz
        cell_N = self.supercell.N
        if self.nz % cell_nz != 0:
            raise ArithmeticError("could not divide twinstack into integer "
                                  "number of cells; stack is not z-periodic")
        n_cells = self.nz // cell_nz
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        n_atoms_cell = n_basis_atoms * cell_N

        t1 = time()
        with open(file_path, 'w+') as file_:
            # first line
            file_.write("1 1 %d %d\n" % (n_cells, n_atoms_cell))

            # second line (ignored)
            file_.write("# l1 l2 l3 k id (generated by 'nwlattice' package)\n")

            # data lines
            for row in self.get_map():
                file_.write("%d %d %d %d %d\n" % row)
        t2 = time()
        self.print("wrote %d atoms to data file '%s' in %f seconds"
                   % (n_atoms_cell * n_cells, file_path, t2 - t1))

    @staticmethod
    def get_cyclic_nz(*args):
        """
        Converts `nz` to the nearest multiple of 3 such that 3-plane super cell
        structures maintain lattice periodicity given z-periodicity

        NOTE: must override this method for twinning structures!

        :param args: args[0] should be `nz`
        :return: an int `nhi` or `nlo`, the multiple of 3 nearest to `nz`
        """
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

    @staticmethod
    def print(s):
        if AStackLattice.PRINT:
            print(s)


class ATwinStackLattice(AStackLattice):
    @classmethod
    @abstractmethod
    def from_dimensions(cls, *args):
        """create stack lattice from measurements instead of indices"""
        raise NotImplementedError

    def __init__(self, planes, vz_unit, vr, q_max, theta):
        super().__init__(planes, vz_unit, vr)
        self._q_max = q_max
        self._theta = theta
        self._deg = theta * 180. / np.pi
        self._qtr = Quaternion.rotator([0, 0, 1], theta)

    @property
    def q_max(self):
        return self._q_max

    def get_points(self) -> np.ndarray:
        """
        Return an array of all atom points of type `t`

        :return: array of 3-points indicating all locations of type `t` atoms
        """
        n_supercell_planes = len(self.supercell.planes)
        n_basis_atoms = sum([len(self.basis[t]) for t in self.basis])
        atom_pts = np.zeros((self.N * n_basis_atoms, 3))
        n_ID = 0
        for i in range(self._nz):
            plane = self.supercell.planes[i % n_supercell_planes]
            for t in self.basis:
                for bpt in self.basis[t]:
                    ss_nz = self.supercell.nz
                    if (i % (2 * ss_nz) > ss_nz) and t > 1:
                        _bpt = self._qtr.rotate(bpt)  # rotate non-primary bpts
                    else:
                        _bpt = bpt

                    atom_pts[n_ID:(n_ID + plane.N)] = (
                            plane.get_points(center=True)
                            + self.vr[i]
                            + self.vz[i]
                            + _bpt
                    )
                    n_ID += plane.N

        return atom_pts + self._v_center_com
