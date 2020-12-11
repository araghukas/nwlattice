class PlaneSize(object):
    """
    A size information handler for planar lattices
    """

    def __init__(self, scale, n_xy=None, width=None):
        """
        :param scale: lattice scale (ex: lattice constant)
        :param n_xy: structure specific integer thickness index indicating width
        :param width: width in sase units as `a0`
        """
        if not (width or n_xy):
            raise ValueError("must specify wither `n_xy` or `width`")

        if scale <= 0:
            raise ValueError("`scale` must be a positive number")
        self._scale = scale

        if n_xy is not None and n_xy <= 0:
            raise ValueError("`n_xy` must be a positive integer")
        self._n_xy = n_xy

        if width is not None and width <= 0:
            raise ValueError("`width` must be a positive number")
        self._width = width

        # size calculator functions
        self._n_xy_func = None
        self._width_func = None
        self._area_func = None

    def __str__(self):
        return (self.__repr__() + "\n"
                                  "scale: {:<20}\n"
                                  "n_xy : {:<20}\n"
                                  "width: {:<20}\n"
                                  "area : {:<20}"
                ).format(self.scale, self.n_xy, self.width, self.area)

    @property
    def n_xy(self):
        if self._n_xy is None:
            self._n_xy = self._n_xy_func(self.scale, self._width)
        return self._n_xy

    @property
    def width(self):
        return self._width_func(self.scale, self.n_xy)

    @property
    def scale(self):
        return self._scale

    @property
    def area(self):
        return self._area_func(self.scale, self.n_xy)


class NanowireSize(PlaneSize):
    """
    A size information handler for nanowire lattices
    """

    def __init__(self, scale, unit_dz, n_xy=None, nz=None,
                 width=None, length=None):
        """
        :param scale: lattice scale (ex: lattice constant)
        :param nz: number of planes stacked along z-axis
        :param n_xy: structure specific integer thickness index indicating width
        :param length: length in same units as `a0`
        :param width: width in sase units as `a0`
        """
        super().__init__(scale, n_xy, width)
        if not (nz or length):
            raise ValueError("must specify either `nz` or `length`")
        self._unit_dz = unit_dz
        self._nz = nz
        self._length = length

        # size calculator functions
        self._nz_func = None
        self._length_func = None

    def __str__(self):
        return (self.__repr__() + "\n"
                                  "scale : {:<20}\n"
                                  "n_xy  : {:<20}\n"
                                  "width : {:<20}\n"
                                  "nz    : {:<20}\n"
                                  "length: {:<20}\n"
                                  "area  : {:<20}\n"
                ).format(self.scale, self.n_xy, self.width, self.nz,
                         self.length,
                         self.area)

    @property
    def area(self):
        return self._area_func()

    @property
    def unit_dz(self):
        return self._unit_dz

    @property
    def nz(self):
        if self._nz is None:
            self._nz = self._nz_func(self.scale, self._length, self.unit_dz)
        return self._nz

    @property
    def length(self):
        return self._length_func(self.scale, self.nz, self.unit_dz)


class NanowireSizePeriodic(NanowireSize):
    """
    A size information handler for periodic nanowire lattices
    """

    def __init__(self, scale, unit_dz, n_xy=None, nz=None, q=None,
                 width=None, length=None, period=None):
        super().__init__(scale, unit_dz, n_xy, nz, width, length)
        if not (q or period):
            raise ValueError("must specify either `p` or `period`")

        self._q = q
        self._q_func = None

        self._period = period
        self._period_func = None

    @property
    def q(self):
        if self._q is None:
            self._q = self._q_func(self.scale, self._period)
        return self._q

    @property
    def period(self):
        return self._period_func(self.scale, self.q)

    def fix_nz(self, nz):
        self._nz = nz


if __name__ == "__main__":
    size = NanowireSizePeriodic(1.0, unit_dz=0.5, width=15, length=50, q=13)
