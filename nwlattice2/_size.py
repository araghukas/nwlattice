class PlaneSize(object):
    """
    A size information handler for planar lattices
    """

    @property
    def n_xy(self):
        if self._n_xy is None:
            self._n_xy = self._n_xy_func(self._width)
        return self._n_xy

    @property
    def width(self):
        return self._width_func(self.n_xy)

    @property
    def area(self):
        return self._area_func(self.n_xy)

    def __init__(self, scale, n_xy=None, width=None):
        """
        :param scale: lattice scale (ex: lattice constant)
        :param n_xy: structure specific integer thickness index indicating width
        :param width: width in sase units as `a0`
        """
        if not (width or n_xy):
            raise ValueError("must specify wither `n_xy` or `width`")
        self._scale = scale
        self._n_xy = n_xy
        self._width = width

        # size calculator functions
        self._n_xy_func = None
        self._width_func = None
        self._area_func = None


class NWSize(PlaneSize):
    """
    A size information handler for nanowire lattices
    """

    @property
    def nz(self):
        if self._nz is None:
            self._nz = self._nz_func(self._length)
        return self._nz

    @property
    def length(self):
        return self._length_func(self.nz)

    def __init__(self, scale, nz=None, n_xy=None,
                 length=None, width=None):
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
        self._nz = nz
        self._length = length

        # size calculator functions
        self._nz_func = None
        self._length_func = None


class NWSizePeriodic(NWSize):
    """
    A size information handler for periodic nanowire lattices
    """

    @property
    def p(self):
        if self._p is None:
            self._p = self._p_func(self._period)
        return self._p

    @property
    def period(self):
        return self._period_func(self.p)

    def __init__(self, scale, nz=None, n_xy=None, p=None,
                 length=None, width=None, period=None):
        super().__init__(scale, nz, n_xy, length, width)
        if not (p or period):
            raise ValueError("must specify either `p` or `period`")

        self._p = p
        self._period = period

        # size calculator functions
        self._p_func = None
        self._period_func = None
