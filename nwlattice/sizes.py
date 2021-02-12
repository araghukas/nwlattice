SIZE_PROPERTIES = [
    "scale",
    "width",
    "length",
    "unit_dz",
    "period",
    "fraction",
    "area",
    "n_xy",
    "nz",
    "q",
    "indexer"
]


class NanowireSizeCompound:
    """
    A size container that combines one or more NanowireSize objects
    """

    def __init__(self, **kwargs):
        for k in kwargs:
            if k in SIZE_PROPERTIES:
                self.__setattr__(k, kwargs[k])

    def __str__(self):
        s = "<NanowireSize instance>\n"
        s_args = []
        props = self.props()
        for prop, val in props.items():
            try:
                if int(val) == val:
                    s_args.append("<\t{:<10}: {:<15,d}>".format(prop, val))
                else:
                    s_args.append("<\t{:<10}: {:<15,.2f}>".format(prop, val))
            except TypeError:
                s_args.append("<\t{:<10}: {}>".format(prop, val))

        s += "\n".join(s_args)
        return s

    def props(self):
        p_dict = {}
        for prop in SIZE_PROPERTIES:
            if hasattr(self, prop):
                p_dict[prop] = self.__getattribute__(prop)
        return p_dict


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
            raise ValueError("must specify either `n_xy` or `width`")

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
        s = "<NanowireSize instance>\n"
        s_args = []
        props = self.props()
        for prop, val in props.items():
            try:
                if int(val) == val:
                    s_args.append("<\t{:<10}: {:<15,d}>".format(prop, val))
                else:
                    s_args.append("<\t{:<10}: {:<15,.2f}>".format(prop, val))
            except TypeError:
                s_args.append("<\t{:<10}: {}>".format(prop, val))

        s += "\n".join(s_args)
        return s

    def props(self):
        p_dict = {}
        for prop in SIZE_PROPERTIES:
            if hasattr(self, prop):
                p_dict[prop] = self.__getattribute__(prop)
        return p_dict

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

    def fix_nz(self, nz):
        self._nz = nz


class NanowireSizeRandom(NanowireSize):
    def __init__(self, scale, unit_dz, fraction, n_xy=None, nz=None,
                 width=None, length=None):
        super().__init__(scale, unit_dz, n_xy, nz, width, length)
        self._fraction = fraction

    @property
    def fraction(self):
        return self._fraction


class NanowireSizePeriodic(NanowireSize):
    """
    A size information handler for periodic nanowire lattices
    """

    def __init__(self, scale, unit_dz, n_xy=None, nz=None, q=None,
                 width=None, length=None, period=None):
        super().__init__(scale, unit_dz, n_xy, nz, width, length)
        if not (q or period):
            raise ValueError("must specify either `q` or `period`")

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


class NanowireSizeArbitrary(NanowireSize):
    """
    A size information handler for arbitrary nanowire lattices
    """

    def __init__(self, scale, unit_dz, n_xy=None, nz=None,
                 width=None, length=None):
        super().__init__(scale, unit_dz, n_xy, nz, width, length)
        self._index = None
        self._indexer = None

    @property
    def index(self):
        if self._index is None:
            new_nz, self._index = self._indexer(self.nz)
            if new_nz:  # option to bypass forcing nz change
                self._nz = new_nz
        return self._index

    @property
    def indexer(self):
        return self._indexer
