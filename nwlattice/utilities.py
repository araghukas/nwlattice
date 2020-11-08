import numpy as np

# often-used constants
ROOT2 = 1.4142135623730951
ROOT3 = 1.7320508075688772
ROOT6 = 2.449489742783178


# ------------------------------------------------------------------------------
# rotation tools
# ------------------------------------------------------------------------------
class Quaternion(object):
    """ a basic quaternion object """

    def __init__(self, a, b, c, d):
        self._q = np.array([a, b, c, d], dtype=np.double)
        self._isZeroQuaternion = np.all(self._q == 0)
        self._conj = None

    def __str__(self):
        return "Quaternion({})".format(self.q)

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(*(self.q + other.q))
        elif np.isscalar(other):
            q_new = self.q.copy()
            q_new[0] += other
            return Quaternion(*q_new)
        else:
            raise ValueError("can not add {} to Quaternion".format(type(other)))

    def __neg__(self):
        return Quaternion(*(self.q * -1))

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(*(self.q - other.q))
        elif np.isscalar(other):
            q_new = self.q.copy()
            q_new[0] -= other
            return Quaternion(*q_new)
        else:
            raise ValueError(
                "can not subtract {} from Quaternion".format(type(other)))

    def __mul__(self, other):  # self * other
        if isinstance(other, Quaternion):
            r_new = self.r * other.r - np.dot(self.v, other.v)
            v_new = self.r * other.v + other.r * self.v + np.cross(self.v,
                                                                   other.v)
            return Quaternion.from_rv(r_new, v_new)
        elif np.isscalar(other):
            return Quaternion(*(other * self.q))
        else:
            raise ValueError(
                "can not multiply Quaternion with type {}".format(type(other)))

    def __rmul__(self, other):  # other * self
        if isinstance(other, Quaternion):
            r_new = other.r * self.r - np.dot(other.v, self.v)
            v_new = other.r * self.v + self.r * other.v + np.cross(other.v,
                                                                   self.v)
            return Quaternion.from_rv(r_new, v_new)
        elif np.isscalar(other):
            return self.__mul__(other)  # scalar multiplication commutes
        else:
            raise ValueError(
                "can not multiply Quaternion with type {}".format(type(other)))

    def __truediv__(self, other):  # self / other
        if isinstance(other, Quaternion):
            return self / other
        elif np.isscalar(other):
            return Quaternion(*(self.q / other))
        else:
            raise ValueError(
                "can not divide Quaternion by type {}".format(type(other)))

    def __rtruediv__(self, other):  # other / self
        if self._isZeroQuaternion:
            raise ValueError("division by zero-Quaternion")
        if isinstance(other, Quaternion):
            return other * self.inv
        elif np.isscalar(other):
            return self.__truediv__(other)
        else:
            raise ValueError(
                "can not divide type {} by Quaternion".format(type(other)))

    @classmethod
    def from_rv(cls, r, v):
        if np.isscalar(r) and len(v) == 3:
            return Quaternion(*np.insert(v, 0, r))
        else:
            raise ValueError("r must be a scalar and v must be a 3-vector")

    @classmethod
    def rotator(cls, axis, theta):
        # angle theta in radians!
        axis = np.asarray(axis) / np.linalg.norm(axis)
        axis *= np.sin(theta / 2.)
        r = np.cos(theta / 2.)
        q = Quaternion.from_rv(r, axis)
        return q

    @classmethod
    def point(cls, p):
        p = np.asarray(p)
        return Quaternion.from_rv(0, p)

    @property
    def r(self):
        # real part
        return self._q[0]

    @property
    def v(self):
        # vector part
        return self._q[1:]

    @property
    def q(self):
        return self._q.copy()

    @property
    def inv(self):
        return self.conj / self.mag**2

    @property
    def conj(self):
        if self._conj is None:
            self._conj = Quaternion(*(np.array([1, -1, -1, -1] * self.q)))
        return self._conj

    @property
    def mag(self):
        return np.sqrt(np.sum(self.q**2))

    @property
    def versor(self):
        return self / self.mag

    @property
    def axis(self):
        # unit vector parallel to rotation axis
        return self.q[1:] / np.linalg.norm(self.q[1:])

    def rotate(self, points):
        # rotate point or array of points
        points = np.asarray(points)
        squeeze_out = False
        if points.ndim == 0:
            raise ValueError(
                "quaternion rotations only defined for 3-coordinate points")
        if points.ndim == 1:
            squeeze_out = True
            points = np.expand_dims(points, 0)

        inv = self.inv
        r_qinv = inv.r
        v_qinv = inv.v
        output = np.empty((len(points), 3))
        for i, point in enumerate(points):  # can probably vectorize this...
            r_qp = -np.dot(self.v, point)
            v_qp = self.r * point + np.cross(self.v, point)
            output[i] = r_qp * v_qinv + r_qinv * v_qp + np.cross(v_qp, v_qinv)

        if squeeze_out:
            output.squeeze()
        return output

    @staticmethod
    def qrotate(points, axis, theta):
        q = Quaternion.rotator(axis, theta)
        return q.rotate(points)


def get_tetrahedral_set(v, ortho=None):
    """
    treat input vector as first tetrahedral axis, return remaining 3

    :param v: first 3-vector
    :param ortho: (optional in lieu of default choice) axis for first rotation
    :return: three vectors from tetrahedral rotations of v
    """
    v = np.reshape(v, (3,))
    if ortho is None:
        # default ortho options based on `v`
        if v[0] != 0. and v[1] != 0.:
            ortho_v1 = np.array([-v[1], v[0], 0.])
        elif v[0] != 0. and v[2] != 0.:
            ortho_v1 = np.array([-v[2], 0., v[0]])
        elif v[1] != 0. and v[2] != 0.:
            ortho_v1 = np.array([0., -v[2], v[1]])
        elif v[0] != 0.:
            ortho_v1 = np.array([0., v[0], 0.])
        elif v[1] != 0.:
            ortho_v1 = np.array([0., 0., v[1]])
        elif v[2] != 0.:
            ortho_v1 = np.array([v[2], 0., 0.])
        else:
            raise ValueError("can not rotate zero vector")
    elif abs(np.dot(v, ortho)) <= np.finfo(np.float64).eps:
        ortho_v1 = ortho
    else:
        angle = np.arccos(
            np.dot(v, ortho) / np.linalg.norm(v) * np.linalg.norm(ortho)
        ) * 180 / np.pi
        raise ValueError("argument vectors are not orthogonal, "
                         "angle = %f degrees" % angle)

    # perform rotations
    theta = 1.9106332362490184  # ~109.4˚ in rad
    q1 = Quaternion.rotator(ortho_v1, theta)
    q2 = Quaternion.rotator(v, np.pi / 3)
    v2 = q1.rotate(v)[0]
    v3 = q2.rotate(v2)[0]
    v4 = q2.rotate(v2)[0]
    return v2, v3, v4
