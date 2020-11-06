import numpy as np


def qrotate(points, axis, theta):
    q = Quaternion.rotator(axis, theta)
    return q.rotate(points)


class Quaternion(object):
    """ a basic quaternion object """

    def __init__(self, a, b, c, d):
        self._q = np.array([a, b, c, d], dtype=np.double)
        self._isZeroQuaternion = np.all(self._q == 0)

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
        return Quaternion(*(np.array([1, -1, -1, -1] * self.q)))

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
        squeezeOut = False
        if points.ndim == 0:
            raise ValueError(
                "quaternion rotations only defined for 3-coordinate points")
        if points.ndim == 1:
            squeezeOut = True
            points = np.expand_dims(points, 0)

        inv = self.inv
        r_qinv = inv.r
        v_qinv = inv.v
        output = np.empty((len(points), 3))
        for i, point in enumerate(points):  # can probably vectorize this...

            r_qp = -np.dot(self.v, point)
            v_qp = self.r * point + np.cross(self.v, point)
            output[i] = r_qp * v_qinv + r_qinv * v_qp + np.cross(v_qp, v_qinv)

        if squeezeOut:
            output.squeeze()
        return output
