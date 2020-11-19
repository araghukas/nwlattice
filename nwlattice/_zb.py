import numpy as np
from nwlattice.utilities import Quaternion

# TODO: clean this up

ROOT3 = np.sqrt(3.)
ROOT2 = np.sqrt(2.)
ROOT6 = ROOT2 * ROOT3

q = Quaternion.rotator([0, 0, 1], np.pi / 6)

"""
Some vectors in units of the cubic lattice constant of a Zincblende crystal
"""

# GaAs lattice constant at 300 K
a0_GaAs = 5.65315  # Angstroms

# second basis atom vector (units of lattice constant)
ZINCBLENDE_BASIS2_TWIN = np.array([0., 0., ROOT3 / 4])

# alternative second basis atom vector (units of lattice constant)
# ZINCBLENDE_BASIS2_HEX = np.array([0.353553, -0.204124, -0.144338])
ZINCBLENDE_BASIS2_HEX = np.array([0., -0.40824829, -0.14433757])
ZINCBLENDE_BASIS2_HEX2 = np.array([0.353553, -0.204124, -0.144338])

# inter-planar spacing between Ga-planes (or As-planes)
PLANE_DZ = np.array([0., 0., 1 / ROOT3])

# xy-plane offset distance get Ga in one plane and Ga in plane above
PLANE_DX = q.rotate(np.array([1 / ROOT6, 0., 0.]))

# nearest neighbor vector
NN_VECTOR = PLANE_DZ + PLANE_DX
