#!/usr/bin/env python3
from nwlattice import (ZBTwinFaceted,
                       WZPristine0001,
                       FCCPristine100,
                       ZBWZMixed)


# GaAs conventional cell lattice constant at 300 K, for example
a0_GaAs = 5.65315  # [Å]


# The easiest way to initialize the nanowire objects is like below
# This will create a lattice using the same units as `a0`
nw_zb = ZBTwinFaceted.from_dimensions(a0=a0_GaAs, diameter=100, length=500,
                                      period=75, z_periodic=True)

# Since the lattice is discrete, it will approximate the measurements above
# The resulting measurements are stored as properties of the instance
print()
print("`nw_zb` measurements")
print("--------------------")
print("diameter:", nw_zb.D)
print("length:", nw_zb.L)
print("period:", nw_zb.P)
print("cross sectional area:", nw_zb.area)
print()

# to save the points call the `write_points` method with a path for the file
nw_zb.write_points("./zb_twin_faceted_test.data")


# Here's a couple others:
# All the lattice classes implement similar interfaces
nw_wz = WZPristine0001.from_dimensions(a0=a0_GaAs, diameter=100, length=500)
nw_wz.write_points("./wz_pristine_test.data")

nw_fcc = FCCPristine100.from_dimensions(a0_GaAs, 100, 500)
nw_fcc.write_points("./fcc_pristine_test.data")

# the ZBWZMixed class creates a ZB nanowire with WZ segments inserted throughout
# in this example, we get a mixed-phase wire that's ~45% WZ
nw_zbwz = ZBWZMixed.from_dimensions(a0_GaAs, 100, 500, frac=0.45)
nw_zbwz.write_points("./zbwz_mixed_test.data")
