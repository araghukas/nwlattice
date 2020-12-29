import unittest
import inspect
import os
import numpy as np
from nwlattice import base, nw


# from nwlattice import utilities
# utilities.toggle_printing(True)

# ------------------------------------------------------------------------------
class NanowireObjectsTest(unittest.TestCase):
    """convenient base class for every other test in this module"""

    DEFAULT_KWARGS = {
        'scale': 5.65,
        'width': 100.0,
        'length': 250.0,
        'period': 50.0,
        'fraction': 0.5,
        'm0': 0,
        'n_xy': None,
        'nz': None,
        'q': None,
        'force_cyclic': True,
    }

    def setUp(self) -> None:
        self.all_nw_types = self.get_all_simple_nanowire_objects()
        self.argspecs = self.get_argspecs(self.all_nw_types)

    @staticmethod
    def get_all_simple_nanowire_objects():
        obj_list = inspect.getmembers(nw)
        nw_list = []
        for o in obj_list:
            if o[1] is base.ANanowireLattice:
                continue
            try:
                if issubclass(o[1], base.ANanowireLattice):
                    if not issubclass(o[1], base.ACompoundNanowireLattice):
                        nw_list.append(o[1])
            except TypeError:
                pass
        return nw_list

    @staticmethod
    def get_argspecs(obj_list):
        return {
            obj: inspect.getfullargspec(obj) for obj in obj_list
        }

    def get_default_wire(self, t):
        t_args = self.argspecs[t].args[1:]
        kwargs = {k: self.DEFAULT_KWARGS[k] for k in t_args}
        return t(**kwargs)
# ------------------------------------------------------------------------------


class GetPointsTests(NanowireObjectsTest):
    def test_xy_COM_location(self):
        """
        checks that COM is at [0,0,0]
        """
        for t in self.all_nw_types:
            wire = self.get_default_wire(t)
            points = wire.get_points()
            x_mean = 0.
            y_mean = 0.
            n_points = 0
            for pt in points:
                x_mean += pt[0]
                y_mean += pt[1]
                n_points += 1

            x_mean /= n_points
            y_mean /= n_points
            self.assertAlmostEqual(x_mean, 0., delta=0.1, msg=(
                "\n{} COM_x is offset".format(wire.type_name)
            ))
            self.assertAlmostEqual(y_mean, 0., delta=0.1, msg=(
                "\n{} COM_y is offset".format(wire.type_name)
            ))
            print("wire type {} is COM_xy centered ({}, {})"
                  .format(wire.type_name, x_mean, y_mean))


class OutputFilesTest(NanowireObjectsTest):
    data_blank = "./outputs/{}.data"
    map_blank = "./outputs/{}.map"

    def setUp(self) -> None:
        super().setUp()
        self.data_files = {}
        self.map_files = {}

    def test_write_output_files(self):
        """
        checks that map and data can be written (does not check correctness)
        """
        for t in self.all_nw_types:
            wire = self.get_default_wire(t)
            tn = wire.type_name
            wire.write_points(self.data_blank.format(tn))
            wire.write_map(self.map_blank.format(tn))
            print("write successful: {}".format(wire.type_name))

    def tearDown(self) -> None:
        print("Deleting OutputFilesTest outputs...")
        for f in os.listdir("./outputs"):
            if f.endswith(".data") or f.endswith(".map"):
                os.remove("./outputs/" + f)


class InitAnnotationsCompleteTest(NanowireObjectsTest):
    def setUp(self) -> None:
        super().setUp()
        self.correct_annotations = {
            'scale': float,
            'n_xy': int,
            'm_xy': int,
            'nz': int,
            'm0': int,
            'q': int,
            'width': float,
            'length': float,
            'period': float,
            'theta': float,
            'fraction': float,
            'force_cyclic': bool
        }

    def test_initializers_completely_annotated(self):
        """
        checks that all class __init__ signatures are annotated
        """
        for t in self.all_nw_types:
            a = self.argspecs[t]

            # check every arg is annotated
            s1 = set(a.args[1:])
            s2 = set(a.annotations.keys())
            s = s1 - s2
            msg = ("\nfound non-annotated argument(s) {} in type {}"
                   .format(s, t))
            self.assertEqual(s, set(), msg=msg)

            # check every argument is type annotated correctly
            for arg in s1:
                exp = self.correct_annotations[arg]
                act = a.annotations[arg]
                msg = ("\nargument '{}' in {} is annotated as type {}; "
                       "should be {}".format(arg, t, act, exp))
                self.assertEqual(act, exp, msg=msg)


class ClassDocstringsNonEmptyTest(NanowireObjectsTest):
    def test_nonempty_class_docstrings(self):
        """
        checks that there is at least some docstring for every class
        """
        for t in self.all_nw_types:
            self.assertIsNotNone(t.__doc__,
                                 msg="\nclass {} has an empty docstring"
                                 .format(t))
            # print("{}: {}".format(t, t.__doc__))


class ForceCyclicTest(NanowireObjectsTest):
    def setUp(self) -> None:
        super().setUp()
        self.nz_unit_values = {
            nw.FCCPristine111: 3,
            nw.FCCPristine100: 2,
            nw.FCCTwin: -1,
            nw.FCCTwinFaceted: -1,
            nw.HexPristine0001: 2,
            nw.FCCRandomHex: None,
            nw.ZBPristine111: 3,
            nw.DiamondPristine111: 3,
            nw.ZBPristine100: 2,
            nw.DiamondPristine100: 2,
            nw.ZBTwin: -1,
            nw.DiamondTwin: -1,
            nw.ZBTwinFaceted: -1,
            nw.DiamondTwinFaceted: -1,
            nw.WZPristine0001: 2,
            nw.ZBRandomWZ: None,
            nw.DiamondRandomWZ: None
        }

        # check every type has a rule defined above
        s1 = set(self.nz_unit_values.keys())
        s2 = set(self.all_nw_types)
        self.assertSetEqual(s1, s2,
                            msg="no min_z_value for nw type(s) {}"
                            .format(s2 - s1))

    def test_planes_length_is_unit_nz_multiple(self):
        """
        checks that forcing periodic z actually worked
        """
        msg = "wire type {} does not enforce correct periodicity"
        for t in self.all_nw_types:
            wire = self.get_default_wire(t)
            n_unit = self.nz_unit_values[t]
            if n_unit is None:
                print("SKIP: can't force periodicity for nw type {}"
                      .format(t))
            elif n_unit < 0:
                n_unit = 2 * wire.size.q
                self.assertEqual(wire.size.nz % n_unit, 0,
                                 msg=msg.format(t))
            else:
                self.assertEqual(wire.size.nz % n_unit, 0,
                                 msg=msg.format(t))


class TranslationVectorsTest(NanowireObjectsTest):
    def setUp(self):
        self.offset = 10.
        super().setUp()

    def test_add_offset_works_as_expected(self):
        for t in self.all_nw_types:
            wire = self.get_default_wire(t)
            print("checking offset works (for wire type: %s)" % wire.type_name)
            points0 = wire.get_points()
            wire.add_offset(self.offset * np.ones(3))
            points1 = wire.get_points()
            diff = (points1 - points0).flatten()
            msg = "offsets unequal at point "
            for i, x in enumerate(diff):
                self.assertAlmostEqual(x, self.offset, delta=1e-6,
                                       msg=msg + str(i))
            print("writing offset wire: {}".format(wire.type_name))
            wire.write_points("./outputs/{}_offset.data".format(wire.type_name))


@unittest.skip("no need to do this yet")
class MapAndDataAgreementTest(NanowireObjectsTest):
    pass


if __name__ == "__main__":
    # TODO: test init with `period` versus init with `q` give same result
    unittest.main(verbosity=2)
