import unittest
import inspect
from nwlattice import base, nw

# from nwlattice import utilities
# utilities.toggle_printing(True)


class NanowireObjectsTest(unittest.TestCase):
    """convenient base class for every other test in this module"""

    def setUp(self) -> None:
        self.all_nw_types = self.get_all_nanowire_objects()
        self.argspecs = self.get_argspecs(self.all_nw_types)
        self.default_kwargs = {
            'scale': 5.65,
            'n_xy': None,
            'nz': None,
            'q': None,
            'width': 100.0,
            'period': 50.0,
            'length': 250.0,
            'force_cyclic': True,
            'hex_fraction': 0.5,
            'wz_fraction': 0.5
        }

    @staticmethod
    def get_all_nanowire_objects():
        obj_list = inspect.getmembers(nw)
        nw_list = []
        for o in obj_list:
            if o[1] is base.ANanowireLattice:
                continue
            try:
                if issubclass(o[1], base.ANanowireLattice):
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
        kwargs = {k: self.default_kwargs[k] for k in t_args}
        return t(**kwargs)


class OutputFilesTest(NanowireObjectsTest):
    data_blank = "./outputs/{}.data"
    map_blank = "./outputs/{}.map"

    # TODO: write test to compare map file types against data types

    def setUp(self) -> None:
        super().setUp()
        self.data_files = {}
        self.map_files = {}

    def test_write_output_files(self):
        for t in self.all_nw_types:
            wire = self.get_default_wire(t)
            tn = wire.type_name
            wire.write_points(self.data_blank.format(tn))
            wire.write_map(self.map_blank.format(tn))

        # TODO: DiamondPristine111 ends up with width ~50
        # TODO: DiamondRandomWZ width ~69
        # TODO: DiamondTwin width ~69
        # TODO: DiamondTwinFaceted width ~69
        # TODO: FCCPristine111 width ~50
        # TODO: FCCRandomHex width ~ 69
        # TODO: FCCTwin width ~ 69
        # TODO: THEY'RE ALL OUT OF WHACK EXCEPT THE '100' ONES


class AnnotationsCompleteTest(NanowireObjectsTest):
    def setUp(self) -> None:
        super().setUp()
        self.correct_annotations = {
            'scale': float,
            'n_xy': int,
            'm_xy': int,
            'nz': int,
            'q': int,
            'width': float,
            'length': float,
            'period': float,
            'theta': float,
            'wz_fraction': float,
            'hex_fraction': float,
            'force_cyclic': bool
        }

    def test_initializers_completely_annotated(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
