import unittest
import inspect
import os
from random import randint, uniform
from nwlattice import planes, base


class PlaneObjectsTest(unittest.TestCase):
    """convenient base class for every other test in this module"""

    def setUp(self) -> None:
        self.plane_types = self.get_all_plane_objects()
        self.argspecs = self.get_argspecs(self.plane_types)

    @staticmethod
    def get_all_plane_objects():
        obj_list = inspect.getmembers(planes)
        planes_list = []
        for o in obj_list:
            if o[1] is base.APointPlane:
                continue
            try:
                if issubclass(o[1], base.APointPlane):
                    planes_list.append(o[1])
            except TypeError:
                pass
        return planes_list

    @staticmethod
    def get_argspecs(obj_list):
        return {
            obj: inspect.getfullargspec(obj) for obj in obj_list
        }


class AnnotationsCompleteTest(PlaneObjectsTest):
    """checks for correct and complete annotations in initializers"""

    def setUp(self) -> None:
        super().setUp()
        self.correct_annotations = {
            'scale': float,
            'n_xy': int,
            'm_xy': int,
            'width': float,
            'theta': float
        }

    def test_initializers_completely_annotated(self):
        for t in self.plane_types:
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


class ExpectedErrorsTest(PlaneObjectsTest):
    """test random (incorrect) values of arguments to check error thrown"""

    def setUp(self) -> None:
        super().setUp()
        self.default_kwargs = {
            'scale': 1.0,
            'n_xy': None,
            'm_xy': 0,
            'width': 50.0,
            'theta': None
        }

    def _error_values_of(self, arg_name, *values):
        for t in self.plane_types:
            args = self.argspecs[t].args[1:]
            kwargs = {k: self.default_kwargs[k] for k in args if k != arg_name}
            for v in values:
                msg = ("\nplane type {} should raise ValueError when passed "
                       "`{} = {}`".format(t, arg_name, v))
                if arg_name in args:
                    with self.assertRaises(ValueError, msg=msg):
                        kwargs[arg_name] = v
                        t(**kwargs)

    def test_non_positive_scale_errors(self):
        self._error_values_of('scale', 0, randint(-10, 0), uniform(-10, 0))

    def test_non_positive_n_xy_errors(self):
        self._error_values_of('n_xy', 0, randint(-10, 0), uniform(-10, 0))

    def test_non_positive_width_errors(self):
        self._error_values_of('width', 0, randint(-10, 0), uniform(-10, 0))

    def test_non_positive_m_xy_errors(self):
        self._error_values_of('m_xy', randint(-10, -1), uniform(-10, -0.1))


class PlaneDimensionsTest(PlaneObjectsTest):
    """check that planes initialize to expected dimensions"""

    def setUp(self) -> None:
        super().setUp()
        self.scale = 1.0

    def test_input_width_against_real_width(self):
        for t in self.plane_types:
            width = uniform(10., 40.)
            p = t(scale=self.scale, width=width)
            msg = ("\nsize of plane type {} initialized with `width = {}` has "
                   "`size.width = {}`".format(t, width, p.size.width))
            self.assertAlmostEqual(width, p.size.width, delta=self.scale,
                                   msg=msg)

    def test_n_xy_computed_correctly_in_size_obj(self):
        for t in self.plane_types:
            width = uniform(10., 40.)
            n_xy = t.get_n_xy(self.scale, width)
            p = t(scale=self.scale, width=width)
            self.assertEqual(n_xy, p.size.n_xy)

    def test_TwFCC_widths_same_all_m_xy(self):
        width = uniform(10., 40.)
        n_xy = planes.TwFCC.get_n_xy(self.scale, width)
        resulting_widths = []
        for m_xy in range(n_xy):
            p = planes.TwFCC(scale=self.scale, width=width, m_xy=m_xy)
            msg = ("\nTwFCC argument width {} differs from size width {} "
                   "by more than `scale = {}` when `m_xy = {}`"
                   .format(width, p.size.width, self.scale, m_xy))
            self.assertAlmostEqual(width, p.size.width, msg=msg,
                                   delta=self.scale)
            resulting_widths.append(p.size.width)

        w0 = resulting_widths[0]
        self.assertTrue(all([w == w0 for w in resulting_widths]),
                        msg="found unequal widths among valid `m_xy` for TwFCC")


class SmallWidthValuesTest(PlaneObjectsTest):
    def setUp(self) -> None:
        super().setUp()
        self.scale = 1.0
        self.written_files = []

    def test_small_width_writing(self):
        for t in self.plane_types:
            width = uniform(0.01, self.scale)
            p = t(scale=self.scale, width=width)
            file_path = "./{}._data".format(p.type_name)
            p.write_points(file_path)
            self.written_files.append(file_path)
            n_atoms = self.count_written_atoms(file_path)
            print("plane type {} writes {} atoms when n_xy = {}"
                  .format(t, n_atoms, p.size.n_xy))

    def tearDown(self):
        for file_ in self.written_files:
            os.remove(file_)

    @staticmethod
    def count_written_atoms(file_path):
        with open(file_path) as file_:
            n_atoms = 0

            # move down lines unitl atoms section
            line = file_.readline()
            while not line.startswith("Atoms # atomic"):
                line = file_.readline()

            line = file_.readline()
            while line:
                if not line.strip() == "":
                    n_atoms += 1
                line = file_.readline()

            return n_atoms


class WidthInvariantWithScaleTest(PlaneObjectsTest):
    def test_width_independent_of_scale(self):
        width = 100.
        msg = "\nplane type {} widths too far apart for scales {} and {}"
        for t in self.plane_types:
            width_prev = None
            scale_prev = None
            for _ in range(100):
                scale = uniform(1., 5.)
                p = t(scale=scale, width=width)
                if width_prev is None:
                    width_prev = p.size.width
                    scale_prev = scale
                else:
                    delta = max(scale, scale_prev)
                    # will fail with `delta = average(scale, scale_prev)`
                    # will fail with `delta = min(scale, scale_prev)`
                    self.assertAlmostEqual(width, width_prev, delta=delta,
                                           msg=msg.format(t, scale, scale_prev))


if __name__ == '__main__':
    unittest.main(verbosity=2)
