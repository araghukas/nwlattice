import unittest
import inspect
from nwlattice2 import base, nw


class NanowireObjectsTest(unittest.TestCase):
    """convenient base class for every other test in this module"""
    def setUp(self) -> None:
        self.nw_types = self.get_all_nanowire_objects()
        self.argspecs = self.get_argspecs(self.nw_types)

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
            'force_cyclic': bool
        }

    def test_initializers_completely_annotated(self):
        for t in self.nw_types:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)