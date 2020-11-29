from nwlattice.stacks import *


def toggle_printing(b: bool):
    """
    Toggle global printing of messages by all planes and stacks

    :param b: True or False, indicating messages printed or not
    :return:
    """
    from nwlattice.base import APointPlane, AStackLattice
    APointPlane.PRINT = bool(b)
    AStackLattice.PRINT = bool(b)


__version__ = "24Nov2020"
