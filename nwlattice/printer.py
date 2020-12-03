from abc import ABC


class ABCPrinter(ABC):
    PRINT = False

    @staticmethod
    def print(s):
        if ABCPrinter.PRINT:
            print(s)


def toggle_printing(b: bool):
    """
    Toggle global printing of messages by all planes and stacks

    :param b: True or False, indicating messages printed or not
    :return:
    """
    ABCPrinter.PRINT = b
