from abc import ABC, abstractmethod


class IDataWriter(ABC):
    """
    The interface that writes the LAMMPS/phana atom data and map files
    """
    WILL_PRINT = False

    @abstractmethod
    def write_points(self, file_path: str):
        raise NotImplementedError

    @abstractmethod
    def write_map(self, file_path: str):
        raise NotImplementedError

    @staticmethod
    def print(s):
        if IDataWriter.WILL_PRINT:
            print(s)
