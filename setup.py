from setuptools import setup, find_packages
from nwlattice import __version__

setup(
    name="nwlattice",
    version=__version__,
    description="A package for generating atom positions in nanowire "
                "structures",
    author="Ara Ghukasyan",
    install_requires=['numpy'],
    packages=find_packages()
)
