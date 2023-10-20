import importlib
import os

import toml

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # https://github.com/opencv/opencv/issues/21326

from .gui import FringesGUI

try:  # PackageNotFoundError
    fname = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    version = toml.load(fname)["tool"]["poetry"]["version"]
except FileNotFoundError or KeyError:
    version = importlib.metadata.version("fringes_gui")

__version__ = version


def run():
    gui = FringesGUI()
    gui.show()
