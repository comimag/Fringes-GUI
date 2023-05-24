import importlib
import os

import toml

from .gui import FringesGUI

try:
    fname = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    version = toml.load(fname)["tool"]["poetry"]["version"]
except FileNotFoundError or KeyError:
    version = importlib.metadata.version("fringes_gui")

__version__ = version


def run():
    gui = FringesGUI()
    gui.show()
