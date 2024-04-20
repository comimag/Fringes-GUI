import logging
import os
import toml
import importlib

from .gui import FringesGUI

logger = logging.getLogger(__name__)

# use version string in pyproject.toml as the single source of truth
try:
    # in order not to confuse an installed version of a package with a local one,
    # first try the local one (not being installed)
    _meta = toml.load(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"))
    __version__ = _meta["project"]["version"]  # Python Packaging User Guide expects version here
except KeyError:
    __version__ = _meta["tool"]["poetry"]["version"]  # Poetry expects version here
except FileNotFoundError:
    __version__ = importlib.metadata.version("fringes-gui")  # installed version


def run():
    gui = FringesGUI()
    gui.show()
