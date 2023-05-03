import ctypes
import os
import logging as lg
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit
from pyqtgraph.Qt import QtGui, QtWidgets
from pyqtgraph.dockarea import *
import numpy as np
import toml
import fringes as frng

from fringes_gui.setters import set_functionality

# import pyqtgraph.examples
# pyqtgraph.examples.run()


class FringesGUI(QApplication):
    """Simple graphical user interface for the 'fringes' package."""

    def __init__(self):
        super(FringesGUI, self).__init__([])

        fname = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        version = toml.load(fname)["tool"]["poetry"]["version"]
        myappid = "Fringes-GUI" + " " + version  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        pg.setConfigOptions(imageAxisOrder="row-major", useNumba=True)  # useCupy

        self.fringes = frng.Fringes(X=1920, Y=1200)
        self.fringes.logger.setLevel("INFO")
        self.fringes.load(os.path.join(os.path.expanduser("~"), ".fringes.yaml"))
        self.key = ""
        self.visibility = "Expert"
        self.digits = 8  # todo: len(str(self.fringes._Pmax))  # 4 (digits) + 1 (point) + 3 (decimals) = 8 == current length of Pmax?
        self.sub = str.maketrans("1234567890", "₁₂₃₄₅₆₇₈₉₀")
        self.sup = str.maketrans("₁₂₃₄₅₆₇₈₉₀", "1234567890")

        # define window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)

        self.win.setGeometry(QtGui.QGuiApplication.primaryScreen().availableGeometry())  # move to primary screen
        # if Screen.count == 1 and self.desktop.availableGeometry(idx).width() / self.desktop.availableGeometry(idx).height() >= 21 / 9:
        #     self.win.resize(self.desktop.availableGeometry().width() // 2, self.desktop.availableGeometry().height())
        #     # self.win.move(0, 0)  # move to the left
        #     self.win.move(self.desktop.availableGeometry().width() // 2, 0)  # move to the right
        # else:
        #     self.win.showMaximized()  # self.win.showFullScreen()

        self.win.showMaximized()  # self.win.showFullScreen()

        self.win.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), "spirals.png")))

        self.win.setWindowTitle(myappid)

        # set style
        # self.setStyleSheet(open("QTDark.stylesheet").read())

        # import qdarkstyle
        # self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))

        # styles = QtWidgets.QStyleFactory.keys()  # system styles
        # self.setStyle('Fusion')

        # import qtmodern.styles
        # import qtmodern.windows
        # qtmodern.styles.dark(self)
        # self.win = qtmodern.windows.ModernWindow(self.win)  # todo: moving window with win+arrow keys doesn't work
        # pg.setConfigOptions(background=42/255)

        # Create docks, place them into the window one at a time.
        from pyqtgraph.dockarea.Dock import DockLabel

        def updateStylePatched(self):  # from https://gist.github.com/matmr/72487a03da95b99db6ae
            r = '3px'
            if self.dim:
                fg = '#fff'  # gray
                bg = "##6CC0A8"  # green brighter
                border = "#6CC0A8"  # green brighter
            else:
                fg = '#fff'  # white
                bg = "#169D7C"  # green
                border = "#169D7C"  # green

            if self.orientation == 'vertical':
                self.vStyle = """DockLabel {
                    background-color : %s;
                    color : %s;
                    border-top-right-radius: 0px;
                    border-top-left-radius: %s;
                    border-bottom-right-radius: 0px;
                    border-bottom-left-radius: %s;
                    border-width: 0px;
                    border-right: 2px solid %s;
                    padding-top: 3px;
                    padding-bottom: 3px;
                }""" % (bg, fg, r, r, border)
                self.setStyleSheet(self.vStyle)
            else:
                self.hStyle = """DockLabel {
                    background-color : %s;
                    color : %s;
                    border-top-right-radius: %s;
                    border-top-left-radius: %s;
                    border-bottom-right-radius: 0px;
                    border-bottom-left-radius: 0px;
                    border-width: 0px;
                    border-bottom: 2px solid %s;
                }""" % (bg, fg, r, r, border)
                self.setStyleSheet(self.hStyle)

        DockLabel.updateStyle = updateStylePatched

        self.dock_attributes = Dock("Attributes", size=(15, 99))
        self.dock_methods = Dock("Methods", size=(15, 1))
        self.dock_viewer = Dock("Viewer", size=(70, 100))
        self.dock_data = Dock("Data", size=(15, 30))
        self.dock_log = Dock("Log", size=(15, 70))

        self.area.addDock(self.dock_attributes, "left")
        self.area.addDock(self.dock_viewer, "right", self.dock_attributes)
        self.area.addDock(self.dock_methods, "bottom", self.dock_attributes)
        self.area.addDock(self.dock_data, "right", self.dock_viewer)
        self.area.addDock(self.dock_log, "bottom", self.dock_data)

        # Add widgets into each dock.

        # General settings
        self.immerse = False
        self.immerse_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+i"), self.win)

        self.tree = pg.parametertree.ParameterTree()
        self.dock_attributes.addWidget(self.tree)

        # Control
        self.undo_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self.win)
        self.redo_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self.win)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setEnabled(self.resetOK)
        self.reset_button.setToolTip("Press 'Ctrl+R'.")
        self.reset_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self.win)
        self.encode_checkbox = QtWidgets.QCheckBox()
        self.encode_label = QtWidgets.QLabel("      Encode on parameter change")
        self.decode_checkbox = QtWidgets.QCheckBox()
        self.decode_label = QtWidgets.QLabel("      Decode on parameter change")
        self.default_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+D"), self.win)
        self.coordinates_key = QtGui.QShortcut(QtGui.QKeySequence("G"), self.win)
        self.encode_button = QtWidgets.QPushButton("Encode")
        self.encode_button.setToolTip("Press 'E'.")
        self.encode_button.setStyleSheet("" if self.encodeOK else "QPushButton{color: red}")
        self.encode_key = QtGui.QShortcut(QtGui.QKeySequence("E"), self.win)
        self.decode_button = QtWidgets.QPushButton("Decode")
        self.decode_button.setEnabled(False)
        self.decode_button.setToolTip("Press 'D'.")
        self.decode_key = QtGui.QShortcut(QtGui.QKeySequence("D"), self.win)
        self.decode_key.setEnabled(False)
        self.remap_button = QtWidgets.QPushButton("Remap")
        self.remap_button.setEnabled(False)
        self.remap_button.setToolTip("Press 'R'.")
        self.remap_key = QtGui.QShortcut(QtGui.QKeySequence("R"), self.win)
        self.remap_key.setEnabled(False)
        self.curvature_button = QtWidgets.QPushButton("Curvature")
        self.curvature_button.setEnabled(False)
        self.curvature_button.setToolTip("Press 'C'.")
        self.curvature_key = QtGui.QShortcut(QtGui.QKeySequence("C"), self.win)
        self.curvature_key.setEnabled(False)
        self.height_button = QtWidgets.QPushButton("Height")
        self.height_button.setEnabled(False)
        self.height_button.setToolTip("Press 'H'.")
        self.height_key = QtGui.QShortcut(QtGui.QKeySequence("H"), self.win)
        self.height_key.setEnabled(False)
        self.dock_methods.addWidget(self.reset_button, 0, 0, 1, 2)
        # self.dock_methods.addWidget(self.encode_label, 1, 0, 1, 2)
        # self.dock_methods.addWidget(self.encode_checkbox, 1, 0, 1, 2)
        # self.dock_methods.addWidget(self.decode_label, 2, 0, 1, 2)
        # self.dock_methods.addWidget(self.decode_checkbox, 2, 0, 1, 2)
        self.dock_methods.addWidget(self.encode_button, 3, 0)
        self.dock_methods.addWidget(self.decode_button, 3, 1)
        self.dock_methods.addWidget(self.remap_button, 4, 0, 1, 2)
        self.dock_methods.addWidget(self.curvature_button, 5, 0)
        self.dock_methods.addWidget(self.height_button, 5, 1)

        # Viewer
        self.zoomback_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), self.win)  # todo

        self.plot = pg.PlotItem()
        self.plot.setLabel(axis='left', text='y-axis')
        self.plot.setLabel(axis='bottom', text='x-axis')  # todo: set label 'T-axis'
        self.imv = pg.ImageView(view=self.plot)
        # self.imv.ui.histogram.hide()
        # self.imv.ui.roiBtn.hide()
        # self.imv.ui.menuBtn.hide()
        self.dock_viewer.addWidget(self.imv)

        self.info = QtWidgets.QLabel()
        self.dock_viewer.addWidget(self.info)

        # Data
        # self.data_tree = pg.DataTreeWidget()

        class TableView(QtWidgets.QTableWidget):
            def __init__(self, *args):
                QtWidgets.QTableWidget.__init__(self, *args)
                self.resizeRowsToContents()
                self.resizeColumnsToContents()
                # self.setSelectionBehavior(QtWidgets.QTableView.selectRow)
                # self.setSelectionMode(QtWidgets.QTableView.SingleSelection)
                self.setColumnCount(3)
                self.setRowCount(0)
                self.setHorizontalHeaderLabels(["Name", "Video-Shape", "Type"])

            def setData(self, data={}):
                self.setRowCount(len(data))
                for i, row in enumerate(sorted(data)):
                    for j, v in enumerate(row):
                        newitem = QtWidgets.QTableWidgetItem(v)
                        self.setItem(i, j, newitem)

                # self.resizeRowsToContents()
                self.resizeColumnsToContents()

        # self.display_selector = QtWidgets.QComboBox()
        # self.display_selector.setPlaceholderText("Nothing")
        # self.con = Container(self.display_selector)
        class Container:
            @property
            def info(self):
                info = [[str(k), str(v.shape), str(v.dtype)] for k, v in self.__dict__.items() if
                        isinstance(v, np.ndarray)]
                return info

        self.con = Container()
        # self.dock_data.addWidget(self.display_selector, 0, 0)

        self.data_table = TableView()
        self.dock_data.addWidget(self.data_table, 1, 0, 1, 2)
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.setToolTip("Press 'Ctrl+L'.")
        self.load_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self.win)
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setToolTip("Press 'Ctrl+S'.")
        self.save_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self.win)
        self.set_button = QtWidgets.QPushButton("Set data (to be decoded)")
        self.set_button.setToolTip("Press 'Ctrl+Shift+S'.")
        self.set_button.setEnabled(False)
        self.set_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+S"), self.win)
        self.clear_button = QtWidgets.QPushButton("Clear all")
        self.clear_button.setEnabled(self.dataOK)
        self.clear_button.setToolTip("Press 'Ctrl+Shift+C'.")
        self.clear_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+C"), self.win)
        self.clear_key.setEnabled(self.dataOK)
        self.dock_data.addWidget(self.load_button, 2, 0)
        self.dock_data.addWidget(self.save_button, 2, 1)
        self.dock_data.addWidget(self.clear_button, 3, 0)
        self.dock_data.addWidget(self.set_button, 3, 1)

        # self.model = QtWidgets.QFileSystemModel()
        # self.model.setRootPath(self.data.dest)
        # self.view = QtWidgets.QTreeView()
        # self.view.setModel(self.model)
        # self.view.setCurrentIndex(self.model.index(self.data.dest))
        # self.view.setExpanded(self.model.index(self.data.dest), True)
        # self.dock_data.addWidget(self.view, row=3, col=0)

        # Log
        class QPlainTextEditLogger(lg.Handler):
            def emit(self, record):
                self.widget.appendPlainText(self.format(record))

        self.log_widget = QPlainTextEdit()
        self.dock_log.addWidget(self.log_widget)
        handler = QPlainTextEditLogger()
        handler.setFormatter(self.fringes.logger.handlers[0].formatter)
        handler.widget = self.log_widget
        self.fringes.logger.addHandler(handler)

        # todo: reset button or simply restart? save current config on shutdown

        set_functionality(self)

        self.show()
        # self.fringes.logger.info(f"Started {myappid}.")

    def show(self):
        """Display the Application."""
        pg.exec()
        # if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        #     QtWidgets.QApplication.instance().exec_()

    @property
    def SDMOK(self):
        a = self.fringes.D == 2
        b = self.fringes.grid in self.fringes._grids[:2]
        c = not self.fringes.FDM
        return a and b and c

    @property
    def WDMOK(self):
        a = self.fringes._ismono
        b = np.all(self.fringes.N == 3)
        e = not self.fringes.FDM
        return a and b and e

    @property
    def FDMOK(self):
        a = self.fringes.D > 1 or self.fringes.K > 1
        b = not self.fringes.SDM
        c = not self.fringes.WDM
        return a and b and c

    @property
    def muxtxt(self):
        return "Single Shot" if self.fringes.T == 1 else "Crossed Fringes" if self.fringes.D == 2 and (self.fringes.SDM or self.fringes.FDM) else ""

    @property
    def resetOK(self):
        """True if params equal defalts."""
        return self.fringes.params != frng.Fringes().params

    @property
    def encodeOK(self):
        """True if unambiguous measurement range >= length."""
        return np.all(self.fringes.UMR >= self.fringes.L)

    @property
    def dataOK(self):
        """Return True if at least one attribute of Container class is an ndarray object."""
        return any(isinstance(obj, np.ndarray) for obj in self.con.__dict__.values())

    @property
    def set_dataOK(self):
        """True if data to be decoded can be set."""
        flist = [v.shape[0] for k, v in self.con.__dict__.items() if isinstance(v, np.ndarray)]
        return any(frames == self.fringes.T for frames in flist)

    @property
    def decodeOK(self):
        """Return true if data present can be decoded."""
        I = (
            getattr(self.con, self.key) if hasattr(self.con, self.key)
            # else self.con.raw if hasattr(self.con, "raw")
            else self.con.fringes if hasattr(self.con, "fringes")
            else None
        )
        return I is not None and hasattr(I, "ndim") and frng.vshape(I).shape[0] == self.fringes.T

    @property
    def remapOK(self):
        """Returns True if modulation and registration is available."""
        return hasattr(self.con, "registration")  # hasattr(self.con, "modulation")

    @property
    def curvatureOK(self):
        return hasattr(self.con, "registration") and self.con.registration.shape[1] >= 2 and self.con.registration.shape[2] >= 2

    @property
    def heightOK(self):
        return hasattr(self.con, "curvature")

    @property
    def adv_vis(self):  # todo: remove
        """Returns True if the user has at least Expert level access i.e. advanced visibility."""
        return self.visibility in ["Expert", "Guru", "Experimental"]
