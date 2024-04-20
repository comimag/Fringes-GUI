import numpy as np
import keyboard

import run

# self.undo_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self.win)
#         self.redo_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self.win)

# change param
def test_reset():
    run()

    key R
    assert empty con

self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setEnabled(self.resetOK)
        self.reset_button.setToolTip("Press 'Ctrl+R'.")
        self.reset_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self.win)
        self.encode_checkbox = QtWidgets.QCheckBox()
        self.encode_label = QtWidgets.QLabel("      Encode on parameter change")
        self.decode_checkbox = QtWidgets.QCheckBox()
        self.decode_label = QtWidgets.QLabel("      Decode on parameter change")
        self.default_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+D"), self.win)
        self.coordinates_key = QtGui.QShortcut(QtGui.QKeySequence("X"), self.win)
        self.encode_button = QtWidgets.QPushButton("Encode")
        self.encode_button.setToolTip("Press 'E'.")
        self.encode_button.setStyleSheet("" if not self.fringes._ambiguous else "QPushButton{color: red}")
        self.encode_key = QtGui.QShortcut(QtGui.QKeySequence("E"), self.win)
        self.decode_button = QtWidgets.QPushButton("Decode")
        self.decode_button.setEnabled(False)
        self.decode_button.setToolTip("Press 'D'.")
        self.decode_key = QtGui.QShortcut(QtGui.QKeySequence("D"), self.win)
        self.decode_key.setEnabled(False)
        self.register_key = QtGui.QShortcut(QtGui.QKeySequence("R"), self.win)
        self.source_button = QtWidgets.QPushButton("Source")
        self.source_button.setEnabled(False)
        self.source_button.setToolTip("Press 'S'.")
        self.source_key = QtGui.QShortcut(QtGui.QKeySequence("S"), self.win)
        self.source_key.setEnabled(False)
        self.bright_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+B"), self.win)
        self.bright_inverse_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+B"), self.win)
        self.dark_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self.win)
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
        self.dock_methods.addWidget(self.source_button, 4, 0, 1, 2)
        self.dock_methods.addWidget(self.curvature_button, 5, 0)
        self.dock_methods.addWidget(self.height_button, 5, 1)

        # Viewer
        self.zoomback_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), self.win)

        self.plot = pg.PlotItem()
        self.plot.setLabel(axis="left", text="y-axis")
        self.plot.setLabel(axis="bottom", text="x-axis")  # todo: set label 'T-axis'
        # todo: plot.setTitle according to datum
        self.imv = pg.ImageView(view=self.plot)
        # self.imv.ui.histogram.hide()
        # self.imv.ui.roiBtn.hide()
        # self.imv.ui.menuBtn.hide()
        self.dock_viewer.addWidget(self.imv)

        # Data
        # self.data_tree = pg.DataTreeWidget()

        class TableView(QtWidgets.QTableWidget):
            def __init__(self, *args):
                QtWidgets.QTableWidget.__init__(self, *args)
                self.resizeRowsToContents()
                self.resizeColumnsToContents()
                # self.setSelectionBehavior(QtWidgets.QTableView.selectRow)  # todo
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

                self.resizeColumnsToContents()

        class Container:
            @property
            def info(self):
                info = [
                    [str(k), str(v.shape), str(v.dtype)] for k, v in self.__dict__.items() if isinstance(v, np.ndarray)
                ]
                return info

        self.con = Container()

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
