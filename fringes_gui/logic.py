import numpy as np
import os
import functools

from pyqtgraph.Qt import QtWidgets
from PyQt6.QtWidgets import QFileDialog
import pyqtgraph as pg
import cv2
import json
import yaml
import asdf
import toml
import fringes as frng


config = {
    ".json": json.load,
    ".yaml": yaml.safe_load,
    ".toml": toml.load,
    ".asdf": asdf.open,
}

image = {
    ".bmp": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".dip": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),  # i.e. ".bmp"
    # ".jpeg": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".jpg": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".jpe": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".jp2": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    ".png": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".webp": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".pbm": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".pgm": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".ppm": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".pxm": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".pnm": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".sr": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".ras": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    ".tiff": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    ".tif": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
}

numpy = {
    ".mmap": np.load,
    ".npy": np.load,
    # ".npz": np.load  # todo
}

loader = {**config, **image, **numpy}


def set_logic(gui):
    """Assigns functionality to the buttons and keys in the GUI."""

    def immerse():
        gui.immerse = not gui.immerse

        if gui.immerse:
            gui.win.showFullScreen()
        else:
            gui.win.showMaximized()

    def reset():
        gui.fringes.reset()
        gui.update_parameter_tree()

        clear()

    def load():
        """Load data from given directory."""

        flist = QFileDialog.getOpenFileNames(
            caption="Select file(s)",
            # directory=os.path.join(os.path.expanduser("~"), "Videos"),
            # options=QFileDialog.Option.DontUseNativeDialog,
            filter=f"Images {tuple('*' + key for key in image.keys())};;Numpy {tuple('*' + key for key in numpy.keys())};;Config {tuple('*' + key for key in config.keys())}".replace(",", "").replace("'", "")
        )

        if flist[0]:
            with pg.BusyCursor():
                path, base = os.path.split(flist[0][0])
                name, ext = os.path.splitext(base)

                if ext in config.keys():
                    gui.fringes.load(flist[0][0])
                    gui.update_parameter_tree()
                else:  # load data
                    root = name.rstrip("1234567890").rstrip("_")

                    data = loader[ext](flist[0][0])
                    shape = data.shape
                    dtype = data.dtype

                    sequence = np.empty((len(flist[0]),) + shape, dtype)
                    sequence[0] = data
                    for i, f in enumerate(flist[0][1:]):
                        path, base = os.path.split(f)
                        name, ext_i = os.path.splitext(base)
                        root_i = name.rstrip("1234567890").rstrip("_")
                        data_i = loader[ext_i](f)

                        if ext_i == ext and root_i == root and data.shape == data_i.shape and data.dtype == data_i.dtype:
                            sequence[i+1] = data_i
                        else:
                            gui.fringes.logger.error("Files in list dint't match; terminated loading data.")
                            return

                    sequence = frng.vshape(sequence)
                    setattr(gui.con, root, sequence)
                    gui.fringes.logger.info(f"Loaded data from '{path}'.")

                    view(getattr(gui.con, root))
                    gui.data_table.setData(gui.con.info)
                    QtWidgets.QApplication.processEvents()  # refresh event queue

            gui.decode_button.setEnabled(gui.decodeOK)
            gui.decode_key.setEnabled(gui.decodeOK)
            gui.remap_button.setEnabled(gui.remapOK)
            gui.remap_key.setEnabled(gui.remapOK)
            gui.curvature_button.setEnabled(gui.curvatureOK)
            gui.curvature_key.setEnabled(gui.curvatureOK)
            gui.height_button.setEnabled(gui.heightOK)
            gui.height_key.setEnabled(gui.heightOK)
            gui.clear_button.setEnabled(gui.dataOK)
            gui.clear_key.setEnabled(gui.dataOK)
            gui.set_button.setEnabled(gui.set_dataOK)

    def save():
        """Save all data to current directory."""

        path = QFileDialog.getExistingDirectory(
            caption="Select directory",
            # directory=os.path.join(os.path.expanduser("~"), "Videos"),
            # options=QFileDialog.Option.DontUseNativeDialog,
        )

        if os.path.isdir(os.path.abspath(path)):
            with pg.BusyCursor():
                gui.fringes.save(os.path.join(path, "params.yaml"))

                for k, v in gui.con.__dict__.items():
                    if isinstance(v, np.ndarray) and v.size > 0:
                        T, Y, X, C = v.shape = frng.vshape(v).shape
                        color_order = (2, 1, 0, 3) if C == 4 else (2, 1, 0) if C == 3 else 0  # to compensate OpenCV color order
                        color_channels = (1, 3, 4)
                        is_img_shape = v.ndim <= 2 or v.ndim == 3 and v.shape[-1] in color_channels
                        is_vid_shape = v.ndim == 3 or v.ndim == 4 and v.shape[-1] in color_channels
                        is_img_dtype = v.dtype in (bool, np.uint8, np.uint16) or \
                                       v.dtype in (np.float32,) and np.min(v) >= 0 and np.max(v) <= 1  # todo: np.float16, np.float64

                        if is_img_dtype and is_img_shape:  # save as image
                            fname = os.path.join(path, f"{k}.tif")
                            cv2.imwrite(fname, v[..., color_order])
                        elif is_img_dtype and is_vid_shape:  # save as image sequence
                            for t in range(T):
                                fname = os.path.join(path, f"{k}_{str(t + 1).zfill(len(str(T)))}.tif")
                                cv2.imwrite(fname, v[t][..., color_order])
                        else:  # save as numpy array
                            np.save(os.path.join(path, f"{k}.npy"), v)

                gui.fringes.logger.info(f"Saved data to '{path}'.")

    def clear():
        """Clear all data from the gui_util."""

        view(None)

        gui.key = ""

        con_dict = gui.con.__dict__.copy()
        for k, v in con_dict.items():
            if isinstance(v, np.ndarray):
                delattr(gui.con, k)

        gui.data_table.clearContents()
        gui.data_table.setRowCount(0)
        gui.set_button.setEnabled(False)

        gui.decode_button.setEnabled(False)
        gui.decode_key.setEnabled(False)
        gui.remap_button.setEnabled(False)
        gui.remap_key.setEnabled(False)
        gui.curvature_button.setEnabled(False)
        gui.curvature_key.setEnabled(False)
        gui.height_button.setEnabled(False)
        gui.height_key.setEnabled(False)
        gui.clear_button.setEnabled(False)
        gui.clear_key.setEnabled(False)

    def set_data():
        """Set data (to be decoded)."""

        try:
            item = gui.data_table.currentItem().text()  # todo: select first element of selected row

            if hasattr(gui.con, item):
                view(getattr(gui.con, item))
                gui.key = item
                gui.fringes.logger.info(f"Set data to be decoded to '{gui.key}'.")
            else:
                view(None)
                gui.key = None

            gui.decode_button.setEnabled(gui.decodeOK)
            gui.decode_key.setEnabled(gui.decodeOK)
        except Exception:
            pass

    def coordinates():
        """Coordinates being encoded."""
        if hasattr(gui.con, "coordinates"):
            delattr(gui.con, "coordinates")

        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        with pg.BusyCursor():
            gui.con.coordinates = gui.fringes.coordinates()

        view(getattr(gui.con, "coordinates").astype(float))
        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

    def encode():
        """Encode fringes based on the given parameters."""
        if hasattr(gui.con, "fringes"):
            delattr(gui.con, "fringes")

        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        with pg.BusyCursor():
            gui.con.fringes = gui.fringes.encode()

        view(getattr(gui.con, "fringes"))
        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        gui.decode_button.setEnabled(gui.decodeOK)
        gui.decode_key.setEnabled(gui.decodeOK)
        gui.clear_button.setEnabled(gui.dataOK)
        gui.clear_key.setEnabled(gui.dataOK)
        gui.set_button.setEnabled(gui.set_dataOK)
    gui.encode = encode

    def decode():
        """Decode encoded or acquired fringes."""
        if hasattr(gui.con, "brightness"):
            delattr(gui.con, "brightness")
        if hasattr(gui.con, "modulation"):
            delattr(gui.con, "modulation")
        if hasattr(gui.con, "registration"):
            delattr(gui.con, "registration")
        if hasattr(gui.con, "phase"):
            delattr(gui.con, "phase")
        if hasattr(gui.con, "residuals"):
            delattr(gui.con, "residuals")
        if hasattr(gui.con, "orders"):
            delattr(gui.con, "orders")

        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        if hasattr(gui.con, gui.key):
            I = getattr(gui.con, gui.key)
        # elif hasattr(gui.con, "raw"):
        #     I = gui.con.raw
        elif hasattr(gui.con, "fringes"):
            I = gui.con.fringes

        with pg.BusyCursor():
            dec = gui.fringes.decode(I)

            if dec is not None:
                for k, v in dec._asdict().items():
                    setattr(gui.con, k, v)

            view(getattr(gui.con, "registration"))
        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        gui.remap_button.setEnabled(gui.remapOK)
        gui.remap_key.setEnabled(gui.remapOK)
        gui.curvature_button.setEnabled(gui.curvatureOK)
        gui.curvature_key.setEnabled(gui.curvatureOK)
        gui.set_button.setEnabled(gui.set_dataOK)
    gui.decode = decode

    def remap():
        if hasattr(gui.con, "source"):
            delattr(gui.con, "source")

        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        with pg.BusyCursor():
            x = gui.con.registration
            B = gui.con.modulation
            Bmin = 0
            scale = 1
            normalize = True

            gui.con.source = gui.fringes.remap(x, B, Bmin, scale, normalize)

            gui.fringes.logger.info("Remapped.")

        view(getattr(gui.con, "source"))
        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

    def curvature():
        if hasattr(gui.con, "curvature"):
            delattr(gui.con, "curvature")

        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        with pg.BusyCursor():
            gui.con.curvature = frng.curvature(gui.con.registration)

            gui.fringes.logger.info("Computed curvature.")

        view(getattr(gui.con, "curvature"))
        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        gui.height_button.setEnabled(True)
        gui.height_key.setEnabled(True)

    def height():
        if hasattr(gui.con, "height"):
            delattr(gui.con, "height")

        with pg.BusyCursor():
            gui.con.height = frng.height(gui.con.curvature)

            gui.fringes.logger.info("Computed height.")

        view(getattr(gui.con, "height"))
        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

    def view(I=None, Imax=None, autoscale=False, enhance=False, cmap=None):  # todo: color map
        """Display image data in videoshape in the ImageView area of the GUI."""
        try:
            T, Y, X, C = frng.vshape(I).shape
        except:
            T = Y = X = C = None

        if enhance and I is not None and T == 1 and 3 <= C and I.dtype == "uint8":  # improve lightfield-inspection
            try:
                I[..., 1] = 0
                hsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)
                hsv[..., 1] = 255  # 2
                I = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            except:
                pass

        if I is None or not isinstance(I, np.ndarray) or not I.size:
            gui.imv.clear()
        elif I.ndim < 2:
            gui.fringes.logger.error("Can't display array with less than 2 dimensions.")  # todo: convert into videoshape
        elif I.ndim > 4:
            gui.fringes.logger.error("Can't display array with more than 4 dimensions.")  # todo: convert into videoshape
        elif I.dtype.kind in "b":  # bool
            gui.imv.setImage(I, autoLevels=False, levels=(0, 1), autoHistogramRange=False)
            # gui.plot.setLimits(xMin=0, xMax=X, yMin=0, yMax=Y)
        elif I.dtype.kind in "ui":  # uint or int
            if autoscale:
                gui.imv.setImage(I)
            else:
                if I.dtype.kind == "u" or np.min(I) >= 0:
                    Imin = 0
                else:
                    Imin = np.iinfo(I.dtype).min

                if Imax is not None:
                    Imax = int(Imax)
                elif I.dtype.itemsize > 1:  # e.g. 16bit data may hold only 10bit or 12bit or 14bit information
                    if np.any(I > 2 ** 10 - 1):  # 10-bit data
                        Imax = 2 ** 10 - 1
                    elif np.any(I > 2 ** 12 - 1):  # 12-bit data
                        Imax = 2 ** 12 - 1
                    elif np.any(I > 2 ** 14 - 1):  # 14-bit data
                        Imax = 2 ** 14 - 1
                    else:
                        Imax = np.iinfo(I.dtype).max
                else:
                    Imax = np.iinfo(I.dtype).max

                gui.imv.setImage(I, autoLevels=False, levels=(Imin, Imax), autoHistogramRange=False)
                # gui.plot.setLimits(xMin=0, xMax=X, yMin=0, yMax=Y)
        elif I.dtype.kind in "f":  # float
            try:
                if np.isnan(np.amax(I)):
                    I = I.astype(np.float32)  # copy
                    I[np.isnan(I)] = 0
                gui.imv.setImage(I.astype(np.float32, copy=False))  # it's faster for float32 than for float64
            except:
                gui.fringes.logger.error("Couldn't display float data.")

            # if cmap:  # todo: define cmaps
            #     try:
            #         from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
            #
            #         # list the available colormaps
            #         print(Gradients.keys())
            #
            #         # pick one to turn into an actual colormap
            #         spectrumColormap = pg.ColorMap(*zip(*Gradients["spectrum"]["ticks"]))
            #
            #         # create colormaps from the builtins
            #         pos, color = zip(*Gradients[cmap]["ticks"])
            #         cmap = pg.ColorMap(pos, color)
            #
            #         gui.imv.setColorMap(cmap)
            #     except:
            #         pass

            # gui.plot.setLimits(xMin=0, xMax=X, yMin=0, yMax=Y)
        else:
            gui.fringes.logger.error("Can't display array with dtype %s." % I.dtype)

        QtWidgets.QApplication.processEvents()  # refresh event queue
    gui.view = view

    def zoomback():
        pass  # todo: zoom back strg + 0

    def selection_changed():
        """Display the data which was selected."""

        try:
            item = gui.data_table.currentItem().text()  # todo: select first element of selected row
        except:
            return

        if hasattr(gui.con, item):
            view(getattr(gui.con, item))
        else:
            view(None)

    # assign functionality to buttons
    gui.immerse_key.activated.connect(immerse)

    gui.reset_button.clicked.connect(reset)
    gui.reset_key.activated.connect(reset)

    gui.load_button.clicked.connect(load)
    gui.load_key.activated.connect(load)
    gui.save_button.clicked.connect(save)
    gui.save_key.activated.connect(save)
    gui.clear_button.clicked.connect(clear)
    gui.clear_key.activated.connect(clear)

    gui.set_button.clicked.connect(set_data)
    gui.set_key.activated.connect(set_data)
    gui.data_table.itemSelectionChanged.connect(selection_changed)

    gui.coordinates_key.activated.connect(coordinates)
    gui.encode_button.clicked.connect(encode)
    gui.encode_key.activated.connect(encode)
    gui.decode_button.clicked.connect(decode)
    gui.decode_key.activated.connect(decode)

    gui.remap_button.clicked.connect(remap)
    gui.remap_key.activated.connect(remap)
    gui.curvature_button.clicked.connect(curvature)
    gui.curvature_key.activated.connect(curvature)
    gui.height_button.clicked.connect(height)
    gui.height_key.activated.connect(height)

    gui.zoomback_key.activated.connect(zoomback)
