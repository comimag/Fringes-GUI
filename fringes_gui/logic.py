import numpy as np
import os
import glob

# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import functools

from pyqtgraph.Qt import QtWidgets, QtGui
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
import cv2
import json
import h5py
import asdf
import fringes as frng


config = frng.Fringes._loader

image = {  # https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
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
    # ".pfm": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".sr": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".ras": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    ".tiff": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    ".tif": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".exr": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".hdr": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
    # ".pic": functools.partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED),
}

binary = {
    ".mmap": np.load,
    ".npy": np.load,
    ".npz": functools.partial(np.load, allow_pickle=False),
    ".asdf": asdf.open,
    ".hdf": h5py.File,
    ".hdf5": h5py.File,
    ".h5": h5py.File,
}

loader = {**config, **image, **binary}


def set_logic(gui):
    """Assigns functionality to the buttons and keys in the GUI."""

    def immerse():
        gui.immerse = not gui.immerse

        if gui.immerse:
            gui.win.showFullScreen()
        else:
            gui.win.showMaximized()

    def undo():
        if len(gui.param_buffer) and gui.param_index >= 0:
            gui.param_index -= 1
            gui.tree.setParameters(gui.param_buffer[gui.param_index], showTop=False)

    def redo():
        if len(gui.param_buffer) > gui.param_index + 1:
            gui.param_index += 1
            gui.tree.setParameters(gui.param_buffer[gui.param_index], showTop=False)

    def reset():
        with gui.params.treeChangeBlocker():
            gui.params.param("vis").setValue("Expert")  # should be the same as in gui.py
            gui.fringes.reset()
            gui.fringes.save(os.path.join(os.path.expanduser("~"), ".fringes.yaml"))
            gui.update_parameter_tree()
            # gui.params.param("log").setValue("INFO")  # should be the same as in gui.py

        clear()
        gui.reset_button.setEnabled(gui.resetOK)

    def load():
        """Load data from given directory."""

        flist = QFileDialog.getOpenFileNames(
            caption="Select file(s)",
            # directory=os.path.join(os.path.expanduser("~"), "Videos"),
            # options=QFileDialog.Option.DontUseNativeDialog,
            filter=f"All {tuple('*' + key for key in loader.keys())};;"
            f"Images {tuple('*' + key for key in image.keys())};;"
            f"Binary {tuple('*' + key for key in binary.keys())};;"
            f"Config {tuple('*' + key for key in config.keys())};;".replace(",", "").replace("'", ""),
        )[0]

        if flist:
            with pg.BusyCursor():
                path, base = os.path.split(flist[0])
                name, ext = os.path.splitext(base)

                if ext in config.keys():  # load config
                    gui.fringes.load(flist[0])
                    gui.update_parameter_tree()
                else:  # load data
                    data = loader[ext](flist[0])

                    if ext in [".hdf", ".hdf5", ".h5"]:
                        # todo: params

                        key = ""
                        for k, v in data.items():
                            if hasattr(v, "shape") and hasattr(v, "shape"):
                                setattr(gui.con, k, np.array(v))
                                if not key:
                                    key = k

                        if key:
                            gui.fringes.logger.info(f"Loaded data from '{flist[0]}'.")  # only first asdf-file is loaded
                            view(getattr(gui.con, key))
                    elif ext == ".asdf":
                        if "fringes-params" in data:
                            params = data["fringes-params"]
                            gui.fringes.params = params

                        key = ""
                        for k, v in data.items():
                            if isinstance(v, np.ndarray):
                                setattr(gui.con, k, v)
                                if not key:
                                    key = k

                        if key:
                            gui.fringes.logger.info(f"Loaded data from '{flist[0]}'.")  # only first asdf-file is loaded
                            view(getattr(gui.con, key))
                    elif ext == ".npz":
                        for key in data.files:
                            try:
                                datum = frng.vshape(data[key])
                                setattr(gui.con, key, datum)
                            except ValueError:
                                pass  # Object arrays cannot be loaded when allow_pickle=False

                        gui.fringes.logger.info(f"Loaded data from '{flist[0]}'.")  # only first npz-file is loaded
                        view(getattr(gui.con, data.files[0]))
                    else:
                        root = name.rstrip("1234567890").rstrip("_")

                        if len(flist) > 1:
                            # data is only one datum in list of data
                            if all(os.path.splitext(f) == ".npy" for f in flist):
                                for f in flist:
                                    path, base = os.path.split(flist[0])
                                    name, ext = os.path.splitext(base)
                                    datum = np.load(f)
                                    setattr(gui.con, name, datum)
                                gui.fringes.logger.info(f"Loaded data from '{os.path.join(path, root + '*') + ext}'.")
                            elif all(os.path.splitext(f) in image for f in flist):
                                # here, data is only one image (one datum) in image stack (data)
                                datum = data
                                data = np.empty((len(flist),) + datum.shape, datum.dtype)
                                data[0] = datum

                                for i, f in enumerate(flist[1:]):
                                    path, base = os.path.split(f)
                                    name, ext_ = os.path.splitext(base)
                                    root_ = name.rstrip("1234567890").rstrip("_")

                                    if ext_ == ext and root_ == root:
                                        datum = loader[ext_](f)

                                        if datum.shape == data.shape[1:] and datum.dtype == data.dtype:
                                            data[i + 1] = datum
                                        else:
                                            gui.fringes.logger.error(
                                                "Files in list dint't match shape and dtype. " "Terminated loading data.")
                                            return

                                gui.fringes.logger.info(f"Loaded data from '{os.path.join(path, root + '*') + ext}'.")
                        else:
                            gui.fringes.logger.info(f"Loaded data from '{flist[0]}'.")

                        data = frng.vshape(data)
                        setattr(gui.con, root, data)
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

        # path = QFileDialog.getExistingDirectory(
        #     caption="Select directory",
        #     # directory=os.path.join(os.path.expanduser("~"), "Videos"),
        #     # options=QFileDialog.Option.DontUseNativeDialog,
        # )

        fname = QFileDialog.getSaveFileName(
            caption="Save dataset",
            filter=f"Misc {tuple('*' + key for key in ['.tif', '.npy'])};;"
                   f"Binary {tuple('*' + key for key in ['.npz', '.asdf'])};;"
                   f"Config {tuple('*' + key for key in config.keys())};;".replace(",", "").replace("'", "")
        )[0]

        path, base = os.path.split(fname)
        name, ext = os.path.splitext(base)

        if os.path.isdir(path):
            with pg.BusyCursor():
                if ext in config and (not gui.con or ext != ".asdf"):  # config can only be asdf if there is no data in gui.con
                    gui.fringes.save(fname)
                    gui.fringes.logger.info(f"Saved config to '{fname}'.")
                elif ext == ".asdf":
                    tree = {}
                    tree["fringes-params"] = gui.fringes.params  # params first!
                    for k, v in vars(gui.con).items():
                        tree[k] = v

                    # Create the ASDF file object from our data tree
                    af = asdf.AsdfFile(tree)

                    # Write the data to a new file
                    af.write_to(fname)

                    gui.fringes.logger.info(f"Saved data to '{fname}'.")
                elif ext == ".npz":
                    # todo: save params to json str to array of type str
                    # s = json.dumps(gui.fringes.params)
                    # a = np.array(s)
                    # a2s = np.array2string(a)
                    # s2 = json.loads(a2s)

                    gui.fringes.save(os.path.join(path, "params.yaml"))

                    tree = vars(gui.con)
                    np.savez(fname, **tree)
                else:  # choose file format disk space effiently
                    gui.fringes.save(os.path.join(path, "params.yaml"))

                    for k, v in gui.con.__dict__.items():
                        if isinstance(v, np.ndarray) and v.size > 0:
                            T, Y, X, C = v.shape = frng.vshape(v).shape
                            color_order = (
                                (2, 1, 0, 3) if C == 4 else (2, 1, 0) if C == 3 else 0
                            )  # compensate OpenCV color order
                            color_channels = (1, 3, 4)
                            is_img_shape = v.ndim <= 2 or v.ndim == 3 and v.shape[-1] in color_channels
                            is_vid_shape = v.ndim == 3 or v.ndim == 4 and v.shape[-1] in color_channels
                            is_img_dtype = (
                                v.dtype in (bool, np.uint8, np.uint16)
                                # or v.dtype in (np.float32,)  # here, OpenCV uses LogLuv high dynamic range encoding (4 bytes per pixel)
                                # and np.min(v) >= 0
                                # and np.max(v) <= 1
                            )  # todo: np.float16, np.float64
                            is_exr_shape = False  # todo: exr_shape
                            is_exr_dtype = v.dtype in (np.float16, np.float32, np.uint32)

                            if is_img_dtype and is_img_shape:  # save as image
                                fname = os.path.join(path, f"{k}.tif")
                                cv2.imwrite(fname, v[..., color_order])
                            elif is_img_dtype and is_vid_shape:  # save as image sequence
                                for t in range(T):
                                    fname = os.path.join(path, f"{k}_{str(t + 1).zfill(len(str(T)))}.tif")
                                    cv2.imwrite(fname, v[t][..., color_order])
                            # elif is_exr_dtype:
                            #     pass  # todo
                            else:  # save as numpy array
                                np.save(os.path.join(path, f"{k}.npy"), v)
                    else:  # executes only after the loop completes normally
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
        if hasattr(gui.con, "uncertainty"):
            delattr(gui.con, "uncertainty")
        if hasattr(gui.con, "exposure"):
            delattr(gui.con, "exposure")
        if hasattr(gui.con, "visibility"):
            delattr(gui.con, "visibility")

        # todo: generic deletion
        # for key in gui.fringes._verbose_output:
        #     if hasattr(gui.con, key):
        #         delattr(gui.con, key)

        gui.data_table.setData(gui.con.info)
        QtWidgets.QApplication.processEvents()  # refresh event queue

        if hasattr(gui.con, gui.key):
            I = getattr(gui.con, gui.key)
        # elif hasattr(gui.con, "raw"):
        #     I = gui.con.raw
        elif hasattr(gui.con, "fringes"):
            I = gui.con.fringes

        flist = glob.glob(os.path.join(os.path.dirname(frng.decoder.__file__), "__pycache__", "decoder*decode*.nbc"))
        if max(os.path.getmtime(file) for file in flist) < os.path.getmtime(__file__):
            dialog = QMessageBox()
            dialog.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), "numba-blue-icon-rgb.svg")))
            dialog.setWindowTitle("Info")
            dialog.setIcon(QMessageBox.Icon.Information)
            dialog.setText("For the compitationally expensive decoding we make use of the just-in-time compiler Numba. During the first execution, an initial compilation is executed. This can take several tens of seconds up to single digit minutes, depending on your CPU. However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster, approaching the speeds of code written in C.")
            dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
            dialog.exec()

        with pg.BusyCursor():
            dec = gui.fringes.decode(I)

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
            if hasattr(gui.con, "modulation"):
                B = gui.con.modulation
                Bmin = 0
                B[B < Bmin] = 0
            else:
                B = None
            mode = gui.fringes.mode  # todo

            gui.con.source = gui.fringes.remap(x, B, mode=mode)

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
            gui.fringes.logger.error(
                "Can't display array with less than 2 dimensions."
            )  # todo: convert into videoshape
        elif I.ndim > 4:
            gui.fringes.logger.error(
                "Can't display array with more than 4 dimensions."
            )  # todo: convert into videoshape
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
                    if np.any(I > 2**10 - 1):  # 10-bit data
                        Imax = 2**10 - 1
                    elif np.any(I > 2**12 - 1):  # 12-bit data
                        Imax = 2**12 - 1
                    elif np.any(I > 2**14 - 1):  # 14-bit data
                        Imax = 2**14 - 1
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

    gui.undo_key.activated.connect(undo)
    gui.redo_key.activated.connect(redo)
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
