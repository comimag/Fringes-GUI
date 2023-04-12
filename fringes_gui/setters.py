import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from .params import set_params
from .logic import set_logic


def set_functionality(gui):
    """Give the list of parameters defined in Gui their functionality, i.e. set listeners for when they are changed."""

    set_params(gui)

    gui.params = pg.parametertree.Parameter.create(name="Settings", type="group", children=gui.params)
    gui.tree.setParameters(gui.params, showTop=False)
    gui.params_initial = gui.params.saveState()

    set_logic(gui)

    vis = gui.visibility
    if gui.visibility != "Guru":
        gui.params.param("vis").setValue(
            "Guru")  # switching back from Guru mode prevents loading params which don't fit user options
    gui.params.param("vis").setValue(vis)

    def set_param(param, change):
        # param: The immediate child whose tree state has changed.
        #        Note that the change may have originated from a grandchild.
        # changes: List of tuples describing all changes that have been made in this event: (param, changeDescr, data)

        if change[0][1] != "value":
            return

        key = change[0][0].opts["name"]
        val = change[0][2]

        if key not in ["mode", "verbose"] or \
                key == "axis" and gui.fringes.D == 2:
            if hasattr(gui.con, "fringes"):
                delattr(gui.con, "fringes")

                if hasattr(gui.con, "registration"):
                    gui.view(gui.con.registration)
                elif hasattr(gui.con, gui.key):
                    gui.view(getattr(gui.con, gui.key))
                else:
                    gui.view(None)

            if gui.encode_checkbox.isChecked():
                gui.encode()

        if key not in ["alpha", "dtype", "bias", "amplitude", "visibility", "gamma"] or \
                key == "axis" and gui.fringes.D == 2:
            if hasattr(gui.con, "brightness"):
                delattr(gui.con, "brightness")

            if hasattr(gui.con, "modulation"):
                delattr(gui.con, "modulation")

            if hasattr(gui.con, "registration"):
                delattr(gui.con, "registration")

            if hasattr(gui.con, "phase"):
                delattr(gui.con, "phase")

            if hasattr(gui.con, "orders"):
                delattr(gui.con, "orders")

            if hasattr(gui.con, "residuals"):
                delattr(gui.con, "residuals")

            if hasattr(gui.con, gui.key):
                gui.view(getattr(gui.con, gui.key))
            elif hasattr(gui.con, "fringes"):
                gui.view(gui.con.fringes)
            else:
                gui.view(None)

            if gui.decode_checkbox.isChecked() and gui.decodeOK:
                gui.decode()

            gui.data_table.setData(gui.con.info)
            QtWidgets.QApplication.processEvents()  # refresh event queue

        if key == "vis":
            set_user(val)
            gui.fringes.logger.debug(f"Set visibility to {gui.visibility}.")
            return
        elif key == "log":
            gui.fringes.logger.setLevel(val)
            gui.fringes.logger.debug(f"Set logging level to '{val}'.")
            return

        if key[0] in "Nlvfh" and (key[-1] == "ᵢ" or key[-1].translate(gui.sup).isdigit()) and 2 <= len(key) <= 4:
            key, id = key[0], key[1:]
            idx = id.translate(gui.sup).split(",")
            d = int(idx[0]) if len(idx) == 2 else 0
            k = int(idx[-1]) if idx[-1] != "ᵢ" else 0  # if idx[-1] != "" else 0

            # workaround for problem of changing an element of a property object
            val_old = getattr(gui.fringes, "_" + key).copy()
            if key == "N":
                if val == 1:
                    val_old[:, :] = 1  # enable SSB
                elif gui.fringes.FDM:
                    val_old[:, :] = val
                elif gui.visibility in ["Guru", "Experimental"] or gui.fringes.FDM:
                    val_old[d, k] = val
                else:
                    val_old[:, k] = val
            elif key == "h":
                if val.red() == val.green() == val.blue() == 0:    # handle forbidden value 'black': set back to previous value
                    gui.params.param("col", "H", "h" + id).setValue(gui.fringes.h[k])
                    return
                else:
                    val_old[k] = [val.red(), val.green(), val.blue()]
            elif gui.visibility in ["Guru", "Experimental"] or gui.fringes.FDM:
                val_old[d, k] = val
            else:
                val_old[:, k] = val

            val = val_old
        elif key == "o":
            val *= np.pi

        setattr(gui.fringes, key, val)

        update_parameter_tree()

    def set_user(val):
        user_old = gui.visibility  # used if setting user to experimental
        gui.visibility = val

        with gui.params.treeChangeBlocker():
            if gui.visibility == "Beginner":
                gui.params.param("vid", "T").setLimits((3, gui.fringes._Tmax))
                gui.params.param("vid", "C").hide()
                gui.params.param("vid", "alpha").hide()
                gui.params.param("sys", "grid").setValue(gui.fringes.defaults["grid"])
                gui.params.param("sys", "grid").hide()
                gui.params.param("sys", "angle").setValue(gui.fringes.defaults["angle"])
                gui.params.param("sys", "angle").hide()
                gui.fringes.N = np.maximum(gui.fringes._N, 3)
                set_frequencies()
                gui.params.param("set", "f").hide()
                gui.params.param("set", "reverse").setValue(gui.fringes.defaults["reverse"])
                gui.params.param("set", "reverse").hide()
                gui.params.param("set", "o").setValue(gui.fringes.defaults["o"] / np.pi)
                gui.params.param("set", "o").hide()
                gui.params.param("set", "lmin").setValue(gui.fringes.defaults["lmin"])
                gui.params.param("set", "lmin").hide()
                gui.params.param("set", "vmax").hide()
                gui.params.param("val").hide()
                gui.params.param("val", "dtype").setValue(gui.fringes.defaults["dtype"])
                gui.params.param("val", "A").setValue(gui.fringes.defaults["A"])
                gui.params.param("val", "B").setValue(gui.fringes.defaults["B"])
                gui.params.param("val", "V").setValue(gui.fringes.defaults["V"])
                gui.params.param("val", "gamma").setValue(gui.fringes.defaults["gamma"])
                gui.params.param("mux").hide()
                gui.params.param("mux", "SDM").setValue(gui.fringes.defaults["SDM"])
                gui.params.param("mux", "WDM").setValue(gui.fringes.defaults["WDM"])
                gui.params.param("mux", "FDM", "static").setValue(gui.fringes.defaults["static"])
                gui.params.param("mux", "TDM").setValue(gui.fringes.TDM)
                gui.params.param("col").hide()
                gui.fringes.H = 1
                gui.params.param("uwr", "mode").setValue(gui.fringes.defaults["mode"])
                gui.params.param("uwr", "mode").hide()
                gui.params.param("uwr", "Vmin").setValue(gui.fringes.defaults["Vmin"])
                gui.params.param("uwr", "Vmin").hide()
                gui.params.param("uwr", "verbose").setValue(gui.fringes.defaults["verbose"])
                gui.params.param("uwr", "verbose").hide()
                gui.params.param("quali").hide()
                gui.params.param("quali", "dark").setValue(gui.fringes.defaults["dark"])
            elif gui.visibility == "Expert":
                gui.params.param("vid", "T").setLimits((3, gui.fringes._Tmax))
                gui.params.param("vid", "alpha").hide()
                gui.params.param("sys", "grid").setValue(gui.fringes.defaults["grid"])
                gui.params.param("sys", "grid").hide()
                gui.params.param("sys", "angle").setValue(gui.fringes.defaults["angle"])
                gui.params.param("sys", "angle").hide()
                gui.fringes.N = np.maximum(gui.fringes._N, 2)
                set_frequencies()
                gui.params.param("set", "f").hide()
                gui.params.param("set", "reverse").show()
                gui.params.param("set", "o").show()
                gui.params.param("set", "lmin").show()
                gui.params.param("set", "vmax").hide()
                gui.params.param("val").show()
                gui.params.param("val", "dtype").setValue(gui.fringes.defaults["dtype"])
                gui.params.param("val", "dtype").hide()
                gui.params.param("val", "Imax").setValue(gui.fringes.Imax)
                gui.params.param("val", "Imax").hide()
                gui.params.param("val", "A").show()
                gui.params.param("val", "B").show()
                gui.params.param("val", "V").show()
                gui.params.param("val", "gamma").show()
                gui.params.param("mux").hide()
                gui.params.param("mux", "SDM").setValue(gui.fringes.defaults["SDM"])
                gui.params.param("mux", "WDM").setValue(gui.fringes.defaults["WDM"])
                gui.params.param("mux", "FDM", "static").setValue(gui.fringes.defaults["static"])
                gui.params.param("mux", "TDM").setValue(gui.fringes.TDM)
                gui.params.param("col").hide()
                gui.fringes.M = 1
                gui.params.param("uwr", "mode").setValue(gui.fringes.defaults["mode"])
                gui.params.param("uwr", "mode").hide()
                gui.params.param("uwr", "Vmin").setValue(gui.fringes.defaults["Vmin"])
                gui.params.param("uwr", "Vmin").hide()
                gui.params.param("uwr", "verbose").show()
                gui.params.param("quali").show()
                gui.params.param("quali", "dark").hide()  # Experimental
                gui.params.param("quali", "shot").hide()  # Experimental
            elif gui.visibility == "Guru":
                gui.params.param("vid", "T").setLimits((1, gui.fringes._Tmax))
                gui.params.param("sys", "grid").setValue(gui.fringes.defaults["grid"])
                gui.params.param("sys", "grid").hide()  # todo: experimantal -> guru
                gui.params.param("sys", "angle").setValue(gui.fringes.defaults["angle"])
                gui.params.param("sys", "angle").hide()  # todo: experimantal -> guru
                gui.params.param("vid", "alpha").show()
                gui.params.param("set", "K").setLimits((1, (gui.fringes._Nmax - 1) / 2 / gui.fringes.D if gui.fringes.FDM else gui.fringes._Kmax))
                gui.params.param("set", "f").show()
                # gui.params.param("set", "vmax").show()
                gui.params.param("val", "dtype").show()
                gui.params.param("val", "Imax").show()
                gui.params.param("col", "H").setLimits((1, gui.fringes._Hmax))
                gui.params.param("col", "H").show()
                gui.params.param("mux").show()
                gui.params.param("col").show()
                gui.params.param("uwr", "mode").setValue(gui.fringes.defaults["mode"])
                gui.params.param("uwr", "mode").hide()
                gui.params.param("uwr", "Vmin").setValue(gui.fringes.defaults["Vmin"])
                gui.params.param("uwr", "Vmin").hide()
                gui.params.param("quali").show()
                gui.params.param("quali", "dark").hide()  # Experimental
                gui.params.param("quali", "shot").hide()  # Experimental
            elif gui.visibility == "Experimental":
                if user_old not in ["Guru", "Experimental"]:
                    gui.params.param("vis").setValue("Guru")  # get the settings from Guru mode first
                    gui.params.param("vis").setValue("Experimental")
                gui.params.param("sys", "grid").show()
                gui.params.param("sys", "angle").show()
                gui.params.param("uwr", "mode").show()
                gui.params.param("uwr", "Vmin").show()
                gui.params.param("quali", "dark").show()
                gui.params.param("quali", "shot").show()

            update_parameter_tree()

    def update_parameter_tree():
        is2D = max(gui.fringes.N.ndim, gui.fringes.l.ndim, gui.fringes.v.ndim, gui.fringes.f.ndim) == 2 or \
               gui.fringes.FDM or gui.visibility in ["Guru", "Experimental"]

        children = gui.fringes.D * gui.fringes.K if is2D else gui.fringes.K

        change_indices = len(gui.params.param("set", "v").children()) != children or \
                         gui.fringes.D != gui.params.param("sys", "D").value() or \
                         gui.fringes.K != gui.params.param("set", "K").value() or \
                         gui.fringes.FDM != gui.params.param("mux", "FDM").value() or \
                         gui.params.param("vis").value() in ["Guru", "Experimental"] and \
                         "," not in gui.params.param("set", "v").children()[0].name() or \
                         gui.params.param("vis").value() not in ["Guru", "Experimental"] and \
                         "," in gui.params.param("set", "v").children()[0].name() or \
                         is2D and "," not in gui.params.param("set", "v").children()[0].name() or \
                         not is2D and "," in gui.params.param("set", "v").children()[0].name()

        with gui.params.treeChangeBlocker():
            if gui.fringes.SDM:
                gui.params.param("sys", "grid").setLimits(("image", "Cartesian"))
            else:
                gui.params.param("sys", "grid").setLimits(("image", "Cartesian", "polar", "log-polar"))
            gui.params.param("sys", "grid").setValue(gui.fringes.grid)

            gui.params.param("sys", "angle").setValue(gui.fringes.angle)
            if gui.fringes.grid in gui.fringes._grids[2:]:
                gui.params.param("sys", "angle").setDefault(45)
            else:
                gui.params.param("sys", "angle").setDefault(0)

            gui.params.param("sys", "D").setValue(gui.fringes.D)

            gui.params.param("sys", "D", "axis").setValue(("X", "Y")[gui.fringes.axis])
            gui.params.param("sys", "D", "axis").show(gui.fringes.D == 1)

            gui.params.param("vid", "T").setValue(gui.fringes.T)
            gui.params.param("vid", "T").setLimits((1 if gui.visibility == "Guru" else 3, gui.fringes._Tmax))

            gui.params.param("vid", "Y").setLimits((1, min(gui.fringes._Ymax, gui.fringes._Pmax / gui.fringes.X)))
            gui.params.param("vid", "Y").setValue(gui.fringes.Y)

            gui.params.param("vid", "Y").setLimits((1, min(gui.fringes._Xmax, gui.fringes._Pmax / gui.fringes.Y)))
            gui.params.param("vid", "X").setValue(gui.fringes.X)

            gui.params.param("vid", "C").setValue(gui.fringes.C)

            gui.params.param("vid", "alpha").setValue(gui.fringes.alpha)

            gui.params.param("vid", "L").setValue(gui.fringes.L)

            gui.params.param("set", "K").setValue(gui.fringes.K)

            D = gui.fringes.D if is2D else 1
            for d in range(D):
                for k in range(gui.fringes.K):
                    id = str(k).translate(gui.sub)
                    if is2D:  # prepend index for direction
                        id = str(d).translate(gui.sub) + "," + id

                    if change_indices:  # remove and add child params
                        if d == k == 0:  # in first loop: remove child params
                            gui.params.param("set", "N").clearChildren()
                            gui.params.param("set", "l").clearChildren()
                            gui.params.param("set", "v").clearChildren()
                            gui.params.param("set", "f").clearChildren()

                        if gui.fringes.FDM:  # all shifts must be equal; hence create only one generic index "i"
                            if d == k == 0:
                                gui.params.param("set", "N").addChild(
                                    {
                                        "name": "N" + "ᵢ",
                                        "type": "int",
                                        "value": gui.fringes._N[d, k],
                                        "default": gui.fringes.Nmin if gui.fringes.FDM else gui.fringes.defaults["N"][
                                            0, 0],
                                        "limits": (max(gui.fringes.Nmin,
                                                       1 if gui.visibility == "Guru" else 2 if gui.visibility == "Expert" else 3),
                                                   gui.fringes._Nmax),
                                        "tip": gui.fringes.__class__.N.__doc__,
                                    }
                                )
                        else:
                            gui.params.param("set", "N").addChild(
                                {
                                    "name": "N" + id,
                                    "type": "int",
                                    "value": gui.fringes._N[d, k],
                                    "default": gui.fringes.Nmin if gui.fringes.FDM else gui.fringes.defaults["N"][0, 0],
                                    "limits": (max(gui.fringes.Nmin,
                                                   1 if gui.visibility == "Guru" else 2 if gui.visibility == "Expert" else 3),
                                               gui.fringes._Nmax),
                                    "tip": gui.fringes.__class__.N.__doc__,
                                }
                            )

                        gui.params.param("set", "l").addChild(
                            {
                                "title": "\u03BB" + id,
                                "name": "l" + id,
                                "type": "float",
                                "value": gui.fringes._l[d, k],
                                "default": None,  # gui.fringes.L ** (1 / (k + 1)),
                                "limits": (gui.fringes.lmin, None),
                                "decimals": gui.digits,
                                "tip": gui.fringes.__class__.l.__doc__,
                            }
                        )

                        gui.params.param("set", "v").addChild(
                            {
                                "title": "\u03BD" + id,
                                "name": "v" + id,
                                "type": "float",
                                "value": gui.fringes._v[d, k],
                                "default": None,  # gui.fringes.L ** (1 - 1 / (k + 1)),
                                "limits": (0, gui.fringes.vmax),
                                "decimals": gui.digits,
                                "tip": gui.fringes.__class__.v.__doc__,
                            }
                        )

                        gui.params.param("set", "f").addChild(
                            {
                                "name": "f" + id,
                                "type": "float",
                                "value": gui.fringes._f[d, k],
                                "default": gui.fringes._f[
                                    d, k] if gui.fringes.FDM and gui.fringes.static else gui.fringes.D * k + k if gui.fringes.FDM else 1,
                                "limits": (-gui.fringes.vmax, gui.fringes.vmax),
                                "decimals": gui.digits,
                                "readonly": gui.fringes.FDM and gui.fringes.static,
                                "tip": gui.fringes.__class__.f.__doc__,
                            }
                        )
                    else:  # update child params
                        if gui.fringes.FDM:  # all shifts must be equal; hence create only one generic index "i"
                            if d == k == 0:
                                gui.params.param("set", "N", "N" + "ᵢ").setLimits((max(gui.fringes.Nmin,
                                                                                       1 if gui.visibility == "Guru" else 2 if gui.visibility == "Expert" else 3),
                                                                                   gui.fringes._Nmax))
                                gui.params.param("set", "N", "N" + "ᵢ").setValue(gui.fringes._N[d, k])
                                gui.params.param("set", "N", "N" + "ᵢ").setDefault(
                                    gui.fringes.Nmin if gui.fringes.FDM else gui.fringes.defaults["N"][0, 0])
                        else:
                            gui.params.param("set", "N", "N" + id).setLimits((max(gui.fringes.Nmin,
                                                                                  1 if gui.visibility == "Guru" else 2 if gui.visibility == "Expert" else 3),
                                                                              gui.fringes._Nmax))
                            gui.params.param("set", "N", "N" + id).setValue(gui.fringes._N[d, k])
                            gui.params.param("set", "N", "N" + id).setDefault(
                                gui.fringes.Nmin if gui.fringes.FDM else gui.fringes.defaults["N"][0, 0])

                        gui.params.param("set", "l", "l" + id).setLimits((gui.fringes.lmin, None))
                        gui.params.param("set", "l", "l" + id).setValue(gui.fringes._l[d, k])

                        gui.params.param("set", "v", "v" + id).setLimits((0, gui.fringes.vmax))
                        gui.params.param("set", "v", "v" + id).setValue(gui.fringes._v[d, k])

                        gui.params.param("set", "f", "f" + id).setLimits((-gui.fringes.vmax, gui.fringes.vmax))
                        gui.params.param("set", "f", "f" + id).setValue(gui.fringes._f[d, k])
                        gui.params.param("set", "f", "f" + id).setDefault(None if gui.fringes.FDM else 1)

            gui.params.param("set", "reverse").setValue(gui.fringes.reverse)

            gui.params.param("set", "o").setValue(gui.fringes.o / np.pi)

            gui.params.param("set", "lmin").setValue(gui.fringes.lmin)

            gui.params.param("set", "vmax").setValue(gui.fringes.vmax)

            gui.params.param("set", "UMR").setValue(gui.fringes.UMR.min())

            gui.params.param("val", "dtype").setValue(gui.fringes.dtype)

            gui.params.param("val", "Imax").setValue(gui.fringes.Imax)

            gui.params.param("val", "A").setLimits((gui.fringes.Amin, gui.fringes.Amax))
            gui.params.param("val", "A").setValue(gui.fringes.A)
            gui.params.param("val", "A").setDefault(gui.fringes.Imax / 2)

            gui.params.param("val", "B").setLimits((0, gui.fringes.Bmax))
            gui.params.param("val", "B").setValue(gui.fringes.B)
            gui.params.param("val", "B").setDefault(gui.fringes.Imax / 2)

            gui.params.param("val", "V").setValue(gui.fringes.V)

            gui.params.param("val", "gamma").setValue(gui.fringes.gamma)

            if len(gui.params.param("col", "H").children()) != gui.fringes.H or \
                    gui.params.param("col", "H").value() != gui.fringes.H or \
                    gui.params.param("col", "M").value() != gui.fringes.M:  # remove and add cild params

                gui.params.param("col", "H").clearChildren()

                for h in range(gui.fringes.H):
                    id = str(h).translate(gui.sub)
                    gui.params.param("col", "H").addChild(
                        {
                            "name": "h" + id,
                            "type": "color",
                            "value": gui.fringes.h[h],
                            "default": gui.fringes.defaults["h"][0],
                            "limits": (1, gui.fringes._Hmax),
                            "readonly": gui.visibility in ["Beginner", "Expert"]
                        }
                    )
            else:  # update cild params
                for h in range(gui.fringes.H):
                    id = str(h).translate(gui.sub)
                    gui.params.param("col", "H", "h" + id).setValue(gui.fringes.h[h])

            gui.params.param("col", "M").setValue(gui.fringes.M)

            gui.params.param("col", "H").setValue(gui.fringes.H)

            gui.params.param("mux", "TDM").setValue(gui.fringes.TDM)

            gui.params.param("mux", "SDM").setValue(gui.fringes.SDM)
            gui.params.param("mux", "SDM").setReadonly(not gui.SDMOK)

            gui.params.param("mux", "WDM").setValue(gui.fringes.WDM)
            gui.params.param("mux", "WDM").setReadonly(not gui.WDMOK)

            gui.params.param("mux", "FDM").setValue(gui.fringes.FDM)
            gui.params.param("mux", "FDM").setReadonly(not gui.FDMOK)

            gui.params.param("mux", "FDM", "static").setValue(gui.fringes.static)
            gui.params.param("mux", "FDM", "static").show(gui.fringes.FDM)

            gui.params.param("uwr", "PU").setValue(gui.fringes.PU)

            gui.params.param("uwr", "mode").setValue(gui.fringes.mode)

            gui.params.param("uwr", "Vmin").setValue(gui.fringes.Vmin)

            gui.params.param("uwr", "verbose").setValue(gui.fringes.verbose)

            gui.params.param("quali", "eta").setValue(gui.fringes.eta.max())

            gui.params.param("quali", "dark").setValue(gui.fringes.dark)

            gui.params.param("quali", "quant").setValue(gui.fringes.quant)

            gui.params.param("quali", "shot").setValue(gui.fringes.shot)

            gui.params.param("quali", "u").setValue(gui.fringes.u)

            gui.params.param("quali", "DR").setValue(gui.fringes.DRdB.max())

        gui.reset_button.setEnabled(gui.fringes.params != gui.initials)
        gui.encode_button.setStyleSheet("" if gui.encodeOK else "QPushButton{color: red}")
        gui.decode_button.setEnabled(gui.decodeOK)
        gui.decode_key.setEnabled(gui.decodeOK)
    gui.update_parameter_tree = update_parameter_tree

    def set_shifts():
        gui.fringes.N = "auto"
        update_parameter_tree()

    def set_wavelengths():
        gui.fringes.l = "auto"
        update_parameter_tree()

    def set_periods():
        gui.fringes.v = "auto"
        update_parameter_tree()

    def set_frequencies():
        gui.fringes.f = "auto"
        update_parameter_tree()

    gui.params.param("set", "N").sigActivated.connect(set_shifts)
    gui.params.param("set", "l").sigActivated.connect(set_wavelengths)
    gui.params.param("set", "v").sigActivated.connect(set_periods)
    gui.params.param("set", "f").sigActivated.connect(set_frequencies)
    a = gui.params
    b = gui.params.sigTreeStateChanged

    gui.params.sigTreeStateChanged.connect(set_param)
