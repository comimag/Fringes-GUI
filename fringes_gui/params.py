import logging as lg

import numpy as np


def set_params(gui):
    """Define the parameter tree."""

    vis = {
        "title": "Visibility",
        "name": "vis",
        "type": "list",
        "value": gui.visibility,
        "limits": ["Beginner", "Expert", "Guru", "Experimental"],
        "tip": f"The Visibility defines the type of user that should get access to the feature. "
               f"It does not affect the functionality of the features but is used by the GUI to "
               f"decide which features to display based on the current user level. The purpose "
               f"is mainly to ensure that the GUI is not cluttered with information that is not "
               f"intended at the current visibility level. The following criteria have been used "
               f"for the assignment of the recommended visibility:\n"
               f"\u2B9A Beginner: Features that should be visible for all users via the GUI. This "
               f"is the default visibility. The number of features with 'Beginner' visibility "
               f"should be limited to all basic features so the GUI display is well-organized "
               f"and easy to use.\n"
               f"\u2B9A Expert: Features that require a more in-depth knowledge of the system "
               f"functionality. This is the preferred visibility level for all advanced features.\n"
               f"\u2B9A Guru: - Guru: Advanced features that usually only people"
               f"with a sound background in phase shifting can make good use of.\n"
               f"\u2B9A Experimental: New features that have not been tested yet "
               f"and the system might probably crash at some point.",
    }
    log = {
        "title": "Logging",
        "name": "log",
        "type": "list",
        "value": lg.getLevelName(gui.fringes.logger.level),
        "default": lg.getLevelName(gui.fringes.logger.level),
        "limits": ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        "tip": "Logging level."
    }
    vid = {
        "title": "Video Shape",
        "name": "vid",
        "type": "group",
        "value": gui.fringes.T,
        "limits": [1, gui.fringes._Tmax],
        "tip": gui.fringes.__class__.T.__doc__,
        "children": [
            {
                "title": "Frames",
                "name": "T",
                "type": "int",
                "value": gui.fringes.T,
                "default": gui.fringes.T,
                "limits": (1 if gui.visibility == "Guru" else 3, gui.fringes._Tmax),
                "tip": gui.fringes.__class__.T.__doc__,
            },
            {
                "title": "Height",
                "name": "Y",
                "type": "int",
                "value": gui.fringes.Y,
                "default": gui.fringes.Y,  # defaults["Y"]
                "limits": (
                    1, min(gui.fringes._Ymax, gui.fringes._Pmax / gui.fringes.X) if gui.visibility == "Guru" else 7680
                ),
                "step": 10,
                "suffix": " px",
                "tip": gui.fringes.__class__.Y.__doc__,
            },
            {
                "title": "Width",
                "name": "X",
                "type": "int",
                "value": gui.fringes.X,
                "default": gui.fringes.X,  # defaults["X"]
                "limits": (
                    1, min(gui.fringes._Xmax, gui.fringes._Pmax / gui.fringes.Y) if gui.visibility == "Guru" else 7680
                ),
                "step": 10,
                "suffix": " px",
                "tip": gui.fringes.__class__.X.__doc__,
            },
            {
                "title": "Colors",
                "name": "C",
                "type": "int",
                "value": gui.fringes.C,
                "readonly": True,
                "visible": gui.visibility == "Guru",
                "tip": gui.fringes.__class__.C.__doc__,
            },
            {
                "title": "alpha",
                "name": "alpha",
                "type": "float",
                "value": gui.fringes.alpha,
                "default": gui.fringes.defaults["alpha"],
                "limits": (1, gui.fringes._alphamax),
                "step": 0.1,
                "decimals": gui.digits,
                "visible": gui.visibility == "Guru",
                "tip": gui.fringes.__class__.alpha.__doc__,
            },
            {
                "title": "Length",
                "name": "L",
                "type": "float",
                "value": gui.fringes.L,
                "decimals": gui.digits,
                "suffix": "px",
                "readonly": True,
                "tip": gui.fringes.__class__.L.__doc__,
            },

        ]
    }
    sys = {
            "title": "Coordinates",
            "name": "sys",
            "type": "group",
            "children": [
                {
                    "title": "Grid",
                    "name": "grid",
                    "type": "list",
                    "value": gui.fringes.grid,
                    "default": gui.fringes.defaults["grid"],
                    "limits": gui.fringes._grids,
                    "visible": gui.visibility == "Guru",
                    "tip": gui.fringes.__class__.grid.__doc__,
                },
                {
                    "title": "Angle",
                    "name": "angle",
                    "type": "float",
                    "value": gui.fringes.angle,
                    "default": gui.fringes.defaults["angle"],
                    "limits": (-360, 360),
                    "decimals": gui.digits,
                    "suffix": "°",
                    "visible": gui.visibility == "Guru",
                    "tip": gui.fringes.__class__.angle.__doc__,
                },
                {
                    "title": "Directions",
                    "name": "D",
                    "type": "int",
                    "value": gui.fringes.D,
                    "default": gui.fringes.defaults["D"],
                    "limits": (1, gui.fringes._Dmax),
                    "tip": gui.fringes.__class__.D.__doc__,
                    "children": [
                        {
                            "title": "Axis",
                            "name": "axis",
                            "type": "list",
                            "value": ("X", "Y")[gui.fringes.axis],
                            "default": ("X", "Y")[gui.fringes.defaults["axis"]],
                            "limits": ("X", "Y"),
                            "visible": gui.fringes.D == 1,
                            "tip": gui.fringes.__class__.axis.__doc__,
                        },
                    ],
                },
            ]
        }
    set = {
            "title": "Set",
            "name": "set",
            "type": "group",
            "children": [
                {
                    "title": "Sets",
                    "name": "K",
                    "type": "int",
                    "value": gui.fringes.K,
                    "default": gui.fringes.defaults["K"],
                    "limits": (1, gui.fringes._Kmax),
                    # 1 if gui.visibility != "Guru" else gui.fringes._Kmax
                    "tip": gui.fringes.__class__.K.__doc__,
                },
                {
                    "title": "Shifts",
                    "name": "N",
                    "type": "action",
                    "tip": "Reset values to defaults.",
                    "children": [
                        {
                            "name": "N" + str(d).translate(gui.sub) + ", " + str(k).translate(gui.sub),
                            "type": "int",
                            "value": gui.fringes._N[d, k],
                            "default": gui.fringes.Nmin if gui.fringes.FDM else gui.fringes.defaults["N"][0, 0],
                            "limits": (max(gui.fringes.Nmin, 1 if gui.visibility == "Guru" else 2 if gui.visibility == "Expert" else 3), gui.fringes._Nmax),
                            "tip": gui.fringes.__class__.N.__doc__,
                        } for d in range(gui.fringes.D) for k in range(gui.fringes.K)
                    ] if gui.visibility == "Guru" or gui.fringes.N.ndim > 1 else [  # todo: FDM: N_i
                        {
                            "name": "N" + str(k).translate(gui.sub),
                            "type": "int",
                            "value": gui.fringes.N[k],
                            "default": gui.fringes.Nmin if gui.fringes.FDM else gui.fringes.defaults["N"][0, 0],
                            "limits": (max(gui.fringes.Nmin, 1 if gui.visibility == "Guru" else 2 if gui.visibility == "Expert" else 3), gui.fringes._Nmax),
                            "tip": gui.fringes.__class__.N.__doc__,
                        } for k in range(gui.fringes.K)
                    ],
                },
                {
                    "title": "Wavelengths",
                    "name": "l",
                    "type": "action",
                    "tip": "Set optimal wavelengths automatically.",
                    "children": [
                        {
                            "title": "\u03BB" + str(d).translate(gui.sub) + ", " + str(k).translate(gui.sub),
                            "name": "l" + str(d).translate(gui.sub) + ", " + str(k).translate(gui.sub),
                            "type": "float",
                            "value": gui.fringes._l[d, k],
                            "default": None,  # gui.fringes.L ** (1 / (k + 1)),
                            "limits": (gui.fringes.lmin, None),
                            "decimals": gui.digits,
                            "tip": gui.fringes.__class__.l.__doc__,
                        } for d in range(gui.fringes.D) for k in range(gui.fringes.K)
                    ] if gui.visibility == "Guru" or gui.fringes.l.ndim > 1 else [
                        {
                            "title": "\u03BB" + str(k).translate(gui.sub),
                            "name": "l" + str(k).translate(gui.sub),
                            "type": "float",
                            "value": gui.fringes.l[k],
                            "default": None,  # gui.fringes.L ** (1 / (k + 1)),
                            "suffix": " px",
                            "limits": (gui.fringes.lmin, None),
                            "decimals": gui.digits,
                            "tip": gui.fringes.__class__.l.__doc__,
                        } for k in range(gui.fringes.K)
                    ],
                },
                {
                    "title": "Periods",
                    "name": "v",
                    "type": "action",
                    "tip": "Set optimal periods automatically.",
                    "children": [
                        {
                            "title": "\u03BD" + str(d).translate(gui.sub) + ", " + str(k).translate(gui.sub),
                            "name": "v" + str(d).translate(gui.sub) + ", " + str(k).translate(gui.sub),
                            "type": "float",
                            "value": gui.fringes._v[d, k],
                            "default": None,  # gui.fringes.L ** (1 - 1 / (k + 1)),
                            "limits": (0, gui.fringes.vmax),
                            "decimals": gui.digits,
                            "tip": gui.fringes.__class__.v.__doc__,
                        } for d in range(gui.fringes.D) for k in range(gui.fringes.K)] if gui.visibility == "Guru" or gui.fringes.v.ndim > 1 else [
                        {
                            "title": "\u03BD" + str(k).translate(gui.sub),
                            "name": "v" + str(k).translate(gui.sub),
                            "type": "float",
                            "value": gui.fringes.v[k],
                            "default": None,  # gui.fringes.L ** (1 - 1 / (k + 1)),
                            "limits": (0, gui.fringes.vmax),
                            "decimals": gui.digits,
                            "tip": gui.fringes.__class__.v.__doc__,
                        } for k in range(gui.fringes.K)
                    ],
                },
                {
                    "title": "Frequencies",
                    "name": "f",
                    "type": "action",
                    "visible": gui.visibility == "Guru",
                    "tip": "Reset values to defaults.",
                    "children": [
                        {
                            "name": "f" + str(d).translate(gui.sub) + ", " + str(k).translate(gui.sub),
                            "type": "float",
                            "value": gui.fringes._f[d, k],
                            "default": None if gui.fringes.FDM else 1,
                            "limits": (-gui.fringes.vmax, gui.fringes.vmax),
                            "decimals": gui.digits,
                            "readonly": gui.fringes.FDM and gui.fringes.static,
                            "tip": gui.fringes.__class__.f.__doc__,
                        } for d in range(gui.fringes.D) for k in range(gui.fringes.K)
                    ] if gui.visibility == "Guru" or gui.fringes.FDM or gui.fringes.f.ndim > 1 else [
                        {
                            "name": "f" + str(k).translate(gui.sub),
                            "type": "float",
                            "value": gui.fringes.f[k],
                            "default": 1,
                            "limits": (-gui.fringes.vmax, gui.fringes.vmax),
                            "decimals": gui.digits,
                            "tip": gui.fringes.__class__.f.__doc__,
                        } for k in range(gui.fringes.K)
                    ],
                },
                {
                    "title": "Reverse",
                    "name": "reverse",
                    "type": "bool",
                    "value": gui.fringes.reverse,
                    "default": gui.fringes.defaults["reverse"],
                    "tip": gui.fringes.__class__.reverse.__doc__,
                },
                {
                    "title": "Offset",
                    "name": "o",
                    "type": "float",
                    "value": gui.fringes.o / np.pi,
                    "default": gui.fringes.defaults["o"] / np.pi,
                    "limits": (-2, 2),
                    "step": 0.5,
                    "decimals": gui.digits,
                    "suffix": "\U0001D745",  # \U0001D70B
                    "tip": gui.fringes.__class__.o.__doc__,
                },
                {
                    "title": "\u03BB\u2098\u1D62\u2099",
                    "name": "lmin",
                    "type": "int",
                    "value": gui.fringes.lmin,
                    "default": gui.fringes.defaults["lmin"],
                    "limits": (4, gui.fringes.L),
                    "suffix": "px",
                    "tip": gui.fringes.__class__.lmin.__doc__,
                },
                {
                    "title": "\u03BD\u2098\u2090\u2093",
                    "name": "vmax",
                    "type": "float",
                    "value": gui.fringes.vmax,
                    # "default": gui.fringes.defaults["vmax"],
                    "decimals": gui.digits,
                    "readonly": True,
                    "visible": False,  # todo
                    "tip": gui.fringes.__class__.vmax.__doc__,
                },
                {
                    "title": "Range",
                    "name": "UMR",
                    "type": "float",
                    "value": gui.fringes.UMR.min(),
                    "decimals": gui.digits,
                    "suffix": " px",
                    "readonly": True,
                    "tip": gui.fringes.__class__.UMR.__doc__,
                },
            ]
        }
    val = {
            "title": "Values",
            "name": "val",
            "type": "group",
            "children": [
                {
                    "title": "Type",
                    "name": "dtype",
                    "type": "list",
                    "value": gui.fringes.dtype,
                    "default": gui.fringes.defaults["dtype"],
                    "limits": gui.fringes._dtypes,
                    "visible": gui.visibility == "Guru",
                    "tip": gui.fringes.__class__.dtype.__doc__,
                },
                {
                    "title": "I\u2098\u2090\u2093",
                    "name": "Imax",
                    "type": "int",
                    "value": gui.fringes.Imax,
                    "readonly": True,
                    "visible": gui.visibility == "Guru",
                    "tip": gui.fringes.__class__.Imax.__doc__,
                },
                {
                    "title": "Bias",
                    "name": "A",
                    "type": "float",
                    "value": gui.fringes.A,
                    "default": gui.fringes.defaults["A"],
                    "limits": (gui.fringes.B, min(gui.fringes.A, gui.fringes.Imax - gui.fringes.A)),
                    "decimals": gui.digits,
                    "tip": gui.fringes.__class__.A.__doc__,
                },
                {
                    "title": "Amplitude",
                    "name": "B",
                    "type": "float",
                    "value": gui.fringes.B,
                    "default": gui.fringes.defaults["B"],
                    "limits": (0, gui.fringes.Imax - gui.fringes.B),
                    "decimals": gui.digits,
                    "tip": gui.fringes.__class__.B.__doc__,
                },
                {
                    "title": "Exposure",
                    "name": "beta",
                    "type": "float",
                    "value": gui.fringes.beta,
                    "default": gui.fringes.defaults["beta"],
                    "limits": (0, 0.5),
                    "step": 0.1,
                    "decimals": gui.digits,
                    "tip": gui.fringes.__class__.beta.__doc__,
                },
                {
                    "title": "Visibility",
                    "name": "V",
                    "type": "float",
                    "value": gui.fringes.V,
                    "default": gui.fringes.defaults["V"],
                    "limits": (0, 1),
                    "step": 0.1,
                    "decimals": gui.digits,
                    "tip": gui.fringes.__class__.V.__doc__,
                },
                {
                    "title": "Gamma",
                    "name": "gamma",
                    "type": "float",
                    "value": gui.fringes.gamma,
                    "default": gui.fringes.defaults["gamma"],
                    "limits": (0, gui.fringes.__class__._gammamax),
                    "step": 0.1,
                    "decimals": gui.digits,
                    "tip": gui.fringes.__class__.gamma.__doc__,
                },
            ],
        }
    col = {
        "title": "Color",
        "name": "col",
        "type": "group",
        "visible": gui.visibility == "Guru",
        "expanded": np.all(gui.fringes.h != gui.fringes._hues[0]),
        "children": [
            {
                "title": "Averaging",
                "name": "M",
                "type": "float",
                "value": gui.fringes.M,
                "default": gui.fringes.defaults["M"],
                "limits": (1 / 255, gui.fringes._Mmax),
                "tip": gui.fringes.__class__.M.__doc__,
            },
            {
                "title": "Hues",
                "name": "H",
                "type": "int",
                "value": gui.fringes.H,
                "default": gui.fringes.defaults["H"],
                "limits": (1, gui.fringes._Hmax),
                "tip": gui.fringes.__class__.H.__doc__,
                "children": [
                    {
                        "name": "h" + str(h).translate(gui.sub),
                        "type": "color",
                        "value": gui.fringes.h[h],
                        "default": gui.fringes.defaults["h"][0],
                    } for h in range(gui.fringes.H)
                ],
            },
        ],
    }
    mux = {
            "title": "Multiplexing",
            "name": "mux",
            "type": "group",
            "visible": gui.visibility == "Guru",
            "expanded": gui.fringes.FDM or gui.fringes.SDM or gui.fringes.WDM,
            "tip": "Multiplexing method.",
            "children": [
                {
                    "title": "TDM",
                    "name": "TDM",
                    "type": "bool",
                    "value": gui.fringes.TDM,
                    "default": gui.fringes.defaults["TDM"],
                    "readonly": True,
                    "tip": gui.fringes.__class__.TDM.__doc__,
                },
                {
                    "title": "SDM",
                    "name": "SDM",
                    "type": "bool",
                    "value": gui.fringes.SDM,
                    "default": gui.fringes.defaults["SDM"],
                    "tip": gui.fringes.__class__.SDM.__doc__,
                },
                {
                    "title": "WDM",
                    "name": "WDM",
                    "type": "bool",
                    "value": gui.fringes.WDM,
                    "default": gui.fringes.defaults["WDM"],
                    "tip": gui.fringes.__class__.WDM.__doc__,
                },
                {
                    "title": "FDM",
                    "name": "FDM",
                    "type": "bool",
                    "value": gui.fringes.FDM,
                    "default": gui.fringes.defaults["FDM"],
                    "tip": gui.fringes.__class__.FDM.__doc__,
                    "children": [
                        {
                            "title": "Static",
                            "name": "static",
                            "type": "bool",
                            "value": gui.fringes.static,
                            "default": gui.fringes.defaults["static"],
                            "visible": gui.fringes.FDM,
                            "tip": gui.fringes.__class__.static.__doc__,
                        },
                    ]
                },
            ],
        }
    uwr = {
            "title": "Unwrapping",
            "name": "uwr",
            "type": "group",
            "children": [
                {
                    "title": "Method",
                    "name": "PU",
                    "type": "str",
                    "value": gui.fringes.PU,
                    "readonly": True,
                    "tip": gui.fringes.__class__.PU.__doc__,
                },
                {
                    "title": "Mode",
                    "name": "mode",
                    "type": "list",
                    "value": gui.fringes.mode,
                    "default": gui.fringes.defaults["mode"],
                    "limits": gui.fringes._modes,
                    "visible": gui.visibility == "Guru",
                    "tip": gui.fringes.__class__.mode.__doc__,
                },
                {
                    "title": "V\u2098\u1D62\u2099",  # todo: Vmin
                    "name": "Vmin",
                    "type": "float",
                    "value": gui.fringes.Vmin,
                    "default": gui.fringes.defaults["Vmin"],
                    "limits": (0, 1),
                    "step": 0.1,
                    "decimals": gui.digits,
                    "visible": gui.visibility == "Guru",
                    "tip": gui.fringes.__class__.Vmin.__doc__
                },
                {
                    "title": "Verbose",
                    "name": "verbose",
                    "type": "bool",
                    "value": gui.fringes.verbose,
                    "default": gui.fringes.defaults["verbose"],
                    "tip": gui.fringes.__class__.verbose.__doc__,
                },
            ]
        }
    quali = {
            "title": "Quality",
            "name": "quali",
            "type": "group",
            "visible": gui.visibility in ["Expert", "Guru"],
            "children": [
                {
                    "title": "Efficiency",  # \u03B7
                    "name": "eta",
                    "type": "float",
                    "value": gui.fringes.eta.max(),
                    "decimals": gui.digits,
                    "readonly": True,
                    "tip": gui.fringes.__class__.eta.__doc__,
                },
                {
                    "title": "Dark Noise",
                    "name": "dark",
                    "type": "float",
                    "value": gui.fringes.dark,
                    "defaults": gui.fringes.defaults["dark"],
                    "limits": (0, np.sqrt(gui.fringes.Imax)),
                    "step": 0.1,
                    "decimals": gui.digits,
                    "visible": gui.visibility == "Guru",
                    "tip": gui.fringes.__class__.dark.__doc__
                },
                {
                    "title": "Quantization Noise",
                    "name": "quant",
                    "type": "float",
                    "value": gui.fringes.quant,
                    "readonly": True,
                    "decimals": gui.digits,
                    "tip": gui.fringes.__class__.quant.__doc__
                },
                {
                    "title": "Shot Noise",
                    "name": "shot",
                    "type": "float",
                    "value": gui.fringes.shot,
                    "defaults": 0,
                    "decimals": gui.digits,
                    "readonly": True,
                    "visible": gui.visibility == "Guru",
                    "tip": gui.fringes.__class__.shot.__doc__
                },
                {
                    "title": "Uncertainty\u2098\u1D62\u2099",
                    "name": "u",
                    "type": "float",
                    "value": gui.fringes.u,
                    "decimals": gui.digits,
                    "suffix": "px",
                    "readonly": True,
                    "tip": gui.fringes.__class__.u.__doc__
                },
                {
                    "title": "Dynamic Range",
                    "name": "DR",
                    "type": "float",
                    "value": gui.fringes.DRdB.max(),
                    "decimals": gui.digits,
                    "suffix": "dB",
                    "readonly": True,
                    "tip": gui.fringes.__class__.DRdB.__doc__,
                },
            ]
        }

    gui.params = [vis, log, vid, sys, set, val, col, mux, uwr, quali]
