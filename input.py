import xml.etree.ElementTree as ET
import dill
import material as mat_module
import sys
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import math
from pathlib import Path
import os


class Layer:
    def __init__(self, layerelem, root):
        """
creates a Layer objects that holds various information about the TPS layer
        :param layerelem: layer element from xml tree
        :param root: xml root element
        """

        # Check if layer is ablative
        self.ablative = True if layerelem.attrib["number"] == "1" and layerelem.find("ablative").text == "True" \
            else False
        if layerelem.find("ablative") is not None:
            if int(layerelem.attrib["number"]) > 1 and layerelem.find("ablative").text == "True":
                raise ValueError("Material at layer no. %i cannot be ablative. Only the first layer can."
                                 % int(layerelem.attrib["number"]))

        if self.ablative:
            ablative_vals = {
                "pressure": float(root.find("options").find("ambient").find("pressure").text),
                "atmosphere": root.find("options").find("ambient").find("atmosphere").text
            }
        else:
            ablative_vals = None

        # Check if layer is corrugated
        if layerelem.find("corrugated") is not None:
            if layerelem.find("corrugated").text == "True":
                self.corrugated = True
                mat_core_name = layerelem.find("material_core").text
                mat_web_name = layerelem.find("material_web").text
                corrugated_vals = {
                    "mat_core": find_existing(mat_core_name, "material"),
                    "mat_web": find_existing(mat_web_name, "material"),
                    "dc": float(layerelem.find("thickness").text),
                    "dw": float(layerelem.find("web_thickness").text),
                    "p": float(layerelem.find("half_cell_length").text),
                    "theta": math.radians(float(layerelem.find("corrugation_angle").text))
                }
            else:
                self.corrugated = False
                corrugated_vals = None
        else:
            self.corrugated = False
            corrugated_vals = None

        # If ablative, check whether bprime or hg shall be read from file
        if self.ablative:
            ablative_vals["bprime"] = find_existing(layerelem.find("bprime").text) if layerelem.find("bprime") is not None else None
            ablative_vals["hgas"] = find_existing(layerelem.find("hgas").text) if layerelem.find("hgas") is not None else None

        # Check if layer is not corrugated and ablative
        if self.ablative and self.corrugated:
            raise ValueError("Material cannot be ablative and corrugated.")

        # Construct material
        if not self.corrugated:
            matname = layerelem.find("material").text
            mat_path = find_existing(matname, "material")
            self.material = readMaterial(mat_path, self.ablative, self.corrugated,
                                         ablative_vals, corrugated_vals)
        else:
            self.material = mat_module.createMaterial("Corrugated_Mat", ablative=False, corrugated=True,
                                                      corrugated_vals=corrugated_vals)

        self.thickness = float(layerelem.find("thickness").text)
        self.firstcell = float(layerelem.find("firstcell").text) if layerelem.find(
            "firstcell") is not None else self.thickness / 100.0
        self.maxgrowth = float(layerelem.find("maxgrowth").text) if layerelem.find("maxgrowth") is not None else 1.1
        self.number = int(layerelem.attrib["number"]) - 1


class Input:
    def __init__(self, xmlfile):
        """
reads the input xml file and stores the information
        :param xmlfile: path to xml file
        """
        # Read xml file
        tree = ET.parse(find_existing(xmlfile))

        root = tree.getroot()

        # Read information layer by layer
        layerelems = root.find("layers")
        numLayers = len(layerelems.findall("layer"))
        if numLayers == 0:
            raise ValueError("You must specify at least one layer.")
        else:
            self.layers = [None] * numLayers
            for layerelem in layerelems:
                layer = Layer(layerelem, root)
                self.layers[int(layerelem.attrib["number"]) - 1] = layer

        # Boundary conditions
        self.BCfrontType = root.find("options").find("BCs").find("front").attrib["type"]
        if self.BCfrontType in ("heatflux",):
            try:
                value = float(root.find("options").find("BCs").find("front").find("value").text)
                self.BCfrontValue = lambda t: value
            except ValueError:
                # Open file
                csv_file = root.find("options").find("BCs").find("front").find("value").text
                csv_path = find_existing(csv_file)
                with open(csv_path) as f:
                    data = pd.read_csv(f, sep=';', decimal='.', header=0)
                self.BCfrontValue = interp1d(data.values[:, 0], data.values[:, 1], kind='linear',
                                             fill_value=0.0, bounds_error=False)
            if self.layers[0].ablative:
                try:
                    value = float(root.find("options").find("BCs").find("front").find("coef").text)
                    self.aerocoef = lambda t: value
                except ValueError:
                    # Open file
                    csv_file = root.find("options").find("BCs").find("front").find("coef").text
                    csv_path = find_existing(csv_file)
                    with open(csv_path) as f:
                        data = pd.read_csv(f, sep=';', decimal='.', header=0)
                    self.aerocoef = interp1d(data.values[:, 0], data.values[:, 1], kind='linear',
                                             fill_value=0.0, bounds_error=False)

        elif self.BCfrontType in ("aerodynamic",):
            try:
                value = float(root.find("options").find("BCs").find("front").find("value").text)
                self.BCfrontValue = lambda t: value
            except ValueError:
                # Open file
                csv_file = root.find("options").find("BCs").find("front").find("value").text
                csv_path = find_existing(csv_file)
                with open(csv_path) as f:
                    data = pd.read_csv(f, sep=';', decimal='.', header=0)
                self.BCfrontValue = interp1d(data.values[:, 0], data.values[:, 1], kind='linear',
                                             fill_value=0.0, bounds_error=False)
            self.BLEdgeT_at_t = float(root.find("options").find("BCs").find("front").find("BL_Edge_Temperature").text)
            self.SurfT_at_t = float(root.find("options").find("BCs").find("front").find("Surface_Temperature").text)
            self.tforT = float(root.find("options").find("BCs").find("front").find("Time_of_Temperature").text)
            mat = self.layers[0].material
            self.aerocoef = lambda t: self.BCfrontValue(self.tforT) / (
                        mat.hatmo(self.BLEdgeT_at_t) - mat.hatmo(self.SurfT_at_t))
            self.BLedge_h = lambda t: self.BCfrontValue(t) / self.aerocoef(t) + mat.hatmo(self.SurfT_at_t)

        elif self.BCfrontType == "recovery_enthalpy":
            if not self.layers[0].ablative:
                raise ValueError("Reovery enthalpy BC not implemented for non-ablative cases.")
            else:
                try:
                    value = float(root.find("options").find("BCs").find("front").find("value").text)
                    self.BLedge_h = lambda t: value
                except ValueError:
                    # Open file
                    csv_file = root.find("options").find("BCs").find("front").find("value").text
                    csv_path = find_existing(csv_file)
                    with open(csv_path) as f:
                        data = pd.read_csv(f, sep=';', decimal='.', header=0)
                    self.BLedge_h = interp1d(data.values[:, 0], data.values[:, 1], kind='linear',
                                             fill_value=0.0, bounds_error=False)

                try:
                    value = float(root.find("options").find("BCs").find("front").find("coef").text)
                    self.aerocoef = lambda t: value
                except ValueError:
                    # Open file
                    csv_file = root.find("options").find("BCs").find("front").find("coef").text
                    csv_path = csv_file
                    with open(csv_path) as f:
                        data = pd.read_csv(f, sep=';', decimal='.', header=0)
                    self.aerocoef = interp1d(data.values[:, 0], data.values[:, 1], kind='linear',
                                             fill_value=0.0, bounds_error=False)
        else:
            raise ValueError("Unsupported front BC %s" % self.BCfrontType)
        self.BCbackType = root.find("options").find("BCs").find("back").attrib["type"]
        if self.BCbackType != "adiabatic":
            raise ValueError("Unsupported back BC %s" % self.BCbackType)

        # Time stepping
        if root.find("options").find("time").find("file") is not None:
            f = root.find("options").find("time").find("file").text
            f_path = find_existing(f)
            self.ts = pd.read_csv(f_path, sep=';', decimal='.', header=0).to_numpy().flatten()
            self.tStart = self.ts[0]
            self.tEnd = self.ts[1]
            self.ts = self.ts[1:]
        else:
            self.tStart = float(root.find("options").find("time").find("start").text)
            self.tEnd = float(root.find("options").find("time").find("end").text)
            self.tDelta = float(root.find("options").find("time").find("delta").text)
            self.ts = np.arange(self.tStart + self.tDelta, self.tEnd + 1e-8, self.tDelta)
        self.write_step = int(root.find("options").find("time").find("write_every").text)

        # Initialization
        self.initType = root.find("options").find("init").attrib["type"]
        self.initValue = float(root.find("options").find("init").find("value").text)

        # Turbulent flow
        self.Tamb = float(root.find("options").find("ambient").find("temperature").text)
        self.turbflow = bool(root.find("options").find("ambient").find("turbulent_flow").text)


def readMaterial(matname, ablative, corrugated, ablative_vals=None, corrugated_vals=None):
    # Read .matp file if it was specified
    if matname.name[-5:] == ".matp":
        with open(matname, 'rb') as matpfile:
            material = dill.load(matpfile)
        # if it is an ablative material check whether the data was generated for the right pressure and atmosphere
        if type(material) is mat_module.AblativeMaterial:
            if material.pressure == ablative_vals["pressure"] and \
                    material.atmosphere == ablative_vals["atmosphere"]:
                print("Using %s" % matname)  # nothing has changed, continue
            else:
                # if not, run material generation again
                print("Pressure or atmosphere not matching. Constructing material again...")
                material = mat_module.createMaterial(matname.parent, ablative=True, corrugated=False,
                                                     pressure=ablative_vals["pressure"],
                                                     atmosphere=ablative_vals["atmosphere"])
        else:
            print("Using %s" % matname)  # non-ablative material has no problem specific options, continue
    else:
        # if no file was specified, generate a material file and store it
        if ablative:
            material = mat_module.createMaterial(matname, ablative=True, corrugated=False,
                                                 pressure=ablative_vals["pressure"],
                                                 atmosphere=ablative_vals["atmosphere"],
                                                 bprime=ablative_vals["bprime"],
                                                 hgas=ablative_vals["hgas"])
        else:
            material = mat_module.createMaterial(matname, ablative=False, corrugated=corrugated,
                                                 corrugated_vals=corrugated_vals)

    return material


directory_map = {"material": "Data/Materials"}


def find_existing(name, kind=None):
    # Check if absolute path
    if Path(name).root != '' and Path.exists(Path(name)):
        return Path(name)
    # Check if in cwd
    elif Path.exists(Path.joinpath(Path.cwd(), name)):
        return Path.joinpath(Path.cwd(), name)

    # For further checks environment variable needs to be set
    hs_dir = os.getenv("HOTSTARSHIP_DIR")
    if hs_dir is None:
        raise ValueError(
            "Environment variable HOTSTARSHIP_DIR is not set and directory %s could not be located." % name)

    # Check if in HS_DIR
    if Path.exists(Path.joinpath(Path(hs_dir), name)):
        return Path.joinpath(Path(hs_dir), name)

    try:
        directory_map[kind]
    except KeyError:
        raise ValueError("Directory %s could not be located." % name)

    # Check if in Hot-STARSHIP standard directory
    if Path.exists(Path.joinpath(Path(hs_dir), directory_map[kind], name)):
        return Path.joinpath(Path(hs_dir), directory_map[kind], name)
    # Check if in Hot-STARSHIP standard directory and ".matp" exists in same name folder
    elif kind == "material" and name[-5:] == ".matp" and Path.exists(
            Path.joinpath(Path(hs_dir), directory_map[kind], name[:-5], name)):
        return Path.joinpath(Path(hs_dir), directory_map[kind], name[:-5], name)
    else:
        raise ValueError("Directory %s could not be located." % name)


if __name__ == "__main__":
    input = Input("/Users/nils/PycharmProjects/H_STARSHIP/Input/input.xml")
