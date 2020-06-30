import xml.etree.ElementTree as ET
import dill
import material
import sys
from scipy.interpolate import interp1d
import pandas as pd


class Layer:
    def __init__(self, layerelem, root):
        """
creates a Layer objects that holds various information about the TPS layer
        :param layerelem: layer element from xml tree
        :param root: xml root element
        """
        matname = layerelem.find("material").text
        self.ablative = True if layerelem.attrib["number"] == "1" and layerelem.find("ablative").text == "True" \
            else False
        if layerelem.find("ablative") is not None:
            if int(layerelem.attrib["number"]) > 1 and layerelem.find("ablative").text == "True":
                raise ValueError("Material at layer no. %i cannot be ablative. Only the first layer can."
                                 % int(layerelem.attrib["number"]))
        # Read .matp file if it was specified
        if matname[-5:] == ".matp":
            with open(matname, 'rb') as matpfile:
                self.material = dill.load(matpfile)
            # if it is an ablative material check whether the data was generated for the right pressure and atmosphere
            if type(self.material) is material.AblativeMaterial:
                if self.material.pressure == float(root.find("options").find("ambient").find("pressure").text) and \
                        self.material.atmosphere == root.find("options").find("ambient").find("atmosphere").text:
                    print("Using %s" % matname)  # nothing has changed, continue
                else:
                    # if not, run material generation again
                    self.material = material.createMaterial(matname, ablative=True,
                                            pressure=float(root.find("options").find("ambient").find("pressure").text),
                                            atmosphere=root.find("options").find("ambient").find("atmosphere").text)
            else:
                print("Using %s" % matname)  # non-ablative material has no problem specific options, continue
        else:
            # if no file was specified, generate a material file and store it
            ablativeLayer = True if layerelem.attrib["number"] == "1" and self.ablative else False
            self.material = material.createMaterial(matname, ablative=ablativeLayer,
                                        pressure=float(root.find("options").find("ambient").find("pressure").text),
                                        atmosphere=root.find("options").find("ambient").find("atmosphere").text)

        self.thickness = float(layerelem.find("thickness").text)
        self.firstcell = float(layerelem.find("firstcell").text) if layerelem.find("firstcell") is not None else self.thickness/100.0
        self.maxgrowth = float(layerelem.find("maxgrowth").text) if layerelem.find("maxgrowth") is not None else 1.1
        self.number = int(layerelem.attrib["number"])-1


class Input:
    def __init__(self, xmlfile):
        """
reads the input xml file and stores the information
        :param xmlfile: path to xml file
        """
        # Read xml file
        tree = ET.parse(xmlfile)

        root = tree.getroot()

        # Read information layer by layer
        layerelems = root.find("layers")
        numLayers = len(layerelems.findall("layer"))
        if numLayers == 0:
            raise ValueError("You must specify at least one layer.")
        else:
            self.layers = [None]*numLayers
            for layerelem in layerelems:
                layer = Layer(layerelem, root)
                self.layers[int(layerelem.attrib["number"])-1] = layer

        # Boundary conditions
        self.BCfrontType = root.find("options").find("BCs").find("front").attrib["type"]
        if self.BCfrontType in ("heatflux",):
            try:
                value = float(root.find("options").find("BCs").find("front").find("value").text)
                self.BCfrontValue = lambda t: value
            except ValueError:
                # Open file
                csv_file = root.find("options").find("BCs").find("front").find("value").text
                with open(csv_file) as f:
                    data = pd.read_csv(f, sep=';', decimal='.', header=0)
                self.BCfrontValue = interp1d(data.values[:, 0], data.values[:, 1], kind='linear')
                if self.layers[0].ablative:
                    self.aerocoef = float(root.find("options").find("BCs").find("front").find("coef").text)
        elif self.BCfrontType in ("aerodynamic",):
            try:
                value = float(root.find("options").find("BCs").find("front").find("value").text)
                self.BCfrontValue = lambda t: value
            except ValueError:
                # Open file
                csv_file = root.find("options").find("BCs").find("front").find("value").text
                with open(csv_file) as f:
                    data = pd.read_csv(f, sep=';', decimal='.', header=0)
                self.BCfrontValue = interp1d(data.values[:, 0], data.values[:, 1], kind='linear')
            self.BLEdgeT_at_t = float(root.find("options").find("BCs").find("front").find("BL_Edge_Temperature").text)
            self.SurfT_at_t = float(root.find("options").find("BCs").find("front").find("Surface_Temperature").text)
            self.tforT = float(root.find("options").find("BCs").find("front").find("Time_of_Temperature").text)
            mat = self.layers[0].material
            self.aerocoef = self.BCfrontValue(self.tforT)/(mat.hatmo(self.BLEdgeT_at_t)-mat.hatmo(self.SurfT_at_t))
            self.BLedge_h = lambda t: self.BCfrontValue(t)/self.aerocoef + mat.hatmo(self.SurfT_at_t)
        else:
            raise ValueError("Unsupported front BC %s" % self.BCfrontType)
        self.BCbackType = root.find("options").find("BCs").find("back").attrib["type"]
        if self.BCbackType != "adiabatic":
            raise ValueError("Unsupported back BC %s" % self.BCbackType)

        # Time stepping
        self.tStart = float(root.find("options").find("time").find("start").text)
        self.tEnd = float(root.find("options").find("time").find("end").text)
        self.tDelta = float(root.find("options").find("time").find("delta").text)
        self.write_step = int(root.find("options").find("time").find("write_every").text)

        # Initialization
        self.initType = root.find("options").find("init").attrib["type"]
        self.initValue = float(root.find("options").find("init").find("value").text)

        # Turbulent flow
        self.Tamb = float(root.find("options").find("ambient").find("temperature").text)
        self.turbflow = bool(root.find("options").find("ambient").find("turbulent_flow").text)


if __name__ == "__main__":

    input = Input("/Users/nils/PycharmProjects/H_STARSHIP/Input/input.xml")