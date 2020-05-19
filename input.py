import xml.etree.ElementTree as ET
import dill
import material
import sys


class Layer:
    def __init__(self, layerelem, root):

        matname = layerelem.find("material").text
        # Read .matp file if it was specified
        if matname[-5:] == ".matp":
            with open(matname, 'rb') as matpfile:
                self.material = dill.load(matpfile)
            # if it is an ablative material check whether the data was generated for the right pressure and atmosphere
            if type(self.material) is material.AblativeMaterial:
                if self.material.pressure == float(root.find("options").find("ambient").find("pressure").text) and \
                        self.material.atmosphere == root.find("options").find("ambient").find("atmosphere").text:
                    pass  # nothing has changed, continue
                else:
                    # if not, run material generation again
                    self.material = material.createMaterial(matname, ablative=True,
                                            pressure=float(root.find("options").find("ambient").find("pressure").text),
                                            atmosphere=root.find("options").find("ambient").find("atmosphere").text)
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

        # Read xml file
        tree = ET.parse(xmlfile)

        root = tree.getroot()

        # Read information layer by layer
        self.ablative = bool(root.find("layers").findall("layer")[0].find("ablative").text)
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
            self.BCfrontValue = float(root.find("options").find("BCs").find("front").find("value").text)

        # Time stepping
        self.tStart = float(root.find("options").find("time").find("start").text)
        self.tEnd = float(root.find("options").find("time").find("end").text)
        self.tDelta = float(root.find("options").find("time").find("delta").text)

        # Initialization
        self.initType = root.find("options").find("init").attrib["type"]
        self.initValue = float(root.find("options").find("init").find("value").text)


if __name__ == "__main__":

    input = Input("/Users/nils/PycharmProjects/H_STARSHIP/Input/input.xml")