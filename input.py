import xml.etree.ElementTree as ET
import dill
import material

class Layer:
    def __init__(self, layerelem, root):

        matname = layerelem.find("material").text
        if matname[-6:-1] == ".matp":
            self.material = dill.load(matname)
            if type(self.material) is material.AblativeMaterial:
                if self.material.pressure == float(root.find("options").find("ambient").find("pressure").text) and \
                        self.material.atmosphere == root.find("options").find("ambient").find("atmosphere").text:
                    pass # nothing has changed, continue
                else:
                    self.material = material.createMaterial(matname, ablative=True,
                                            pressure=float(root.find("options").find("ambient").find("pressure").text),
                                            atmosphere=root.find("options").find("ambient").find("atmosphere").text)
        else:
            ablative = True if layerelem.attrib["number"] == "1" and bool(layerelem.find("ablative").text) else False
            self.material = material.createMaterial(matname, ablative=ablative,
                                        pressure=float(root.find("options").find("ambient").find("pressure").text),
                                        atmosphere=root.find("options").find("ambient").find("atmosphere").text)

        self.thickness = float(layerelem.find("thickness").text)
        self.firstcell = float(layerelem.find("firstcell").text) if layerelem.find("firstcell") is not None else self.thickness/100.0
        self.maxgrowth = float(layerelem.find("maxgrowth").text) if layerelem.find("maxgrowth") is not None else 1.1


class Input:
    def __init__(self, xmlfile):
        tree = ET.parse(xmlfile)

        root = tree.getroot()

        layerelems = root.find("layers")

        numLayers = len(layerelems.findall("layer"))
        if numLayers == 0:
            raise ValueError("You must specify at least one layer.")
        else:
            self.layers = [None]*numLayers
            for layerelem in layerelems:
                layer = Layer(layerelem, root)
                self.layers[int(layerelem.attrib["number"])-1] = layer


if __name__ == "__main__":

    input = Input("/Users/nils/PycharmProjects/H_STARSHIP/Input/input.xml")