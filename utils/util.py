import xmltodict

def parse_xml(xml_path):
    xml_data = None
    with open(xml_path) as fd:
        xml_data = xmltodict.parse(fd.read())
        xml_data = xml_data["annotation"]

    if "object" not in xml_data:
        xml_data["object"] = []
    elif type(xml_data["object"]) != list:
        xml_data["object"] = [xml_data["object"]]

    return xml_data

