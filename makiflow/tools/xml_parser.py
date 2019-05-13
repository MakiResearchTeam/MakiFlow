import os

from lxml import etree
from tqdm import tqdm


class XmlParser:
    """ Used for taking data from Pascal dataset xml. Make sure you have the same data format
    in case you want to use it.
    """
    def __init__(self):
        self.result_list = list()
        self.classes = set()
        self.classes_count = {}

    def parse_all_in_dict(self, source_path, num_files=None):
        """
        Parse all files in directory
        :param source_path: path to folder, what contains the target xml files
        :param num_files: TODO
        :return: list of dictionaries with params of xml
        """
        if num_files is not None:
            
            i = 0
            self.result_list = list()
            for root_dir, _, files in os.walk(source_path):
                for file in tqdm(files):
                    res = self.parse_xml(os.path.join(root_dir, file))
                    self.result_list.append(res)
                    i += 1
                    if i > num_files:
                        break
            return self.result_list
        
        else:
            
            self.result_list = list()
            for root_dir, _, files in os.walk(source_path):
                for file in tqdm(files):
                    res = self.parse_xml(os.path.join(root_dir, file))
                    self.result_list.append(res)
            return self.result_list
            

    def parse_xml(self, to_xml_path):
        """
        Parse the single xml file
        :param to_xml_path: path to xml file
        :return: dict with params of xml
        """
        result = dict()
        with open(to_xml_path) as xml_file:
            xml = xml_file.read()

        root = etree.fromstring(xml)
        filename = root.xpath('filename')
        result['filename'] = filename[0].text
        
        folder = root.xpath('folder')
        result['folder'] = folder[0].text

        object_list = list()
        objects = root.xpath('object')
        for obj in objects:
            name = obj.xpath('name')[0].text
            box = [float(obj.xpath('bndbox/xmin')[0].text), float(obj.xpath('bndbox/ymin')[0].text),
                   float(obj.xpath('bndbox/xmax')[0].text), float(obj.xpath('bndbox/ymax')[0].text)]
            
            # Add category to the set of classes
            self.classes.add(name)
            try:
                self.classes_count[name] += 1
            except:
                self.classes_count[name] = 1
                    
                    
            object_list.append({
                'name': name,
                'box': box
            })
        result['objects'] = object_list

        depth = int(root.xpath('size/depth')[0].text)
        width = int(root.xpath('size/width')[0].text)
        height = int(root.xpath('size/height')[0].text)
        result['size'] = (depth, width, height)

        return result

    def get_last_results(self):
        return self.result_list


if __name__ == "__main__":
    parser = XmlParser()
    print(parser.parse_all_in_dict(source_path='xml_source'))