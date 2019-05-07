import os

from lxml import etree


class XmlCreator:

    def __init__(self, dst):
        self.dst = dst

    def create_pascal_voc_xml(self, filename, template):
        filename = os.path.join(self.dst, filename)
        return PascalVocXml(filename).fill_template(template)


class PascalVocXml:

    def __init__(self, filename):
        self.dst = filename

        self.root = etree.Element('annotation')
        self.tree = etree.ElementTree(self.root)
        self.filename = etree.SubElement(self.root, 'filename')
        self.folder = etree.SubElement(self.root, 'folder')
        self.segmented = etree.SubElement(self.root, 'segmented')

        self.size = etree.SubElement(self.root, 'size')
        self.depth = etree.SubElement(self.size, 'depth')
        self.height = etree.SubElement(self.size, 'height')
        self.width = etree.SubElement(self.size, 'width')

        self.source = etree.SubElement(self.root, 'source')
        self.annotation = etree.SubElement(self.source, 'annotation')
        self.database = etree.SubElement(self.source, 'database')
        self.image = etree.SubElement(self.source, 'image')

    def fill_template(self, template: dict):
        self.filename.text = template['filename']
        self.folder.text = template['folder']
        self.segmented.text = template['segmented']
        self.depth.text = template['depth']
        self.height.text = template['height']
        self.width.text = template['width']

        self.annotation.text = template['annotation']
        self.database.text = template['database']
        self.image.text = template['image']
        return self

    def add_objects(self, objects: list):
        for obj in objects:
            object_root = etree.SubElement(self.root, 'object')

            etree.SubElement(object_root, 'name').text = obj['name']

            bndbox = etree.SubElement(object_root, 'bndbox')
            etree.SubElement(bndbox, 'xmax').text = obj['xmax']
            etree.SubElement(bndbox, 'xmin').text = obj['xmin']
            etree.SubElement(bndbox, 'ymax').text = obj['ymax']
            etree.SubElement(bndbox, 'ymin').text = obj['ymin']

            etree.SubElement(object_root, 'difficult').text = '0'
            etree.SubElement(object_root, 'pose').text = 'Unspecified'
            point = etree.SubElement(object_root, 'point')

            etree.SubElement(point, 'x')
            etree.SubElement(point, 'y')

    def save(self):
        with open(self.dst + '.xml', 'wb') as f:
            f.write(etree.tostring(self.tree))
        # outFile = open(self.dst + '.xml', 'w')
        # self.tree.write(outFile)
