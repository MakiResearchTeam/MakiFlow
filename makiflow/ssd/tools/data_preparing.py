from __future__ import absolute_import
from makiflow.tools.json_parser import JsonParser
from makiflow.tools.xml_parser import XmlParser
from tqdm import tqdm


class DataFormat:
    COCO = 0
    PascalVOC = 1

class DataPreparator:
    def __init__(self, data_format, path_to_annotations, path_to_data, num_files=None):
        """
        Parameters
        ----------
        data_format : int
            Responsible for the data format: COCO or PascalVOC. 0 - COCO, 1 - PascalVOC. Use DataFormat class' constants for
            this purpose: DataFormat.COCO, DataFormat.PascalVOC.
        path_to_annotations : string
            Path to folder where annotations lie if data format is PascalVOC like or path to json with annotations if data format
            is COCO like.
        path_to_data : string
            Path to folder where images lie.
        num_files : int
            Number of annotations (and images) to load later. Leave it None if you want to load all the data.
        """
        self.data_format = data_format
        if data_format == DataFormat.COCO:
            self.parser = JsonParser()
        elif data_format == DataFormat.PascalVoc:
            self.parser = XmlParser()
        else:
            raise ValueError('Unknown data format: {}'.format(str(data_format)))
        
        self.path_to_annotations = path_to_annotations
        self.path_to_data = path_to_data
        self.num_files = num_files
        
    
    def __load_annotations(self):
        print('Loading annotations...')
        if self.data_format == DataFormat.COCO:
            self.annotation_dict = self.parser.parse_coco_json(self.path_to_annotations, self.num_files)
        elif self.data_format == DataFormat.PascalVOC:
            self.annotation_dict = self.parser.parse_all_in_dict(self.path_to_annotations, self.num_files)
        print('Annotations loaded.')
    
    
    def __load_images(self):
        print('Loading images...')
        self.images = []
        self.bboxes = []
        for annotation in tqdm(self.annotation_dict):
            image = cv2.imread('/mnt/data/coco_set/train2017/{}'.format(annotation['filename']))
            
            
        