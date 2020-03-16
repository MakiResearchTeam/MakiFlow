# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import ujson
from tqdm import tqdm

class JsonParser:

    def __init__(self):
        self.result_list = []
        self.classes = set()
        self.classes_count = {}

    def parse_coco_json(self, to_json_path, channels=3, num_files=None):
        """
        Parse the single json file from CocoJson to [{file, [bboxes]}]
        :param channels: count of channels in Image
        :param to_json_path: path to json.json file
        :return: dict with params of json.json
        """
        result_list = []
        result = dict()
        json_file = open(to_json_path)

        json_str = ujson.load(json_file)
        json_file.close()

        info_images = json_str['images']
        info_images = sorted(info_images, key=lambda image: image['id'])

        info_bboxes = json_str['annotations']
        info_bboxes = sorted(info_bboxes, key=lambda info_bbox: info_bbox['image_id'])

        raw_categories = json_str['categories']
        categories = {}
        for rc in raw_categories:
            categories[rc['id']] = rc['name']
        
        ii = 0
        if num_files is not None:
            num_iterations = num_files
            print('OK')
        else:
            num_iterations = len(info_images)
            print('NO OK')
        iterator = tqdm(info_images)
        for img in iterator:
            id = img['id']
            result['filename'] = img['file_name']
            result['size'] = (img['width'], img['height'], channels)

            objects_list = []
            for i in range(len(info_bboxes)):
                if id != info_bboxes[0]['image_id']:
                    break

                info_bbox = info_bboxes.pop(0)
                name = categories[info_bbox['category_id']]
                
                # Add category to the set of classes
                self.classes.add(name)
                try:
                    self.classes_count[name] += 1
                except:
                    self.classes_count[name] = 1
                
                x1 = info_bbox['bbox'][0]
                y1 = info_bbox['bbox'][1]
                x2 = x1 + info_bbox['bbox'][2]
                y2 = y1 + info_bbox['bbox'][3]
                box = [x1, y1, x2, y2]
                objects_list.append({
                    'name': name,
                    'box': box
                })
            result['objects'] = objects_list
            result_list.append(result)
            result = dict()
            
            ii += 1
            if ii > num_iterations:
                iterator.close()
                break
            
            
        self.result_list = result_list
        return result_list

    def get_last_results(self):
        return self.result_list
