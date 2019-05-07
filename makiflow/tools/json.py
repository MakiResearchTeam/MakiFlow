import json


class JsonParser:

    def __init__(self):
        self.result_list = []

    def parse_coco_json(self, to_json_path, channels=3):
        """
        Parse the single json file from CocoJson to [{file, [bboxes]}]
        :param channels: count of channels in Image
        :param to_json_path: path to json.json file
        :return: dict with params of json.json
        """
        result_list = []
        result = dict()
        with open(to_json_path) as json_file:
            json_str = json_file.read()

        json_str = json.loads(json_str)

        info_images = json_str['images']
        info_images = sorted(info_images, key=lambda image: image['id'])

        info_bboxes = json_str['annotations']
        info_bboxes = sorted(info_bboxes, key=lambda info_bbox: info_bbox['image_id'])

        raw_categories = json_str['categories']
        categories = {}
        for rc in raw_categories:
            categories[rc['id']] = rc['name']

        for img in info_images:
            id = img['id']
            result['filename'] = img['file_name']
            result['size'] = (img['width'], img['height'], channels)

            objects_list = []
            for i in range(len(info_bboxes)):
                if id != info_bboxes[0]['image_id']:
                    break

                info_bbox = info_bboxes.pop(0)
                name = categories[info_bbox['category_id']]
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
        self.result_list = result_list
        return result_list

    def get_last_results(self):
        return self.result_list
