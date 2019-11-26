import os
import random
import re
from threading import Thread

import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class ImageFactoryForSSD:
    """
    This class must create threads for all set of options
    """

    def __init__(self, image_w, image_h, line_len, dst, font_sizes=None, path_to_fonts='example/fonts',
                 path_to_texts='example/text'):
        """
        :param image_w: width of output image
        :param image_h: height of output image
        :param line_len: max count of chars in image
        :param dst: path than image would be saves
        """
        if font_sizes is None:
            font_sizes = [28, 32, 34, 38, 42]
        self.dst = dst
        self.font_sizes = font_sizes
        self.image_w = image_w
        self.image_h = image_h
        self.line_len = line_len
        self.path_to_fonts = path_to_fonts
        self.path_to_texts = path_to_texts
        os.makedirs(os.path.join(dst, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dst, 'pascalVocXml'), exist_ok=True)

    def run_threads(self, file_name=None):
        """
        Run thread for each set of options (like source file, font name and font size)
        :param file_name: name of source file if it's None, program will be create images from default texts. Source
        file must be in the txt format (see example in readme on GitHub)
        """
        if file_name is None:
            for root, _, files in os.walk(self.path_to_texts):
                for file in files:
                    file_name = os.path.join(root, file)
                    for f_root, _, fonts in os.walk(self.path_to_fonts):
                        for cur_font in fonts:
                            for size in self.font_sizes:
                                self.thread_start(file_name, os.path.join(f_root, cur_font), size)
            pass
        else:
            for f_root, _, fonts in os.walk('example/fonts'):
                for cur_font in fonts:
                    for size in self.font_sizes:
                        self.thread_start(file_name, os.path.join(f_root, cur_font), size)
            pass
        pass

    def thread_start(self, file_name, font_path, font_size):
        """
        Launch the tread for this options
        :param file_name: the file where the text will be taken from
        :param font_path: current font
        :param font_size: current font size
        :return:
        """
        table_name = 'result/tables/' + f'{file_name.split("/")[-1]}.' + str(font_path.split('/')[-1]) + str(font_size)
        new_thread = ImageGeneratorForRNN(file_name, self.image_w, self.image_h, self.line_len, font_path, font_size,
                                          table_name, dst=self.dst)
        new_thread.start()
        pass


# noinspection PyMethodMayBeStatic
class ImageGeneratorForRNN(Thread):
    """
    imageGenerator extends the Thread, for improve speed
    """

    def __init__(self, file_name, image_w, image_h, line_len, font_name, font_size, table_name, dst):
        """
        :param file_name: the file where the text will be taken from
        :param image_w: width of output image
        :param image_h: height of output image
        :param line_len: max count of chars in image
        :param font_name: current font
        :param font_size: current font size
        :param table_name: name of file in which the tag table will be stored
        :param dst: path than image would be saves
        """
        super().__init__()
        self.table = None
        self.counter = 1
        self.dst = dst
        self.image_w = image_w
        self.image_h = image_h
        self.line_len = line_len
        self.file_name = file_name
        self.font_name = font_name
        self.font_size = font_size
        self.table_name = table_name
        self.dist_path = os.path.join(self.dst,
                                      'images',
                                      f'{file_name.split("/")[-1]}.{font_name.split("/")[-1]}.{font_size}',
                                      '')
        self.regexp = \
            re.compile(
                r"([\t\r\n ]*[.,'\"{[(?!:;]?((\d+[.,]?\d*)|([.,'\"?!:;]?[а-яА-Я]+-?[а-яА-Я]*[.,'\"?!:;]?))[\t\r\n ]*)")

        os.makedirs(self.dist_path, exist_ok=True)

    def run(self):
        self.parse_file()
        # self.table.to_csv(self.table_name, index=False)
        pass

    def add_object_to_xml(self, xml, bbox):
        if bbox[0][0] > 0 and bbox[0][1] > 0 and bbox[1][0] > 0 and bbox[1][1] > 0:
            label = 'word'
        else:
            label = 'stump'
        obj = {
            'name': label,
            'xmax': str(bbox[1][0]),
            'xmin': str(bbox[0][0]),
            'ymax': str(bbox[0][1]),
            'ymin': str(bbox[1][1])
        }
        xml.add_objects([obj])

    def parse_file(self):
        """
        This method can open the target file and read it, check words for match with regexp and send
         it into image generator method
        :return:
        """
        from makiflow.tools.xml_creator import XmlCreator
        xml_creator = XmlCreator(os.path.join(self.dst, 'pascalVocXml'))

        with open(self.file_name, 'r') as f:
            count = 1

            static_data = {
                'filename': f'{count}.png',
                'folder': f'{self.dist_path.split("/")[-2]}',
                'segmented': '0',
                'depth': '1',
                'height': str(self.image_h),
                'width': str(self.image_w),
                'annotation': 'PASCAL VOC2011',
                'database': 'Handwritten-like text images',
                'image': 'Unspecified'
            }
            xml = xml_creator.create_pascal_voc_xml(f'{self.font_name.split("/")[-1]}-{self.font_size}-{count}',
                                                    static_data)

            font = ImageFont.truetype(self.font_name, self.font_size)
            font_h = font.getsize('word')[1]

            img, draw, shift_h = self.draw_line_marking()
            top = shift_h - font_h
            left = self.get_tab_length()

            for line in f:
                if count == 150:
                    break

                if len(line) < 2:
                    continue
                words = line.split(' ')
                # TODO переключить на грейскейл
                for word in words:
                    word = word.strip()
                    if self.regexp.fullmatch(word):
                        draw.text((left, top), word, 0, font=font)
                        box_shape = font.getsize(word)
                        self.add_object_to_xml(xml, ((left, top), box_shape))
                        # draw.rectangle((left - 5, top + 2, left + box_shape[0], top + box_shape[1]),
                        #                outline=(1, 124, 0), width=2)
                        left += font.getsize(word)[0] + self.get_indent_between_word(with_noise=False)

                        if left >= self.image_w:
                            top += self.get_indent_between_lines(random_ratio=0, with_noise=False)

                            if top + font.getsize(word)[1] > self.image_h - self.get_indent_between_lines() / 3:
                                path = f'{self.dist_path}{count}'
                                img.save(f'{path}.png', 'PNG')
                                xml.save()
                                xml = xml_creator. \
                                    create_pascal_voc_xml(f'{self.font_name.split("/")[-1]}-{self.font_size}-{count}',
                                                          static_data)
                                img, draw, shift_h = self.draw_net_marking() if random.random() > 0.8 \
                                    else self.draw_line_marking()

                                top = shift_h - font_h
                                left = self.get_tab_length()
                                count += 1
                                break

                            left = self.get_tab_length()
                pass
            pass
        pass

    def create_clear_image(self):
        img = Image.new('L', (self.image_w, self.image_h), color='white')
        draw = ImageDraw.Draw(img)
        return img, draw

    def indent_between_noise(self, orient):
        if orient == 'w':
            return random.randint(-10, 5)
        if orient == 'h':
            return random.randint(-2, 5)
        pass

    def get_indent_between_lines(self, line_h=20, random_ratio=0.8, with_noise=True):
        res = line_h if random.random() < random_ratio else line_h * 2
        if with_noise:
            res += self.indent_between_noise('h')
        return res

    def get_tab_length(self, tab_length=40, random_ratio=0.8):
        return tab_length if random.random() > random_ratio else 0

    def get_indent_between_word(self, indent_between=15, with_noise=True):
        res = indent_between
        if with_noise:
            res += self.indent_between_noise('w')
        return res

    def draw_net_marking(self, line_h=20):
        img, draw = self.create_clear_image()
        shift_h = random.randint(0, line_h)
        for i in range(0, self.image_h // line_h):
            draw.line(((0, i * line_h + shift_h), (self.image_w, i * line_h + shift_h)), fill=120, width=1)
        return img, draw, shift_h

    def draw_line_marking(self, line_h=20, line_w=20):
        img, draw = self.create_clear_image()
        shift_h = random.randint(0, line_h)
        shift_w = random.randint(0, line_w)
        for i in range(0, self.image_w // line_w):
            draw.line(((i * line_w + shift_w, 0), (i * line_h + shift_w, self.image_h)), fill=120, width=1)
        for i in range(0, self.image_h // line_h):
            draw.line(((0, i * line_h + shift_h), (self.image_w, i * line_h + shift_h)), fill=120, width=1)
        return img, draw, shift_h

    def create_and_open_table(self):
        table = pd.DataFrame({'path': [], 'feature': []})
        table.columns = ['path', 'feature']
        self.table = table
        pass

    def put_row(self, path, feature):
        self.table.loc[len(self.table)] = {'path': path, 'feature': feature}
        pass

    def create_image(self, line):
        """
        Method create image that contains the input line
        :param line: input line
        :return:
        """
        path = os.path.join(self.dst, f'{self.dist_path}', f'{self.counter}.png')

        img = Image.new('RGB', (self.image_w, self.image_h), color='white')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.font_name, self.font_size)

        if font.getsize(line)[0] > self.image_w:
            return

        draw.text((0, 0), line, (0, 0, 0), font=font)

        img.save(path, 'PNG')

        self.put_row(f'{self.dist_path}' + f'{self.counter}.png', line)
        self.counter += 1
        pass


if __name__ == "__main__":
    factory = ImageFactoryForSSD(500, 500, 0, 'result')
    factory.run_threads()

    # img = Image.new('RGB', (100, 100), color='white')
    # draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype('ofont.ru_ScriptS.ttf', 30)
    # draw.text((0, -10), 'ttttttt', (0, 0, 0), font=font)
    # img.show()
