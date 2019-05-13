import os
import re
from threading import Thread

import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class ImageFactoryForRNN:
    """
    This class must create threads for all set of options
    """

    def __init__(self, image_w, image_h, line_len, dst, font_sizes=None):
        """
        Parameters
        ----------

            image_w : int
                width of output image
            image_h : int
                height of output image
            line_len : int
                max count of chars in image
            dst : string
                path than image would be saves
        """
        if font_sizes is None:
            font_sizes = [28, 32, 34, 38, 42]
        self.dst = dst
        self.font_sizes = font_sizes
        self.image_w = image_w
        self.image_h = image_h
        self.line_len = line_len
        if not os.path.exists(os.path.join(dst, 'result')):
            os.mkdir(os.path.join(dst, 'result'))

    def run_threads(self, file_name=None):
        """
        Run thread for each set of options (like source file, font name and font size)

        Parameters

            file_name : string
                name of source file if it's None, program will be create images from default texts. Source
                file must be in the txt format (see example in readme on GitHub)
        """
        if file_name is None:
            for root, _, files in os.walk('texts'):
                for file in files:
                    file_name = os.path.join(root, file)
                    for f_root, _, fonts in os.walk('fonts'):
                        for cur_font in fonts:
                            for size in self.font_sizes:
                                self.thread_start(file_name, os.path.join(f_root, cur_font), size)
            pass
        else:
            for f_root, _, fonts in os.walk('fonts'):
                for cur_font in fonts:
                    for size in self.font_sizes:
                        self.thread_start(file_name, os.path.join(f_root, cur_font), size)
            pass
        pass

    def thread_start(self, file_name, font_path, font_size):
        """
        Launch the tread for this options

        Parameters
        ----------

            file_name : string
                the file where the text will be taken from
            font_path : string
                current font
            font_size : int
                current font size
        """
        table_name = 'result/tables/' + f'{file_name.split("/")[-1]}.' + str(font_path.split('/')[-1]) + str(font_size)
        new_thread = ImageGeneratorForRNN(file_name, self.image_w, self.image_h, self.line_len, font_path, font_size,
                                          table_name, dst=self.dst)
        new_thread.start()
        pass


class ImageGeneratorForRNN(Thread):
    """
    imageGenerator extends the Thread, for improve speed
    """

    def __init__(self, file_name, image_w, image_h, line_len, font_name, font_size, table_name, dst):
        """
        Parameters
        ----------

            file_name : string
                the file where the text will be taken from
            image_w : int
                width of output image
            image_h : int
                height of output image
            line_len : int
                max count of chars in image
            font_name : string
                current font
            font_size : int
                current font size
            table_name : string
                name of file in which the tag table will be stored
            dst : string
                path than image would be saves
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
        self.dist_path = os.path.join('result', f'{file_name.split("/")[-1]}.{font_name.split("/")[-1]}.{font_size}',
                                      '')
        self.regexp = \
            re.compile(
                r"([\t\r\n ]*[.,'\"{[(?!:;]?((\d+[.,]?\d*)|([.,'\"?!:;]?[а-яА-Я]+-?[а-яА-Я]*[.,'\"?!:;]?))[\t\r\n ]*)")
        if not os.path.exists(os.path.join(dst, self.dist_path)):
            os.mkdir(os.path.join(dst, self.dist_path))

    def run(self):
        self.__create_and_open_table()
        self.parse_file()
        self.table.to_csv(self.table_name, index=False)
        pass

    def parse_file(self):
        """
        This method can open the target file and read it, check words for match with regexp and send
         it into image generator method
        """
        with open(self.file_name, 'r') as f:
            for line in f:

                if len(line) < 2:
                    continue
                words = line.split(' ')

                for i in range(len(words) - 1):

                    if (len(words[i]) + 1 + len(words[i + 1])) < self.line_len and \
                            self.regexp.fullmatch(words[i]) and self.regexp.fullmatch(words[i + 1]):
                        self.create_image(words[i] + ' ' + words[i + 1])
                    else:
                        if len(words[i]) < self.line_len and self.regexp.fullmatch(words[i]):
                            self.create_image(words[i])
                if self.regexp.fullmatch(words[len(words) - 1]):
                    self.create_image(words[len(words) - 1])
                pass
            pass
        pass

    def __create_and_open_table(self):
        table = pd.DataFrame({'path': [], 'feature': []})
        table.columns = ['path', 'feature']
        self.table = table
        pass

    def __put_row(self, path, feature):
        self.table.loc[len(self.table)] = {'path': path, 'feature': feature}
        pass

    def create_image(self, line):
        """
        Method create image that contains the input line

        Parameters
        ----------

            line : string
                input line
        """
        path = os.path.join(self.dst, f'{self.dist_path}', f'{self.counter}.png')

        img = Image.new('RGB', (self.image_w, self.image_h), color='white')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.font_name, self.font_size)

        if font.getsize(line)[0] > self.image_w:
            return

        draw.text((0, 0), line, (0, 0, 0), font=font)

        img.save(path, 'PNG')

        self.__put_row(f'{self.dist_path}' + f'{self.counter}.png', line)
        self.counter += 1
        pass
