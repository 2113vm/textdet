import json

from ycapi.contour import Contour


def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


class BB:
    def __init__(self, bb):
        self.bb = Contour(bounding_rect=self._preprocess_bb(bb))

    @staticmethod
    def _preprocess_bb(v):
        """
        vertices: {'vertices': [{'x': '1541', 'y': '39'}, top left
                                {'x': '1541', 'y': '98'}, bot left
                                {'x': '1840', 'y': '98'}, bot right
                                {'x': '1840', 'y': '39'}] top right
                  }
        1-4
        | |
        2-3
        return: Tuple[int, int, int, int] <=> bounding_rect = [xmin, ymin, width, height]
        """
        v = v['vertices']
        return tuple([int(v[0].get('x', 0)), int(v[0].get('y', 0)),
                      int(v[2].get('x', 0)) - int(v[0].get('x', 0)),
                      int(v[1].get('y', 0)) - int(v[0].get('y', 0))])


class Line:
    def __init__(self, line_dict):
        self.line_bb = BB(line_dict['boundingBox'])
        self.words = [Word(word) for word in line_dict['words']]


class Word:
    def __init__(self, word_dict):
        self.word_bb = BB(word_dict['boundingBox'])
        self.language = word_dict['languages']
        self.text = word_dict['text']
        self.confident = word_dict['confidence']


class Block:
    def __init__(self, block_dict):
        self.block_bb = BB(block_dict['boundingBox'])
        self.lines = [Line(line) for line in block_dict['lines']]

    @staticmethod
    def _preprocess_bb(v):
        """
        vertices: {'vertices': [{'x': '1541', 'y': '39'}, top left
                                {'x': '1541', 'y': '98'}, bot left
                                {'x': '1840', 'y': '98'}, bot right
                                {'x': '1840', 'y': '39'}] top right
                  }
        1-4
        | |
        2-3
        return: Tuple[int, int, int, int] <=> bounding_rect = [xmin, ymin, width, height]
        """
        return tuple([int(v[0]['x']), int(v[0]['y']),
                      int(v[2]['x']) - int(v[0]['x']),
                      int(v[1]['y']) - int(v[0]['y'])])


class Page:
    def __init__(self, page_dict):
        self.blocks = [Block(block) for block in page_dict['blocks']]
        self.width = page_dict['width']
        self.height = page_dict['height']


class JsonDataClass:
    """
    Structure:
    - Page:[
        - Block:[
            - Line: [
                - Word
                ]
            ]
        ]

    """

    def __init__(self, data):

        self.pages = []

        for meta_result in data['results']:
            for result in meta_result['results']:
                for pages in result['textDetection']['pages']:
                    self.pages.append(Page(pages))


def get_words_from_json(parsed_json):
    words = []

    for page in parsed_json.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    words.append(word)

    return words


def write(filename, text):
    with open(filename, 'w') as f:
        f.write(text)


def get_crop(img, box):
    x, y, w, h = box
    return img[y: y + h, x: x + w]