from typing import List

import csv
import cv2
from dataclasses import dataclass, field
import json
from imutils import perspective
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread, imsave
from skimage.draw import polygon2mask
from tqdm import tqdm


class Region:
    def __init__(self, region_id, region_shape_attributes, region_attributes):
        self.region_id = region_id
        self._region_shape_attributes = region_shape_attributes
        self._region_attributes = region_attributes
        self.is_empty_region_shape_attr = not bool(region_shape_attributes)
        self.is_empty_region_attr = not bool(region_attributes)

        for attr, value in self._region_shape_attributes.items():
            setattr(self, attr, value)

        for attr, value in self._region_attributes.items():
            setattr(self, attr, value)


@dataclass
class Filename:
    filename: str
    file_size: int
    file_attributes: dict
    region_count: int
    regions: List[Region] = field(default_factory=list)

    def append_region(self, region_shape_attributes, region_attributes):
        """

        :param region_shape_attributes:
        {"name":"polygon","all_points_x":,"all_points_y":} or
        "{""name"":""rect"",""x"":125,""y"":234,""width"":520,""height"":708}"
        :param region_attributes: {}
        :return:
        """
        region_id = self.regions[-1].region_id + 1 if self.regions else 0
        region = Region(region_id, region_shape_attributes, region_attributes)
        self.region_count += 1
        self.regions.append(region)


class Dataset:
    def __init__(self, via_file_path):
        self.via_file_path = via_file_path

        self.files = {}

    def read(self, via_file_path):

        with open(via_file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row['filename']
                file_size = int(row['file_size'])
                file_attributes = json.loads(row['file_attributes'])
                region_count = int(row['region_count'])
                region_id = int(row['region_id'])
                region_shape_attributes = json.loads(row['region_shape_attributes'])
                region_attributes = json.loads(row['region_attributes'])
                region = Region(region_id=region_id,
                                region_attributes=region_attributes,
                                region_shape_attributes=region_shape_attributes)
                if filename in self.files:
                    self.files[filename].regions.append(region)
                else:
                    self.files[filename] = Filename(filename, file_size, file_attributes, region_count, [region])

    def write(self, path_to_save):
        data = []
        for file in self:
            filename = str(file.filename)
            file_size = str(file.file_size)
            file_attributes = '"{}"'
            region_count = str(file.region_count)
            for region in file.regions:
                region_id = str(region.region_id)
                region_shape_attribute = json.dumps(region._region_shape_attributes).replace(' ', '')
                region_attributes = json.dumps(region._region_attributes).replace(' ', '')

                data.append([
                    filename,
                    file_size,
                    file_attributes,
                    region_count,
                    region_id,
                    region_shape_attribute,
                    region_attributes
                ])
        columns = [
            'filename',
            'file_size',
            'file_attributes',
            'region_count',
            'region_id',
            'region_shape_attributes',
            'region_attributes'
            ]
        pd.DataFrame(data=data, columns=columns).to_csv(path_to_save, index=False)

    def __getitem__(self, item):
        return self.files[item]

    def __iter__(self):
        self.__i = 0
        return self

    def __next__(self):
        if self.__i == len(self.files):
            raise StopIteration
        current_key = list(self.files.keys())[self.__i]
        current_elem = self.files[current_key]
        self.__i += 1
        return current_elem


class VIA:
    def __init__(self, via_file_path, image_dir, dataset):
        self.via_file_path = via_file_path
        self.image_dir = image_dir
        self.dataset = dataset

    def _read(self):
        pass

    def cut(self, img_dir_to_save):

        for file in tqdm(self.dataset):
            img_path = Path(self.image_dir) / Path(file.filename)
            img = imread(str(img_path))
            for region in file.regions:
                if region.is_empty_region_shape_attr:
                    continue
                region_img = self._get_img_by_region(img, region)
                img_name_to_save = f'{Path(file.filename).stem}-{str(region.region_id)}{Path(file.filename).suffix}'
                img_path_to_save = Path(img_dir_to_save) / Path(img_name_to_save)
                imsave(str(img_path_to_save), region_img)

    @staticmethod
    def _get_img_by_region(img, region):
        if region.name == 'polygon':
            mask = polygon2mask(img.shape, [[y, x] for y, x in zip(region.all_points_y, region.all_points_x)])
            mask = mask.astype(np.uint8)
            pts = cv2.goodFeaturesToTrack(mask[:, :, 0], 4, 0.05, 100)
            transformed_mask = perspective.four_point_transform(mask * img, pts.reshape(4, 2))
        else:
            transformed_mask = img[region.y: region.y + region.height, region.x: region.x + region.width, :]
        return transformed_mask

    def get_markup_by_filename(self, filename):
        pass

    def get_masks(self):
        pass


if __name__ == '__main__':

    csv_file_path = 'data/meta.csv'
    img_dir = 'data/cleaned_sigs'

    via = VIA(via_file_path=csv_file_path, image_dir=img_dir, dataset=Dataset(csv_file_path))


"""
TODO:
1. warning for empty markup
2. convert contours to csv file 
"""