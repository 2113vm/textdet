from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


class NotRotatedBB:
    def __init__(self, coords):
        x1, y1, x2, y2, x3, y3, x4, y4 = coords
        self.x_max = max(x1, x2, x3, x4)
        self.x_min = min(x1, x2, x3, x4)
        self.y_max = max(y1, y2, y3, y4)
        self.y_min = min(y1, y2, y3, y4)
        self.h = self.y_max - self.y_min
        self.w = self.x_max - self.x_min
        self.area = self.h * self.w

    def __sub__(self, other):
        """
        Intersection of two boxes
        :param other: NotRotatedBB
        :return: int.
        """
        x_inter = (self.w + other.w) - (max(self.x_max, other.x_max) - min(self.x_min, other.x_min))
        y_inter = (self.h + other.h) - (max(self.y_max, other.y_max) - min(self.y_min, other.y_min))

        if x_inter > 0 and y_inter > 0:
            return x_inter * y_inter
        else:
            return 0

    def __add__(self, other):
        """
        Union of two boxes
        :param other:
        :return:
        """
        return (self.area + other.area) - (self - other)


class Validator:
    """
    TP : True Positive (based on comparing the bounding boxes/lines)
    NGT : Number of bounding boxes/lines in the GT
    NAlg : Number of detected bounding boxes/lines returned by the Algorithm
    TH : Threshold value (1)
    Precision : P = TP / NALG
    Recall : R = TP / NGT
    F1-Score : F1 = 2PR / (P + R)
    """
    def __init__(self, gt_path, alg_path, threshold):
        """

        :param gt_path: Path of file with ground_truth.
        Format gt.txt:
        1,2,3,4,5,6,7,8
        2,3,4,5,6,7,8,9
        :param alg_path: Path of file with text detector result
        Format alg.txt:
        1,2,3,4,5,6,7,8
        2,3,4,5,6,7,8,9
        :param threshold: It's threshold for IOU. If iou > threshold, then true_positive += 1
        """
        self.threshold = threshold

        with open(gt_path, 'r') as gt_file:
            gt = gt_file.readlines()

        with open(alg_path, 'r') as alg_file:
            alg = alg_file.readlines()

        self.gt_boxes = [NotRotatedBB(list(map(int, line.strip().split(',')[:8]))) for line in gt]
        self.alg_boxes = [NotRotatedBB(list(map(int, line.strip().split(',')[:8]))) for line in alg]

    def get_metrics(self):
        ious = np.zeros((len(self.gt_boxes), len(self.alg_boxes)))

        for gt_num, gt in enumerate(self.gt_boxes):
            for alg_num, alg in enumerate(self.alg_boxes):
                ious[gt_num, alg_num] = self._iou(gt, alg)

        true_positive = (ious > self.threshold).astype(int).sum()
        precision = true_positive / len(self.alg_boxes)
        recall = true_positive / len(self.gt_boxes)
        f_measure = (2 * precision * recall) / (precision + recall)
        return true_positive, precision, recall, f_measure

    @staticmethod
    def _iou(box1: NotRotatedBB, box2: NotRotatedBB):
        return (box1 - box2) / (box1 + box2)


class Manager:
    def __init__(self, gt_dir, alg_dir, threshold):
        self.gt_dir = Path(gt_dir)
        self.alg_dir = Path(alg_dir)
        self.threshold = threshold
        print(f"Found {len(list(self.gt_dir.glob('*.txt')))} in gt_dir, "
              f"{len(list(self.alg_dir.glob('*.txt')))} in alg_dir")

    def evaluate(self):
        data = []
        for gt_path in tqdm(self.gt_dir.glob('*.txt'), total=len(list(self.gt_dir.glob('*.txt')))):
            alg_pathes = [
                self.alg_dir / gt_path.name,
                self.alg_dir / (gt_path.name.replace('gt_', '')),
                self.alg_dir / ('res_' + gt_path.name.replace('gt_', '')),
            ]
            alg_pathes = list(filter(lambda p: p.exists(), alg_pathes))
            if not alg_pathes:
                print(f"Skipped {gt_path} because couldn't match {gt_path} to anything in alg_dir")
                data.append([gt_path.name.replace('.txt', '.jpg'), 0, 0, 0, 0])
                continue
            alg_path = alg_pathes[0]
            data.append([
                gt_path.name.replace('.txt', '.jpg'),
                *Validator(gt_path, alg_path, self.threshold).get_metrics()
            ])
        return pd.DataFrame(data=data, columns=['image_name', 'true_positive', 'precision', 'recall', 'f_measure'])


if __name__ == '__main__':
    gt_dir, alg_dir, threshold, output_filename = sys.argv[1:5]
    Manager(gt_dir, alg_dir, float(threshold)).evaluate().to_csv(output_filename, index=False)

