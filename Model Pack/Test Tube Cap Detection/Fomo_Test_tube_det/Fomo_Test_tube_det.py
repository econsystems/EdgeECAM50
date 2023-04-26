'''
Copyright 2022 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
'''

from interface.output_decoder import OutputDecoderInterface
from interface_utils.labelboundingboxscore import LabelBoundingBoxScore
from typing import Type, Union, List

import numpy as np
import math
import json

class Centroid(object):
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx+dy*dy)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return str({'x': self.x, 'y': self.y, 'label': self.label})

class BoundingBox(object):

    def from_x_y_h_w(x, y, h, w):
        return BoundingBox(x, y, x+w, y+h)

    def from_dict(d: dict):
        return BoundingBox(d['x0'], d['y0'], d['x1'], d['y1'])

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    def close(self, other, atol) -> bool:
        return (np.isclose(self.x0, other.x0, atol=atol) and
                np.isclose(self.y0, other.y0, atol=atol) and
                np.isclose(self.x1, other.x1, atol=atol) and
                np.isclose(self.y1, other.y1, atol=atol))

    def project(self, width:int, height:int):
        return BoundingBox(self.x0 * width, self.y0 * height,
                        self.x1 * width, self.y1 * height)

    def floored(self):
        return BoundingBox(math.floor(self.x0), math.floor(self.y0),
                        math.floor(self.x1), math.floor(self.y1))

    def transpose_x_y(self):
        return BoundingBox(self.y0, self.x0, self.y1, self.x1)

    def centroid(self):
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        return Centroid(cx, cy, label=None)

    def width(self):
        return self.x1 - self.x0

    def height(self):
        return self.y1 - self.y0

    def update_with_overlap(self, other) -> bool:
        """ update ourselves with any overlap. return true if there was overlap """
        if (other.x0 > self.x1 or other.x1 < self.x0 or
            other.y0 > self.y1 or other.y1 < self.y0):
            return False
        if other.x0 < self.x0:
            self.x0 = other.x0
        if other.y0 < self.y0:
            self.y0 = other.y0
        if other.x1 > self.x1:
            self.x1 = other.x1
        if other.y1 > self.y1:
            self.y1 = other.y1
        return True

    def __eq__(self, other) -> bool:
        return self.close(other, atol=1e-8)

    def __iter__(self):
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1

    def __repr__(self) -> str:
        return str({'x0': self.x0, 'y0': self.y0, 'x1': self.x1, 'y1': self.y1})

class BoundingBoxLabelScore(object):

    def from_dict(d: dict):
        bbox = BoundingBox.from_dict(d['bbox'])
        return BoundingBoxLabelScore(bbox, d['label'], d['score'])

    def from_bounding_box_labels_file(fname):
        """
        Parse a bounding_box.labels file as exported from Studio and return a
        dictionary with key = source filename & value = [BoundingBoxLabelScore]
        """
        with open(fname, 'r') as f:
            labels = json.loads(f.read())
            if labels['version'] != 1:
                raise Exception(f"Unsupported file version [{labels['version']}]")
            result = {}
            for fname, bboxes in labels['boundingBoxes'].items():
                bbls = []
                for bbox in bboxes:
                    x, y = bbox['x'], bbox['y']
                    w, h = bbox['width'], bbox['height']
                    bbls.append(BoundingBoxLabelScore(
                        BoundingBox.from_x_y_h_w(x, y, h, w),
                        bbox['label']))
                result[fname] = bbls
            return result

    def __init__(self, bbox: BoundingBox, label: int, score: float=None):
        self.bbox = bbox
        self.label = label
        self.score = score

    def centroid(self):
        centroid = self.bbox.centroid()
        centroid.label = self.label
        return centroid

    def __eq__(self, other) -> bool:
        if self.score is None or other.score is None:
            score_equal = self.score == other.score
        else:
            score_equal = np.isclose(self.score, other.score)
        return (score_equal and
                self.bbox == other.bbox and
                self.label == other.label)

    def __repr__(self) -> dict:
        return str({'bbox': str(self.bbox), 'label': self.label,
                    'score': self.score })

class Labels:
    """Represents a set of labels for a classification problem"""

    def __init__(self, labels: 'list[str]'):
        if len(set(labels)) < len(labels):
            raise ValueError('No duplicates allowed in label names')
        self._labels_str = labels

    # Need to upgrade to numpy >= 1.2.0 to get proper type support
    def __getitem__(self, lookup: 'Union[int, np.integer, str]'):
        if isinstance(lookup, (int, np.integer)):
            if lookup < 0:
                raise IndexError(f'Index {lookup} is too low')
            if lookup >= len(self._labels_str):
                raise IndexError(f'Index {lookup} is too high')
            return Label(self, int(lookup), self._labels_str[lookup])
        elif isinstance(lookup, str):
            return Label(self, self._labels_str.index(lookup), lookup)
        else:
            raise IndexError(f'Index {lookup} is not in the list of labels')

    def __len__(self):
        return len(self._labels_str)

    def __iter__(self):
        for idx in range(0, len(self._labels_str)):
            yield Label(self, idx, self._labels_str[idx])


class Label:
    """Represents an individual label for a classification problem"""

    def __init__(self, labels: Labels, label_idx: int, label_str: str):
        self._labels = labels
        self._label_idx = label_idx
        self._label_str = label_str

    @property
    def idx(self):
        return self._label_idx

    @property
    def str(self) -> str:
        return self._label_str

    @property
    def all_labels(self):
        return self._labels

    def __eq__(self, other):
        if isinstance(other, Label):
            # Individual labels are only the same if they come from the same list
            if list(other.all_labels._labels_str) != list(self.all_labels._labels_str):
                raise ValueError('Cannot compare Label from different sets')
            return self._label_idx == other._label_idx
        raise TypeError('Cannot compare Label with non-labels')

class Fomo_Test_tube_det(OutputDecoderInterface):
    def __init__(self) -> None:
        super().__init__()
        self.zero_point = -128
        self.scale = 0.003921568859368563
        
    def fuse_adjacent(self,bbox_label_scores: List[BoundingBoxLabelScore]) -> List[BoundingBoxLabelScore]:
        """ Fuse adjacent / overlapping bboxes in the same way ei_cube_check_overlap does. """
        if bbox_label_scores == []:
            return []
        collected_bbox_label_scores = []
        for orig_bls in bbox_label_scores:
            had_overlap = False
            for collected_bls in collected_bbox_label_scores:
                if collected_bls.label != orig_bls.label:
                    continue
                had_overlap = collected_bls.bbox.update_with_overlap(orig_bls.bbox)
                if had_overlap:
                    if orig_bls.score > collected_bls.score:
                        collected_bls.score = orig_bls.score
                    break
            if not had_overlap:
                collected_bbox_label_scores.append(orig_bls)
        return collected_bbox_label_scores

    def convert_segmentation_map_to_object_detection_prediction(self,
        segmentation_map: np.ndarray,
        minimum_confidence_rating: float,
        fuse: bool) -> List[BoundingBoxLabelScore]:
        """ Converts (N, N) segmentation map back to bbox and one hot labels.

        Args:
        segmentation_map: (H, W, C) segmentation map output.
        minimum_confidence_rating: threshold for probability.
        fuse: whether to fuse adjacent cells in a manner matching
            ei_cube_check_overlap

        Returns:
        list of (bbox, label, score) tuples representing detections.

        Given a (H, W, C) output from a segmentation model convert back
        to list of (bbox, label, score) as used in Studio. Filter entries to
        have at least minimum_confidence_rating probability.
        """

        # check shape
        if len(segmentation_map.shape) != 3:
            raise Exception("Expected segmentation map to be shaped "
                                f" (H, W, C) but was {segmentation_map.shape}")
        width, height, num_classes_including_background = segmentation_map.shape

        # check it has some non background classes
        if num_classes_including_background < 2:
            raise Exception("Expected at least one non background class but"
                            f" had {num_classes_including_background}"
                            f" (shape {segmentation_map.shape})")

        # will return boxes, labels and scores
        boxes_labels_scores = []

        # check all non background classes (background is class 0)
        for class_idx in range(1, num_classes_including_background):
            # TODO(mat): should we fuse and THEN filter by min conf rating?
            # determine which grid points are at least the minimum confidence rating
            xs, ys = np.where(segmentation_map[:,:,class_idx] > minimum_confidence_rating)
            for x, y in zip(xs, ys):
                # retain class 0 as background
                boxes_labels_scores.append(
                    BoundingBoxLabelScore(
                        BoundingBox(x/width, y/height, (x+1)/width, (y+1)/height),
                        label=class_idx-1,
                        score=float(segmentation_map[x, y, class_idx])),
                )

        if fuse:
            boxes_labels_scores = self.fuse_adjacent(boxes_labels_scores)

        return boxes_labels_scores

    def get_output_labels(self):
        """
        This function must a list of class labels in string format.
          
        Returns:
            []: A List of of class labels in string format.
        """
        return ['bl','pr','rd','yl']
    def decode_output(self,output):
        """    
        This function takes the raw inference output as input and returns the post processed valid
        results based on requirements (e.g threshold or nms). 

        For regression the score value must be filled in the LabelBoundingBoxScore(), 
        For classification the label index must be filled in the label variable, 
        For detection the bounding box, label and score must be filled.
    
        Parameters:
        output (List): A List of numpy array of size equal to the number of outputs of the model.
                       The dimension of each numpy array might be different based on the output's dimension.
    
        Returns:
        label_boxes_scores (List): Returns a list of LabelBoundingBoxScore objects of size equal to the number
                                   detection which needs to be overlayed.
    
        """
        output_map = output[0].astype(np.float32)

        # if output is unquantized, dequantizing the output
        if output[0].dtype != np.float32:
            output_map = (output_map - self.zero_point) * self.scale

        box_label = self.convert_segmentation_map_to_object_detection_prediction(output_map[0],0.5,True)
        label_boxes_scores = []
        for box in box_label:
            label_boxes_scores.append(LabelBoundingBoxScore(box.bbox.y0*160,box.bbox.x0*160,box.bbox.y1*160,box.bbox.x1*160,box.label,box.score))
        return label_boxes_scores
    def get_task(self):
        """
        This function must return the type of task the model performs.
          
        Returns:
            "": A string from any of the following: (regression,classification,detection)
        """
        return "detection"
    def get_model_input(self):
        """
        This function must return model input size. This will be used for rescaling the 
        model output to preview resolution while overlaying. 
          
        Returns:
            (w,h): A Tuple of width and height of the model input
        """
        return (160,160)