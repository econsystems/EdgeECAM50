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

import numpy as np

class Postprocess():
    def __init__(self, config, anchor_boxes, loc,confi, iou):
        self.__input_anchor = anchor_boxes
        self.__loc = loc
        self.__config = config
        self.__scale = config.scale
        self.__confi = confi
        self.__iou = iou
    
    def nms(self,dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


    def forward(self):
        #Box generation
        boxes = np.concatenate((
        self.__input_anchor[:, 0:2] + self.__loc[:, 0:2] * self.__config.variances[0] * self.__input_anchor[:, 2:4],
        self.__input_anchor[:, 2:4] * np.exp(self.__loc[:, 2:4] * self.__config.variances[1]),
        self.__input_anchor[:, 0:2] + self.__loc[:, 4:6] * self.__config.variances[0] * self.__input_anchor[:, 2:4],
        self.__input_anchor[:, 0:2] + self.__loc[:, 6:8] * self.__config.variances[0] * self.__input_anchor[:, 2:4],
        self.__input_anchor[:, 0:2] + self.__loc[:, 8:10] * self.__config.variances[0] * self.__input_anchor[:, 2:4],
        self.__input_anchor[:, 0:2] + self.__loc[:, 10:12] * self.__config.variances[0] * self.__input_anchor[:, 2:4],
        self.__input_anchor[:, 0:2] + self.__loc[:, 12:14] * self.__config.variances[0] * self.__input_anchor[:, 2:4]), 1)
        boxes[:, 0:2] -= boxes[:, 2:4] / 2
        boxes[:, 2:4] += boxes[:, 0:2]

        boxes = boxes[:, :4] # omit landmarks
        boxes = boxes * self.__scale
        
        
        #Score Calculation 
        cls_scores = np.asarray(self.__confi)[:, 1]
        iou_scores = np.asarray(self.__iou)[:,0]
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        _idx = np.where(iou_scores < 0)
        iou_scores[_idx] = 0
        scores = np.sqrt(cls_scores * iou_scores)   
        
        #Get appropriate anchor boxes based on NMS threshold
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.nms(dets, self.__config.nms_threshold)
        return dets[keep, :]

