'''
MIT License

Copyright (c) 2023 e-con Systems India Pvt Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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

