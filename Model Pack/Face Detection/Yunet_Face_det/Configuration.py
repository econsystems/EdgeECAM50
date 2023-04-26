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

class Configuration():
    def __init__(self) :
        self.variances = [0.1, 0.2]
        self.output_scale_confidence_score = 0.00390625
        self.output_zero_point_confidence_score = -128
        self.output_zero_point_loc_iou = -20
        self.output_scale_loc_iou = 0.059299588203430176 
        self.minsizes = [(10.0, 16.0, 24.0), (32.0, 48.0), (64.0, 96.0), (128.0, 192.0, 256.0)]
        self.steps = [ 8, 16, 32, 64 ]
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.3
        self.top_k = 5000
        self.keep_top_k_faster_nms = 750
        self.input_shape = (320,240,3)
        self.scale = [self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]
