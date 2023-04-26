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

from itertools import product as product
import numpy as np

class Preprocess():
    
    def __init__(self, config):
        self.__feature_map = []
        self.__input_shape = config.input_shape
        self.__config = config
        
    def forward(self):
        feature_map_2th = [int(int((self.__input_shape[0]+1)/2)/2), 
                           int(int((self.__input_shape[1]+1)/2)/2)]
        self.__feature_map.append((int((feature_map_2th[0])/2), int((feature_map_2th[1])/2)))
        for index in range(1,4):
            self.__feature_map.append((int(self.__feature_map[index-1][0]/2), int(self.__feature_map[index-1][1]/2)))
        # anchor box creation
        anchors = []
        for k,f in enumerate(self.__feature_map):
            minsize = self.__config.minsizes[k]
            for i,j in product(range(f[0]), range(f[1])):
                for step in minsize:
                    s_kx = step / float(self.__input_shape[1])
                    s_ky = step / float(self.__input_shape[0])
                    cx = (j + 0.5) * self.__config.steps[k] / self.__input_shape[1]
                    cy = (i + 0.5) * self.__config.steps[k] / self.__input_shape[0]
                    anchors.append(( cx, cy, s_kx, s_ky ))
    
        return np.asarray(anchors)