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