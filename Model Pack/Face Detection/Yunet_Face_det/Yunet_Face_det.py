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
import Configuration
import Preprocess
import Postprocess
import numpy as np

class Yunet_Face_det(OutputDecoderInterface):
    def __init__(self) -> None:
        super().__init__()
        self.config = Configuration.Configuration()
        preprocess_obj =  Preprocess.Preprocess(self.config)
        self.anchor_boxes = preprocess_obj.forward()
        
    def get_output_labels(self):
        """
        This function must a list of class labels in string format.
          
        Returns:
            []: A List of of class labels in string format.
        """
        return ['face']
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
        output_loc = output[0].astype(np.float32)

        if output[2].shape[1] == 2:
            output_confi = output[2].astype(np.float32)
            output_ioc = output[1].astype(np.float32)
        else:
            output_confi = output[1].astype(np.float32)
            output_ioc = output[2].astype(np.float32)

        # if output is unquantized, dequantizing the output
        if output[0].dtype != np.float32:
            output_confi = (output_confi - self.config.output_zero_point_confidence_score) * self.config.output_scale_confidence_score
            output_loc = (output_loc - self.config.output_zero_point_loc_iou) * self.config.output_scale_loc_iou
            output_ioc = (output_ioc - self.config.output_zero_point_loc_iou) * self.config.output_scale_loc_iou

 
        postprocess_obj = Postprocess.Postprocess(self.config, self.anchor_boxes, output_loc, output_confi, output_ioc)
        dets = postprocess_obj.forward()
        dets = dets[:self.config.keep_top_k_faster_nms, :]  
        label_boxes_scores = []
        for k in range(dets.shape[0]): # save dets
            if dets[k, 4] < self.config.confidence_threshold: # vis threshold
                continue
            #debug prints faces with box and score
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            score = dets[k, 4]
            # w = xmax - xmin + 1
            # h = ymax - ymin + 1
            label_boxes_scores.append(LabelBoundingBoxScore(xmin,ymin,xmax,ymax,0,score))
        return label_boxes_scores
    def get_model_input(self):
        """
        This function must return model input size. This will be used for rescaling the 
        model output to preview resolution while overlaying. 
          
        Returns:
            (w,h): A Tuple of width and height of the model input
        """
        return (240,320)
    def get_task(self):
        """
        This function must return the type of task the model performs.
          
        Returns:
            "": A string from any of the following: (regression,classification,detection)
        """
        return "detection"