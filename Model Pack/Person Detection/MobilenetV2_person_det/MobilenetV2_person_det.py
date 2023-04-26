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
import numpy as np

class MobilenetV2_person_det(OutputDecoderInterface):
    def __init__(self) -> None:
        super().__init__()
        self.zero_point = -128
        self.scale = 0.00390625

    def get_output_labels(self):
        """
        This function must a list of class labels in string format.
          
        Returns:
            []: A List of of class labels in string format.
        """
        return ['Person not Found','Person detected']
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
        output_quant = output[0].astype(np.float32)

        # if output is unquantized, dequantizing the output
        if output[0].dtype != np.float32:
            output_quant = (output_quant - self.zero_point) * self.scale
        label_boxes_scores = []
        if output_quant[0][0] >= 0.5:
            index = 1
        else:
            index = 0
        label_boxes_scores.append(LabelBoundingBoxScore(label=index,score=output_quant[0][0]))
        return label_boxes_scores
    def get_task(self):
        """
        This function must return the type of task the model performs.
          
        Returns:
            "": A string from any of the following: (regression,classification,detection)
        """

        return "classification"
    def get_model_input(self):
        """
        This function must return model input size. This will be used for rescaling the 
        model output to preview resolution while overlaying. 
          
        Returns:
            (w,h): A Tuple of width and height of the model input
        """
        return (96,96)
