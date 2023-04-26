# EdgeECAM50_USB Model Pack

This repository has the sample tflite model files, configuration files and post processing modules to use on EdgeECAM50_USB. 

***

## Getting started

To make it easy for you to get started with EdgeECAM50_USB, here's a list of recommended steps.


## 1. Install edge-CAMView

- Download the edge-CAMView application from the [e-con Systems Developer Resources](https://developer.e-consystems.com/?_gl=1*1yfpk5*_ga*MTMwNjgxMTQyNy4xNjY0Nzc3OTY0*_ga_2G8R9Q209D*MTY4MjQ5ODQ5OC4xMy4xLjE2ODI0OTg1NTEuNy4wLjA.) website.
- Follow the steps provided in the edge-CAMView_Streaming_Application_Installation_Manual.


## 2. Add Post Processing Modules to edge-CAMView

 - Clone this repo using the git command:
 ``` 
 git clone https://github.com/econsystems/EdgeECAM50.git
 ``` 
 
 - Copy the desired Post Processing Module Folder (e.g. Yunet_Face_det folder) and paste it in the post_process folder present in the application installation path (e.g. C:\Program Files (x86)\e-con Systems\edge-CAMView\post_process).
 
 When you restart the application, you will be able to see the module listed in Postprocessing Module combo box in Preview Tab

## 3. Load Configuration
- Open the edge-ECAMView Application.
- Put the device in Dataset Mode (Preview Tab -> Mode -> Dataset)
- Go to Configuration Tab.
- Click **Browse** button under **Config Load** Section -> Select the desired **.config** file (e.g person detection/person_detection_quant.config). This will load the values in the config file onto the application
- Now, to configure the camera with this configuration, Press **Write Config** button

###  Quantized and UnQuantized configurations
You might find quantized configurations (e.g person detection/person_detection_quant.config) and unquantized configurations (face detection/face_detection_int8_unquantized.config) in this repo. The Key difference in these two configurations is the value of o_quantize variable in user_data. 

When deploying a int8 or uint8 model,
 - If the o_quantize variable is set to true, the output will be quantized int8 or uint8 based on the model. (Default)
 - If the o_quantize variable is set to false, the int8 or uint8 output will be unquantized to float32 using the values present in the output's quantization parameter. 

This makes no difference while using float32 models.

**Note:** Post processing of these quantized or unquantized output values should be taken care in the post processing module based on requirement. 


For creating your own configuration file or to generate configuration file using edge-ECAMView application, Please refer the edge-CAMView_Streaming_Application_Installation_Manual

## 4. Load Model

- Put the device in Dataset Mode (Preview Tab -> Mode -> Dataset)
- Go to Configuration Tab.
- Click **Browse** button under **Model Load** Section -> Select the desired **.tflite** file (e.g person detection/person_detection_int8.tflite). This will load the model and check if model size and required tensor arena size is within the limits and whether the model has any unsupported operators. If any of the above requirements are not met, the model cannot be loaded. 
- If model is loaded successfully, to deploy the model onto the device, Press **Write Model** button


## 5. Inference Output Visualization

 - Put the device in Inference Mode (Preview Tab -> Mode -> Dataset)
 - Start the inference ( Inference -> Click Start)
 - Select the desired post processing module listed in the combo box
 - Click Enable Overlay

This will overlay the model inference output on the preview.
***

## Writing own Post Processing Module 

If you wish to visualize the output of your model using edge-CAMView application, you need to write a custom post processing Module. 

 

### 1. Create a python file named after the post processing module (e.g., **MobilenetV2_person_det.py**) and do the following: 

- import Packages (failing to import these will make the script unusable by application): 
```
from interface.output_decoder import OutputDecoderInterface 

from interface_utils.labelboundingboxscore import LabelBoundingBoxScore 
```
### 2. Create the class with name as same as the file name, inherit the OutputDecoderInterface class 

```
class MobilenetV2_person_det(OutputDecoderInterface):
```
### 3. To create a custom post-processing module, you will need to implement the OutputDecoderInterface interface, which consists of four core methods. 

Implement the following four inference methods: 

- get_output_labels() should return a list of class labels in string format. 
```
def get_output_labels(self):
        """
        This function must a list of class labels in string format.
          
        Returns:
            []: A List of of class labels in string format.
        """
        return ['Person not Found','Person Found']
```

- decode_output() should return a dictionary of bounding boxes, label, and score values. The LabelBoundingBoxScore() object is designed to handle the model output values to bounding box coordinates, label index and scores. If there are multiple objects of interest present, a bounding box for each will be returned. The output here is same as that you see in output box when you start inference in the edge-CAMView application. 

 
```
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
```
- LabelBoundingBoxScore object used above will take x0,y0,x1,y1,label,score as arguments and returns a object. Based on the task these values will be retrieved and overlayed onto the preview.
 ```
# LabelBoundingBoxScore object

class LabelBoundingBoxScore(object):
    def __init__(self,  x0=None, y0=None, x1=None, y1=None, label: int=None, score: float=None):
```

- get_task() should return the task, such as classification, detection, or regression, as a string. For regression the score value must be filled in the LabelBoundingBoxScore() inside decode_output() method above, and for classification the label index must be filled in the label variable, and for detection the bounding box, label and score must be filled inside LabelBoundingBoxScore() method. 
```
    def get_task(self):
        """
        This function must return the type of task the model performs.
          
        Returns:
            "": A string from any of the following: (regression,classification,detection)
        """
        return "classification"
```

- get_model_input() should return tuple of input image size. 
```
    def get_model_input(self):
        """
        This function must return model input size. This will be used for rescaling the 
        model output to preview resolution while overlaying. 
          
        Returns:
            (w,h): A Tuple of width and height of the model input
        """
        return (96,96)
```

So the final implementation of the class should look like this:
```
from interface.output_decoder import OutputDecoderInterface 
from interface_utils.labelboundingboxscore import LabelBoundingBoxScore 

#import other python package
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

``` 
### 4. Create a folder named as same as the file name. E.g: MobilenetV2_person_det and put the MobilenetV2_person_det.py file inside the folder
So the final structure of the module should look like this:
```
MobilenetV2_person_det/  # folder
   '-> MobilenetV2_person_det.py  # python file
        '-> class MobilenetV2_person_det(OutputDecoderInterface):  # class name
```
**Note that all the three name should be the same or else the module will fail to load.**  


### Voila! We have created a custom post processing module. To add this module to the application, follow the steps discussed in [Add Post Processing Modules to edge-CAMView](#2.-Add-Post-Processing-Modules-to-edge-CAMView) session. 


**Note:** If you're using any other python dependencies packages which are not present in the application installation path (e.g. C:\Program Files (x86)\e-con Systems\edge-CAMView\ ). Then you need to add that python module to the installation folder. You can download the python package from the [pip](https://pypi.org/project/pip/) website. 
For example, you can see the nms package in the python_dependencies folder. If your post processing module imports nms, copy the nms folder and paste it in the application installation folder. The same should be done for any other dependencies. 

## Support

If you face any issue with loading the configurations, models or post processing modules onto the application. Create a issue in this repo. 

Please Don't create issue regarding the performance of the model. Since the models provided here is for only evaluation purposes and not production ready. 

**Note:** If you need any other support other than the scope of this repo, please contact us using the Live Chat option available on our website - https://www.e-consystems.com/ 

## Contributing
Forks of this library are welcome and encouraged. 

You have bug reports or fixes to contribute ?
You have implemented any custom post processing module which you think might help other users ? You have a python dependency package which you think needs to be added?

Please go ahead & create a pull request. We will validate the request and merge it.  

## License
This code is made available under the Apache 2 license.
