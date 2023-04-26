# Model Card for Gender classification

## Gender classification:

Gender classification based on images is a truly remarkable application of technology that has the potential to revolutionize the way video analytics is used in various industries. In particular, it can greatly enhance the customer experience in retail, defense and tourist locations by providing tailored marketing, safety, and information. 

##### Use Cases:

- In retail, for example, businesses can use gender classification to show targeted advertisements and promotions to customers. This can lead to higher conversion rates and increased sales.
- In defense, gender classification can be used for surveillance and security purposes. By identifying the gender of individuals in a specific area, security personnel can quickly assess potential threats and respond accordingly.

## Model Description:

**Model type**: Binary Classification

**Model Architecture:** SSR-Net takes
a coarse-to-fine strategy and performs multi-class
classification with multiple stages. Each stage is
only responsible for refining the decision of its previous stage for more accurate gender classification. Thus,
each stage performs a task with few classes and
requires few neurons, greatly reducing the model
size.

**Model Inputs:** Image [64 X 64 X 3]

**Model Outputs:** Within each analyzed image frame, the model output include: -
- Single Value [0.0, 1.0]: ‘MALE’ else ‘FEMALE’

**Note:** This is a pretrained model from https://github.com/shamangary/SSR-Net repository. This is not a production ready model and it is meant to be used for demo purposes only.


## Performance Metrics:

Overall model performance, were assessed, including:

- The area under the ROC curve (AUC) is a commonly used metric for evaluating the performance of gender classification models, including the SSR-Net model. This is because the AUC is a robust and informative metric that can be used to evaluate the performance of binary classification models, such as gender classification.
- The ROC curve is a plot of the true positive rate (sensitivity) against the false positive rate (1-specificity) at different classification thresholds. The AUC is the area under this curve, which ranges between 0 and 1. A higher AUC value indicates better performance of the model.

More information on performance benchmark refer: https://github.com/shamangary/SSR-Net


## Size and Latency:

| Model Type | Size | Inference Time (on EdgeECAM50_USB)|
| ---------- | ----- | ------ |
|   Float32  | 191 kB | 658 ms |


## Limitations:
The following factors may degrade the model’s performance.
- Lighting conditions: Over exposure and under exposure would hide most facial features.
- Occlusions and Pose: Occlusions such as glasses, hats, or scarfs that can hide important facial features. Also out of plane pose would make the network ineffective.
- Bias: The dataset may be biased towards certain age groups and ethnicities, which can affect the model's ability to generalize to unseen data.
- Over-fitting: If the model is over-fitting to the training data, it may perform poorly on unseen test data.
- This is not a production ready model, to be used only for demo purposes.

## Dataset:

The dataset used is called Morph2 dataset.

Summary:
- The Morph2 dataset contains over 55,000 images of faces with a wide range of ages, from infants to elderly adults.
- Split into 80:20 ratios for Training, Validation sets
- The dataset contains a high degree of variability in terms of age, ethnicity, and gender. Additionally, the dataset includes images of individuals with different facial features, hairstyles, and occlusions such as glasses, hats, and scarfs, which can make it challenging for the model to accurately predict the age of a person.

Annotations:
- The gender of the individuals in the images were annotated manually by human annotators based on the visual appearance of the face.


## Credits:
```
@inproceedings{ijcai2018p150,
  title     = {SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation},
  author    = {Tsun-Yi Yang and Yi-Hsuan Huang and Yen-Yu Lin and Pi-Cheng Hsiu and Yung-Yu Chuang},
  booktitle = {Proceedings of the Twenty-Seventh International Joint Conference on
               Artificial Intelligence, {IJCAI-18}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {1078--1084},
  year      = {2018},
  month     = {7},
  doi       = {10.24963/ijcai.2018/150},
  url       = {https://doi.org/10.24963/ijcai.2018/150},
}
```

