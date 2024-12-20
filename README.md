# Investigation of Class Separability within Object Detection Models in Histopathology

> Object detection is one of the most common tasks in histopathological image analysis and generalization is a key requirement for the clinical applicability of deep object detection models.  
However, traditional evaluation metrics often fail to provide insights into why models fail on certain test cases, especially in the presence of domain shifts. In this work, we propose a novel quantitative method for assessing the discriminative power of a model's latent space. Our approach, applicable to all object detection models with known local correspondences  such as the popular RetinaNet, FCOS, or YOLO approaches, allows tracing discrimination across layers and coordinates.  We motivate, adapt, and evaluate two suitable metrics, the generalized discrimination value and the Hellinger distance, and incorporate them into our approach.
Through empirical validation on real-world histopathology datasets, we demonstrate the effectiveness of our method in capturing model discrimination properties and providing insights for architectural optimization. This work contributes to bridging the gap between model performance evaluation and understanding the underlying mechanisms influencing model behavior.

<div align="center">
    <a href="./">
        <img src="./figure/overview_feature_extraction.png" width="95%"/>
    </a>
</div>

## Organization of this repository

This repository contains the code to generate the metrics presented in our paper using the YOLOv7 architecture. The repository is based on the official YOLOv7 implementation. We provide additional code for the training and evaluation on histopathology datasets and to construct the presented metrics. 

- `extract_features.py` contains the code to extract the features from the datasets used in our study.
- `compute_distance.py` contains the code to construct the separability metrics.

A demo to construct the metrics on the official MIDOG 2022 training set will be available soon. 


## Getting started

All requirements needed to run the scripts in this repository can be installed using pip:

```pip install -r requirements.txt```

## Citation