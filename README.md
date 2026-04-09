# Pediatric Pneumonia Detection using Deep Learning

ENSF 617 Group 10

## Overview

This project investigates the performance of different deep learning architectures for **pediatric pneumonia detection** using chest X-ray images. The goal is to compare how model *design paradigms* (CNN vs modern CNN vs transformer) impact classification performance on a limited medical dataset.

## Models Evaluated

The following pretrained models were fine-tuned and compared:

* ResNet50 (CNN - residual architecture)
* DenseNet121 (CNN - dense connectivity)
* ConvNeXt V2 Tiny (modern CNN)
* Swin Transformer V2 Small (vision transformer)

## Dataset

* Pediatric chest X-ray dataset (Kaggle)
* Binary classification: **Pneumonia vs Normal**
* Standard preprocessing and augmentation applied as a fixed dataset

## Results

| Model       | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ----------- | -------- | --------- | ------ | -------- | ------- |
| ResNet50    | 0.9704   | 0.9720    | 0.9704 | 0.9707   | 0.9931  |
| DenseNet121 | 0.9772   | 0.9772    | 0.9772 | 0.9772   | 0.9933  |
| ConvNeXt V2 | 0.9783   | 0.9789    | 0.9783 | 0.9785   | 0.9971  |
| Swin V2     | 0.9795   | 0.9796    | 0.9795 | 0.9794   | 0.9972  |

---

## Contributors
Jason Chiu  
Jason Tieh  
Raghavi Ramasamy  
Waqas Ur Rehman  