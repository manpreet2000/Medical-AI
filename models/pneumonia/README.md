# Pneumonia Detection
[![Documentation Status](https://readthedocs.org/projects/fairscale/badge/?version=latest)](https://fairscale.readthedocs.io/en/latest/?badge=latest) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/fairscale/blob/master/CONTRIBUTING.md)

<!--
[![Contributors][https://img.shields.io/github/contributors/manpreet2000/Medical-AI.svg?style=flat-square]][https://github.com/manpreet2000/Medical-AI/graphs/contributors]
[![Forks][https://img.shields.io/github/forks/manpreet2000/Medical-AI.svg?style=flat-square]][https://github.com/manpreet2000/Medical-AI/network/members]
[![Stargazers][https://img.shields.io/github/stars/manpreet2000/Medical-AI.svg?style=flat-square]][https://github.com/manpreet2000/Medical-AI/stargazers]
[![Issues][https://img.shields.io/github/issues/manpreet2000/Medical-AI.svg?style=flat-square]](https://github.com/manpreet2000/Medical-AI/issues)
 -->
Pneumonia is the leading cause of death among young children and one of the top mortality causes worldwide. The pneumonia detection is usually performed through examine of chest X-Ray radiograph by highly trained specialists. This process is tedious and often leads to a disagreement between radiologists. Computer-aided diagnosis systems showed potential for improving the diagnostic accuracy. In this work, we develop the computational approach for pneumonia regions detection based on single-shot detectors, squeeze-and-extinction deep convolution neural networks and augmentations. The proposed approach was evaluated using Precision , Recall , Accuracy and F1 Score. Our source code is freely available here.

## Dataset 
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.
<img src="image.png" alt="Figure 1. Illustrative Examples of Chest X-Rays in Patients with Pneumonia" width="" height="">

## Introduction
#### Directory Layout 4
    .
    ├── notebook
    │   ├── x-ray-image-classification-using-pytorch.ipynb              # notebooks related to `EDA` and `experiment`
    ├── src
    │   ├── config.py                                                   # contains all the configuration
    |   ├── plot_me.py                                                  # program file for visualisation of dataset 
    |   ├── predict.py                                                  # End-to-end, prediction file
    |   ├── train.py                                                    # training model 
    ├── static
    |   ├── inputImage.jpg                                              # input image
    ├── templates
    |   ├── pneindex.html                                               # html file for the UI
    ├── weights
    |   ├── pne.pt                                                      # trained weights
    ├── pneapp.py                                                       # web app file

  

#### Content
| Directory | Info |
|-----------|--------------|
| `notebooks` | Contains all jupyter notebooks related to `EDA` and `experiment` |
| `src` | Contains all Python files |
| `templates` | Contains HTML file |
| `static` | Contains css, js files and images  |
| `data` | Contains [data](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) which is hidden  |

## How to Install and Run
* Clone this repository and run in command prompt
```bash
pip install -r requirement.txt
``` 
* Run this to start server
```bash
python pneapp.py
``` 
* Update `X-Ray` image and predict if user has `pnemonia` or not
<img src="image.png" alt="Figure 2. Prediction Pneumonia/Normal " width="" height="">


