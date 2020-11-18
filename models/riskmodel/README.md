# Risk Model
[![Documentation Status](https://readthedocs.org/projects/fairscale/badge/?version=latest)](https://fairscale.readthedocs.io/en/latest/?badge=latest) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/fairscale/blob/master/CONTRIBUTING.md)

A risk model is a statistical procedure for assigning to an individual a probability of developing a future adverse outcome in a given time period. The assignment is made 
by combining his or her values for a set of risk-determining covariates with incidence and mortality data and published estimates of the covariates’ effects on the outcome. 
Such risk models are playing increasingly important roles in the practise of medicine, as clinical care becomes more tailored to individual characteristics and needs. 
<img src="readme_images/fig1.jpg" alt="Figure 1.  Risk Model " width="" height="">

## Dataset 
The dataset is organized in 2 files NHANESI_X.csv/NHANESI_Y.csv . There are 9,933 examples .Using data from the prospective Nurses Health Study (NHS).
The National Health and Nutrition Examination Survey (NHANES) is a program of studies designed to assess the health and nutritional status of adults and children in the United States. 
NHANES contains comprehensive health  records  of  patients  from  the  state  of  Arizona linked  across  systems  and  time.

img src="readme_images/fig2.jpg" alt="Figure 2. Illustrative Examples of Risk Model dataset" width="" height="">

## Introduction

#### Directory Layout 
    .
    ├── data                                                            
    │   ├── NHANESI_X.csv                                               # feature data
    │   ├── NHANESI_Y.csv                                               # label data
    ├── src
    │   ├── config.py                                                   # contains all the configuration
    |   ├── utlis.py                                                    # program file for data preparation
    |   ├── predict.py                                                  # End-to-end, prediction file
    |   ├── train.py                                                    # training model 
    ├── static
    |   ├── secondpage.css                                              # CSS files
    |   ├── style.css                                                   # CSS files
    ├── templates
    |   ├── rkindex.html                                                # html file for the UI
    ├── weights
    |   ├── model.pkl                                                   # trained weights
    ├── rapp.py                                                         # web app file
    
#### Content
| Directory | Info |
|-----------|--------------|
| `src` | Contains all Python files |
| `templates` | Contains HTML file |
| `static` | Contains css, js files and images  |
| `data` | Contains [data](https://wwwn.cdc.gov/nchs/nhanes/default.aspx)   |
| `weights` | contains trained model |

## Evaluation 
The proposed approach was evaluated using Precision , Recall , Accuracy and F1 Score. Our source code is freely available here.
<img src="readme_images/fig3.jpeg" alt="Figure 3. Evaluation of Model " width="" height="">

## Prerequisites
* Python 3.4+
* PyTorch and its dependencies

## How to Install and Run
* Clone this repository and run in command prompt
```bash
pip install -r requirement.txt
``` 
* Run this to start server
```bash
python rapp.py
``` 
* Update `Features` and predict user has `risk` or not .

<img src="readme_images/fig4.jpg" alt=" Prediction of Model " width="" height="">


## Train your own model*
* For traning you need to run `train.py` in src directory.
* if want to change epochs, data directory, random seed, learning rate, etc change it from `config.py`.
