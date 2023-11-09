
# UCI Adult Prediction with Scikit-Learn

Hi there! Thank you for checking out my repository! This README_2.md file gives
details on the machine learning algorithms used for prediction while the REAME.md
file dives into using neural networks for prediction to juxtapose their
effectiveness with that of more conventional machine learning models.

## Overview

This repository contains code for deploying a PyTorch neural network in the UCI_Adult_PyTorch.py file
along with various machine learning algorithms in UCI_Adult_Scikit-Learn.py on the UCI Adult dataset
which can be found at UCI_Adult_Data.csv. Here, we focus on the latter. For details on
the PyTorch neural network, please look at the README.md file.

## Table of Contents

- [UCI Adult Prediction with Scikit-Learn](#project-name)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Results](#results)

## Getting Started

Below are some instructions on how to get the project up and running.

### Prerequisites

Main dependencies:

```bash
imbalanced_learn==0.11.0
imblearn==0.0
matplotlib==3.7.2
numpy==1.25.2
pandas==2.1.2
scikit_learn==1.3.0
seaborn==0.13.0
torch==2.0.1
```

### Installation

```bash
# Clone the repository
git clone https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn.git

# Navigate to the project directory
cd UCI_Adult_PyTorch_Scikit-Learn

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the Adult UCI dataset which can be found in the repository at
UCI_Adult_Data.csv. Details can be found at https://archive.ics.uci.edu/dataset/2/adult.
The data cleaning procedure can be found at UCI_Adult_Scikit-Learn.py. Here are some graphs
that summarize the outcome of interest (income) along with the covariates used
for prediction:

![Picture 1](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_Data_Summary_1.png)

![Picture 2](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_Data_Summary_2.png)

Also, the data does have a slight issue regarding class proportions with class 1
(individuals making more than $50,000) being under-sampled. I tested if SMOTE could
improve this. While recall did increase, overall validation accuracy decreased leading
me to not use this method as I prioritize accuracy in general over an increase in recall.

## Training

I create a variety models: logistic regression, lasso logistic regression, ridge logistic regression, elastic net logistic regression, and random forest. In an attempt to find the best hyperparameters for each model, I use a random search.
All models perform very similarly, but the random forest performs marginally better than the rest so I choose that algorithm to evaluate testing performance. I refrain from using boosting due to computational requirements.

Below are the learning curves for each model along with the validation curves for lasso and ridge for varying levels of the penalization parameter.

![Picture 3](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_LR_Learning_Curve.png)

![Picture 4](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_LR_Lasso_Learning_Curve.png)

![Picture 5](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_LR_Ridge_Learning_Curve.png)

![Picture 6](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_Lasso_Ridge_Validation_Curve.png)

![Picture 7](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_LR_Enet_Learning_Curve.png)

![Picture 8](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_RF_Learning_Curve.png)

## Results

Evaluating the random forest model on the testing data gives rise to the following confusion matrix:

![Picture 9](https://github.com/willbrasic/UCI_Adult_PyTorch_Scikit-Learn/blob/main/UCI_Adult_Scikit-Learn_Pictures/UCI_Adult_RF_CM.png)

The accuracy rate for this model is 82.24%.
