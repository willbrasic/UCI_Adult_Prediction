

# UCI Adult Prediction with Pytorch

Hi there! Thank you for checking out my repository! This README.md file gives
details on the neural networks used for prediction while the REAME_2.md file
dives into using simpler machine learning algorithms for prediction to juxtapose
their effectiveness with that of deep learning.

My chosen neural network (82.227% test accuracy rate), which is elucidated below,
performs considerably above the 0.75 quantile (79.037% test accuracy rate)
of all neural networks used for this dataset.
I hope you enjoy reading through my project!

## Overview

This repository contains code for deploying a PyTorch neural network in the UCI_Adult_PyTorch.py file
along with various machine learning algorithms in UCI_Adult_Scikit-Learn.py on the UCI Adult dataset
which can be found at UCI_Adult_Data.csv. Here, we focus on the former. For details on
the simpler machine learning models, please look at the README_2.md file.

## Table of Contents

- [UCI Adult Prediction with PyTorch](#project-name)
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
pandas==2.1.1
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
The data cleaning procedure can be found at UCI_Adult_PyTorch.py. Here are some graphs
that summarize the outcome of interest (income) along with the covariates used
for prediction:

![Picture 1](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Data_Summary_1.png)

![Picture 2](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Data_Summary_2.png)

Also, the data does have a slight issue regarding class proportions with class 0
(individuals making more than $50,000) being under-sampled. I tested if SMOTE could
improve this. While precision did increase, overall validation accuracy decreased leading
me to not use this method as I prioritize accuracy in general over an increase in precision.

For the sake of completeness, here is the confusion matrix when model_1,
which is discussed in the Training section, is used on the data when SMOTE is applied:

![Picture 3](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Smote_Model_1_CM.png)

## Training

I create three neural networks: model_0 which only contains linear activations,
model_1 which contains complex network with multiple hidden layers, BatchNorm, dropout,
and ELU, and model_2 which contains multiple hidden linear (no non-linear layers)
along with BatchNorm and dropout. ELU
(https://pytorch.org/docs/stable/generated/torch.nn.ELU.html) activations
are a good alternative for ReLU that avoids non-differentiability at zero.
All networks use learning rate α = 0.01 and
Nesterov momentum with parameter γ = 0.9 to improve optimization performance.

The training and validation loss and accuracy for model_0
over epochs along with its confusion matrix looks as follows:

![Picture 4](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Train_Val_Model_0.png)

![Picture 5](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Model_0_CM.png)

The mean training and validation accuracy rate for model_0 over thirty epochs
is 81.6424% and 81.8092%, respectively.

The training and validation loss and accuracy for model_1
over epochs along with its confusion matrix looks as follows:

![Picture 6](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Train_Val_Model_1.png)

![Picture 7](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Model_1_CM.png)

The mean training and validation accuracy rate for model_1 over thirty epochs
is 80.8788% and 82.5968%, respectively.

The training and validation loss and accuracy for model_2
over epochs along with its confusion matrix looks as follows:

![Picture 8](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Train_Val_Model_2.png)

![Picture 9](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Model_2_CM.png)

The mean training and validation accuracy rate for model_1 over thirty epochs
is 80.6205% and 81.8572%, respectively.

As evidenced by the accuracy over epochs for each model, the models perform very similar.
However, model_1 has slightly better validation accuracy by roughly 0.7
percentage points to that of model_0 and model_2.
Thus, model_1 is selected for testing.


## Results

The chosen model_1 has a testing accuracy of 82.227%. Here is its confusion matrix:

![Picture 10](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Model_1_Test_CM.png)
