

# UCI Adult Prediction with Pytorch

Hi there! Thank you for checking out my repository! This README.md file gives
details on the neural networks used for prediction while the REAME_2.md file
dives into using simpler machine learning algorithms for prediction to juxtapose
their effectiveness with that of deep learning.

## Overview

This repository contains code for deploying a PyTorch neural network in the UCI_Adult_PyTorch.py file
along with various machine learning algorithms in UCI_Adult_sklearn.py on the UCI Adult dataset
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

Also, the data does have a slight issue with class proportions with class 0
(individuals making less than $50,000) being undersampled. I tested if SMOTE could
improve this. While precision did increase, overall validation accuracy decreased leading
me to not use this method as I prioritize accuracy in general over an increase in precision. 

## Training

I create three neural networks: model_0 which only contains linear activations,
model_1 which contains complex network with multiple hidden layers, BatchNorm, dropout,
and ELU, and model_2 which contains multiple hidden linear
hidden layers (no non-linear layers) along with BatchNorm and dropout. ELU
(https://pytorch.org/docs/stable/generated/torch.nn.ELU.html) activations
are a good alternative for ReLU that avoids non-differentiability at zero.
All networks use Nesterov momentum with parameter γ = 0.9 to improve
optimization performance.

The training and validation loss and accuracy for model_0
over epochs along with its confusion matrix looks as follows:

![Picture 3](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Train_Val_Model_0.png)



The training and validation loss and accuracy for the non-linear neural network
over epochs looks as follows:


As evidenced by the accuracy over epochs for each model, they perform very similar.
However, the linear neural network is less prone to overfitting as indicated by
the smaller gap between training and validation loss curves relative to that of
the non-linear model. Thus, the linear model is selected for testing.

Given such a simple model seemingly works better, this indicates that a more straightforward
prediction algorithm outside of deep learning may be more appropriate for this task.
This is explored in UCI_Adult_Scikit-Learn.py and with details at README_2.md.

## Results

The linear neural network finishes with a testing accuracy of 79.12%. Here is
the confusion matrix:
