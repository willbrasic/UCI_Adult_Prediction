

# UCI Adult Prediction with Pytorch and Scikit-Learn

Hi there! Thank you for checking out my repository! This README.md file gives
details on the neural networks used for prediction while the REAME_2.md file
dives into using simpler machine learning algorithms for prediction to juxtapose
their effectiveness with that of deep learning.

## Overview

This repository contains code for deploying a PyTorch neural network in the UCI_Adult_PyTorch.py file
along with various machine learning algorithms in UCI_Adult_sklearn.py on the UCI Adult dataset
which can be found at UCI_Adult_Data.csv.

## Table of Contents

- [UCI Adult Prediction with PyTorch and sklearn](#project-name)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Results](#results)

## Features

The key features of my project are as follows:

- Neural network using PyTorch
- Logistic regression, random forest, and catboost using sklearn

## Getting Started

Provide instructions on how to get your project up and running on a user's local machine.

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
git clone https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn.git

# Navigate to the project directory
cd UCI_Adult_PyTorch_sklearn

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the Adult UCI dataset which can be found in the repository at
UCI_Adult_Data.csv. Details can be found at https://archive.ics.uci.edu/dataset/2/adult.
The data cleaning procedure can be found at UCI_Adult_PyTorch.py. Here are some graphs
that summarize the outcome of interest (income) along with the covariates used
for prediction

![Picture 1](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Picture_1.png)

![Picture 2](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Picture_2.png)

## Training

I create a two neural networks: one which only contains linear activations
and another more complex network with multiple hidden layers, BatchNorm, dropout,
and ELU (https://pytorch.org/docs/stable/generated/torch.nn.ELU.html) activations
which is a good alternative for ReLU that avoids non-differentiability at zero.
Both networks use Nesterov momentum with parameter γ = 0.9 to improve
optimization performance.

The training and validation loss and accuracy for the linear neural network
over epochs looks as follows:

![Picture 3](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Picture_3.png)


The training and validation los and accuracy for the non-linear neural network
over epochs looks as follows:

![Picture 4](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Picture_4.png)

As evidence by the accuracy over epochs for each model, they perform very similar.
However, the linear neural network is less prone to overfitting as indicated by
the smaller gap between training and validation loss curves relative to that of
the non-linear model. Thus, the linear model is selected for testing.

Given such a simple model seemingly works better, this indicates that more straightforward
prediction algorithm outside of deep learning may be more appropriate for this task.
This is explored in UCI_Adult_Scikit-Learn.py and README_2.md.

## Results

The linear neural network finishes with a testing accuracy of 79.12%. Here is
the confusion matrix:

![Picture 5](https://github.com/willbrasic/UCI_Adult_PyTorch_sklearn/blob/main/UCI_Adult_Pictures/UCI_Adult_Picture_5.png)
