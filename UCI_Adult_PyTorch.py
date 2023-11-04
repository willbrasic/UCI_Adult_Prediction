"""

UCI Adult PyTorch Neural Network

"""


__author__ = "William Brasic"
__email__ =  "wbrasic@arizona.edu"


"""

Preliminaries

"""


# importing necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# setting plot stype to ggplot
plt.style.use('ggplot')

# arrays are printed up to 6 digits and without scientific notation
np.set_printoptions(precision = 6, suppress = True)

# setting working directory
os.chdir('C:\\Users\\wbras\\OneDrive\\Desktop\\GitHub\\UCI_Adult_PyTorch_Scikit-Learn')


"""

Data Preprocessing

"""


# load in data
df = pd.read_csv('UCI_Adult_Data.csv')

# making income a binary variable
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# making gender a binary variable
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# classifying people into full and part time workers
df['hours-per-week'] = df['hours-per-week'].apply(lambda x: 1 if x >= 40 else 0)

# selecting only certain education classes
df = df.query('education == "HS-grad" or education == "Some-college" or education == "Bachelors" \
              or education == "Masters" or education == "Doctorate"')

# dropping '?' workclass and occupation
df.drop(df[df['workclass'] == '?'].index, inplace=True)
df.drop(df[df['occupation'] == '?'].index, inplace=True)
df.drop(df[df['native-country'] == '?'].index, inplace=True)

# one hot encode the education and race columns
new_df = pd.get_dummies(df, columns = ['education', 'race', 'workclass',
                                       'occupation', 'relationship', 'native-country'], dtype = int)

# dropping columns excluded from prediction
new_df.drop('fnlwgt educational-num marital-status \
          capital-gain capital-loss'.split(),
          axis = 1, inplace = True)

# standardize age covariate
new_df['age'] = ( df['age'] - df['age'].mean() ) / df['age'].std()

# The next few lines of code are to deal with our dataset having disproportionate classes
X = new_df.drop('income', axis = 1)
y = new_df['income']

# instantiate SMOTE class
smote = SMOTE(sampling_strategy = 'minority', random_state = 1024)

# use smote to resample the data
X_resampled, y_resampled = smote.fit_resample(X, y)

# inspecting new class proportions
(y_resampled.value_counts() / len(y_resampled)).round(4);

# checking for NaNs
X.isnull().values.any();
y.isnull().values.any();

# train and test split without SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 1024)

# train and validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    test_size = 0.25,
                                                    random_state = 1024)

# convert features and labels to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# creating training, evaluation, and tensor datasets
train_data = TensorDataset(X_train_tensor, y_train_tensor)
val_data = TensorDataset(X_val_tensor, y_val_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

# set the batch size hyperparameter
batch_size = 32

# turn training data into iterables (batches)
train_dataloader = DataLoader(train_data,
    batch_size= batch_size,
    shuffle = True
)

# turn validation data into iterables (batches)
val_dataloader = DataLoader(val_data,
    batch_size= batch_size,
    shuffle = True
)

# turn testing data into iterables
test_dataloader = DataLoader(test_data,
    batch_size = batch_size,
    shuffle = False # don't necessarily have to shuffle the testing data
)

# checking out dataloaders
print(f"Dataloaders: {train_dataloader, val_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")
print(f"Length of val dataloader: {len(val_dataloader)} batches of {batch_size}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")


"""

Data Visualization Using Original Data

"""


# figure size for the two subplots below
plt.figure(figsize = (10, 10))

# countplot of income grouped by gender
plt.subplot(2, 1, 1)
sns.countplot(data = df, x = 'income', hue = 'gender').set_xticklabels(['<=50K', '>50K'])
plt.title('Count Plot of Income by Gender')
plt.xlabel('Income')
plt.ylabel('Count')
plt.legend(title = 'Gender', loc = 'upper right', labels = ['Female', 'Male'])

# kde plot for age grouped by income
plt.subplot(2, 1, 2)
sns.kdeplot(data = df, x = 'age', hue = 'income')
plt.title("KDE Plot for Age by Income")
plt.xlabel('Age')
plt.legend(title = 'Income', loc = 'upper right', labels = ['>50K', '<=50K'])

# show the above plots
plt.tight_layout()
plt.show()

# countplot of income grouped by occupation
sns.countplot(data = df, x = 'income', hue = 'occupation')
plt.title('Count Plot of Income by Occupation')
plt.xlabel('Income')
plt.ylabel('Count')
plt.xticks(range(2), ['<= 50K', '>50K'])
plt.legend(title = 'Occupation', bbox_to_anchor=(1.25, 1), borderaxespad=0)
plt.show()


"""

Building Neural Networks

"""


# creating linear neural network
class nn_model_0(nn.Module):
    # input shape is number of features
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = input_shape, out_features = hidden_units),
            nn.Linear(in_features = hidden_units, out_features = output_shape)
        )

    # defining the forward pass
    def forward(self, x):
        return self.layer_stack(x)


# creating non-linear neural network
class nn_model_1(nn.Module):
    # input shape is number of features
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = input_shape, out_features = hidden_units),
            nn.ELU(),
            nn.BatchNorm1d(num_features = hidden_units),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.ELU(),
            nn.BatchNorm1d(num_features = hidden_units),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = hidden_units, out_features = output_shape),
            nn.ELU()
        )

    # defining the forward pass
    def forward(self, x):
        return self.layer_stack(x)


# creating linear neural network similar to nn_model_2
class nn_model_2(nn.Module):
    # input shape is number of features
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = input_shape, out_features = hidden_units),
            nn.BatchNorm1d(num_features = hidden_units),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.BatchNorm1d(num_features = hidden_units),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = hidden_units, out_features = output_shape)
        )

    # defining the forward pass
    def forward(self, x):
        return self.layer_stack(x)

# instantiate nn_model_0 class
model_0 = nn_model_0(input_shape = X_train_tensor.shape[1], # input should be shape of feature matrix
    hidden_units = 25, # how many units in the hiden layer
    output_shape = 1 # out features is one because we have a binary classifcation problem
)

# instantiate nn_model_1 class
model_1 = nn_model_1(input_shape = X_train_tensor.shape[1],
    hidden_units = 25,
    output_shape = 1
)

# instantiate nn_model_1 class
model_2 = nn_model_2(input_shape = X_train_tensor.shape[1],
    hidden_units = 25,
    output_shape = 1
)


"""

Neural Network Model 0 Creation

"""


# defining the loss function
loss_fn = nn.BCEWithLogitsLoss()

# defining the optimizer (SGD with momentum seemed to work best)
optimizer_0 = torch.optim.SGD(params = model_0.parameters(), lr = 0.01, momentum = 0.9, nesterov = True)

# set seed for reproducibility
torch.manual_seed(1024)

# delete the 'logs' directory before running training and validation loop to avoid avoid TensorBoard plot issue
import shutil
if os.path.exists('logs/') and os.path.isdir('logs/'):
    shutil.rmtree('logs/')

# import TensorBoard
from torch.utils.tensorboard import SummaryWriter

# create a directory to store TensorBoard logs
log_dir = 'logs/'
os.makedirs(log_dir, exist_ok = True)

# initialize TensorBoard writer
writer = SummaryWriter(log_dir = log_dir)

# set the number of epochs (how many times the model will pass over the training data)
epochs = 30

# lists to store training and validation loss, accuracy, predictions, and true labels
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
y_pred_tracker = []
y_true_tracker = []

# parameters needed for early stopping
best_val_loss = float('inf') # initalize to large value
epochs_without_improvement = 0 # keeps track of number of epochs were validation loss has not improved
patience = 10 # once validation loss has not improved for more than 10 epochs, break validation loop

# create training and testing loop for model_0
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-------")

    # parameters to keep track of training over epochs
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_accuracy = 0

    # set the model to train mode
    model_0.train()

    # loop over each batch of data
    for batch, (X, y) in enumerate(train_dataloader):

        # forward pass (model outputs raw logits)
        y_logits = model_0(X).squeeze()

        # convert logits to predictions
        y_pred = torch.round(torch.sigmoid(y_logits))

        # calculate loss
        loss = loss_fn(y_logits, y)

        # optimizer zero grad
        optimizer_0.zero_grad()

        # backward pass
        loss.backward()

        # update parameters
        optimizer_0.step()

        # accumulate batch loss
        train_loss += loss.item()

        # accumulate correct predictions
        train_correct += (y_pred == y).sum().item()

        # accumulate total number of training examples (increases by 32 each time)
        train_total += len(y)

    # divide total train loss by length of train dataloader (average loss per batch for the epoch)
    train_loss /= len(train_dataloader)

    # append the training loss for the epoch to the train_losses list
    train_losses.append(train_loss)

    # write the training loss to file (used for TensorBoard)
    writer.add_scalar('Loss/Train', train_loss, epoch)

    # calculate training accuracy for the epoch
    train_accuracy = 100 * train_correct / train_total

    # append the accuracy for the epoch to the train_accur list
    train_accuracies.append(train_accuracy)

    # write the training accuracy to a file (used for TensorBoard)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

    # parameters to keep track of validation over epochs
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_accuracy = 0

    # set the model to evaluation model
    model_0.eval()

    # begin the inference procedure
    with torch.inference_mode():
        # loop through each batch of data in the validation set
        for batch, (X, y) in enumerate(val_dataloader):

            # forward pass
            y_logits = model_0(X).squeeze()

            # convert logits to binary predictions (0 or 1)
            y_pred = torch.round(torch.sigmoid(y_logits))

            # accumulate validation loss
            val_loss += loss_fn(y_logits, y).item()

            # accumulate correct predictions
            val_correct += (y_pred == y).sum().item()

            # accumulate the length of the batch size (increases by 32 each time)
            val_total += len(y)

        # check if the validation loss has improved for this epoch
        if val_loss < best_val_loss:
            # update the best validation loss
            best_val_loss = val_loss
            # reset the counter
            counter = 0
        # if validation loss did not improve, increase the counter
        else:
            counter += 1

        # if early stopping criteria has been met, break the loop
        if counter >= patience:
            print(f'Early stopping: No improvement in validation loss for {patience} epochs.')
            break

        # divide total validation loss by length of validation dataloader (average loss per batch for epoch)
        val_loss /= len(val_dataloader)

        # append the validation loss for the epoch to the train_losses list
        val_losses.append(val_loss)

        # write the validation loss to file (used for TensorBoard)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # Calculate validation accuracy for the epoch
        val_accuracy = 100 * val_correct / val_total

        # append the accuracy for the epoch to the train_accur list
        val_accuracies.append(val_accuracy)

        # convert y_pred from a tensor to a numpy array
        y_pred_numpy = y_pred.numpy()

        # convert true y label from tensor to numpy array
        y_true_numpy = y.numpy()

        # append the predictions for the batch to the y_pred_tracker list
        y_pred_tracker.extend(y_pred_numpy)

        # append the true labels for the batch to the y_true_tracker list
        y_true_tracker.extend(y_true_numpy)

        # write the validation accuracy to file (used for TensorBoard)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    # print training loss and accuracy for each epoch
    print(f'Train loss: {train_loss:.5f} | Training acc: {train_accuracy:.2f} %\n')

    # print validation loss and accuracy for each epoch
    print(f'Validation loss: {val_loss:.5f} |  Validation acc: {val_accuracy:.2f} %\n')

# mean training loss and accuracy
print(f'The mean training loss and accuracy over {epochs} epochs is {np.array(train_losses).mean().round(4)} \
    and {np.array(train_accuracies).mean().round(4)}, respectively.')

# mean validation loss and accuracy
print(f'The mean validation loss and accuracy over {epochs} epochs is {np.array(val_losses).mean().round(4)} \
    and {np.array(val_accuracies).mean().round(4)}.')

# plot training loss and validation loss
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(range(len(train_losses)), train_losses, label = 'Training Loss', color = 'blue')
plt.plot(range(len(val_losses)), val_losses, label = 'Validation Loss', color = 'red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

# plot training accuracy and validation accuracy
plt.subplot(2, 1, 2)
plt.plot(range(len(train_accuracies)), train_accuracies, label = 'Training Accuracy', color = 'blue')
plt.plot(range(len(val_accuracies)), val_accuracies, label = 'Validation Accuracy', color = 'red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

# show the above plots
plt.tight_layout()
plt.show()

# create confusion matrix
conf_matrix = confusion_matrix(y_true_tracker, y_pred_tracker)

# create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


"""

Neural Network Model 1 Creation

"""


# defining the loss function
loss_fn = nn.BCEWithLogitsLoss()

# defining the optimizer (SGD with momentum seemed to work best)
optimizer_1 = torch.optim.SGD(params = model_1.parameters(), lr = 0.01, momentum = 0.9, nesterov = True)

# set seed for reproducibility
torch.manual_seed(1024)

# delete the 'logs' directory before running training and validation loop to avoid avoid TensorBoard plot issue
import shutil
if os.path.exists('logs/') and os.path.isdir('logs/'):
    shutil.rmtree('logs/')

# import TensorBoard
from torch.utils.tensorboard import SummaryWriter

# create a directory to store TensorBoard logs
log_dir = 'logs/'
os.makedirs(log_dir, exist_ok = True)

# initialize TensorBoard writer
writer = SummaryWriter(log_dir = log_dir)

# set the number of epochs (how many times the model will pass over the training data)
epochs = 30

# lists to store training and validation loss, accuracy, predictions, and true labels
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
y_pred_tracker = []
y_true_tracker = []

# parameters needed for early stopping
best_val_loss = float('inf') # initalize to large value
epochs_without_improvement = 0 # keeps track of number of epochs were validation loss has not improved
patience = 10 # once validation loss has not improved for more than 10 epochs, break validation loop

# create training and testing loop for model_1
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-------")

    # parameters to keep track of training over epochs
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_accuracy = 0

    # set the model to train mode
    model_1.train()

    # loop over each batch of data (recall the batch size of 32)
    for batch, (X, y) in enumerate(train_dataloader):

        # forward pass (model outputs raw logits)
        y_logits = model_1(X).squeeze()

        # convert logits to binary predictions (0 or 1)
        y_pred = torch.round(torch.sigmoid(y_logits))

        # calculate loss
        loss = loss_fn(y_logits, y)

        # optimizer zero grad
        optimizer_1.zero_grad()

        # backward pass
        loss.backward()

        # update parameters
        optimizer_1.step()

        # accumulate batch loss
        train_loss += loss.item()

        # accumulate correct predictions
        train_correct += (y_pred == y).sum().item()

        # accumulate total number of training examples (increases by 32 each time)
        train_total += len(y)

    # divide total train loss by length of train dataloader (average loss per batch for the epoch)
    train_loss /= len(train_dataloader)

    # append the training loss for the epoch to the train_losses list
    train_losses.append(train_loss)

    # write the training loss to file (used for TensorBoard)
    writer.add_scalar('Loss/Train', train_loss, epoch)

    # calculate training accuracy for the epoch
    train_accuracy = 100 * train_correct / train_total

    # append the accuracy for the epoch to the train_accur list
    train_accuracies.append(train_accuracy)

    # write the training accuracy to a file (used for TensorBoard)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

    # parameters to keep track of validation over epochs
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_accuracy = 0

    # set the model to evaluation model
    model_1.eval()

    # begin the inference procedure
    with torch.inference_mode():
        # loop through each batch of data in the validation set
        for batch, (X, y) in enumerate(val_dataloader):

            # forward pass
            y_logits = model_1(X).squeeze()

            # convert logits to binary predictions (0 or 1)
            y_pred = torch.round(torch.sigmoid(y_logits))

            # accumulate validation loss
            val_loss += loss_fn(y_logits, y).item()

            # accumulate correct predictions
            val_correct += (y_pred == y).sum().item()

            # accumulate the length of the batch size (increases by 32 each time)
            val_total += len(y)

        # check if the validation loss has improved for this epoch
        if val_loss < best_val_loss:
            # update the best validation loss
            best_val_loss = val_loss
            # reset the counter
            counter = 0
        # if validation loss did not improve, increase the counter
        else:
            counter += 1

        # if early stopping criteria has been met, break the loop
        if counter >= patience:
            print(f'Early stopping: No improvement in validation loss for {patience} epochs.')
            break

        # divide total validation loss by length of validation dataloader (average loss per batch for epoch)
        val_loss /= len(val_dataloader)

        # append the validation loss for the epoch to the train_losses list
        val_losses.append(val_loss)

        # write the validation loss to file (used for TensorBoard)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # Calculate validation accuracy for the epoch
        val_accuracy = 100 * val_correct / val_total

        # append the accuracy for the epoch to the train_accur list
        val_accuracies.append(val_accuracy)

        # convert y_pred from a tensor to a numpy array
        y_pred_numpy = y_pred.numpy()

        # convert true y label from tensor to numpy array
        y_true_numpy = y.numpy()

        # append the predictions for the batch to the y_pred_tracker list
        y_pred_tracker.extend(y_pred_numpy)

        # append the true labels for the batch to the y_true_tracker list
        y_true_tracker.extend(y_true_numpy)

        # write the validation accuracy to file (used for TensorBoard)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    # print training loss and accuracy for each epoch
    print(f'Train loss: {train_loss:.5f} | Training acc: {train_accuracy:.2f} %\n')

    # print validation loss and accuracy for each epoch
    print(f'Validation loss: {val_loss:.5f} |  Validation acc: {val_accuracy:.2f} %\n')

# mean training loss and accuracy
print(f'The mean training loss and accuracy over {epochs} epochs is {np.array(train_losses).mean().round(4)} \
    and {np.array(train_accuracies).mean().round(4)}, respectively.')

# mean validation loss and accuracy
print(f'The mean validation loss and accuracy over {epochs} epochs is {np.array(val_losses).mean().round(4)} \
    and {np.array(val_accuracies).mean().round(4)}.')

# plot training loss and validation loss
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(range(len(train_losses)), train_losses, label = 'Training Loss', color = 'blue')
plt.plot(range(len(val_losses)), val_losses, label = 'Validation Loss', color = 'red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

# plot training accuracy and validation accuracy
plt.subplot(2, 1, 2)
plt.plot(range(len(train_accuracies)), train_accuracies, label = 'Training Accuracy', color = 'blue')
plt.plot(range(len(val_accuracies)), val_accuracies, label = 'Validation Accuracy', color = 'red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

# show the above plots
plt.tight_layout()
plt.show()

# create confusion matrix
conf_matrix = confusion_matrix(y_true_tracker, y_pred_tracker)

# create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



"""

Neural Network Model 2 Creation

"""


# defining the loss function
loss_fn = nn.BCEWithLogitsLoss()

# defining the optimizer (SGD with momentum seemed to work best)
optimizer_2 = torch.optim.SGD(params = model_2.parameters(), lr = 0.01, momentum = 0.9, nesterov = True)

# set seed for reproducibility
torch.manual_seed(1024)

# delete the 'logs' directory before running training and validation loop to avoid avoid TensorBoard plot issue
import shutil
if os.path.exists('logs/') and os.path.isdir('logs/'):
    shutil.rmtree('logs/')

# import TensorBoard
from torch.utils.tensorboard import SummaryWriter

# create a directory to store TensorBoard logs
log_dir = 'logs/'
os.makedirs(log_dir, exist_ok = True)

# initialize TensorBoard writer
writer = SummaryWriter(log_dir = log_dir)

# set the number of epochs (how many times the model will pass over the training data)
epochs = 30

# lists to store training and validation loss, accuracy, predictions, and true labels
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
y_pred_tracker = []
y_true_tracker = []

# parameters needed for early stopping
best_val_loss = float('inf') # initalize to large value
epochs_without_improvement = 0 # keeps track of number of epochs were validation loss has not improved
patience = 10 # once validation loss has not improved for more than 10 epochs, break validation loop

# create training and testing loop for model_2
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-------")

    # parameters to keep track of training over epochs
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_accuracy = 0

    # set the model to train mode
    model_2.train()

    # loop over each batch of data (recall the batch size of 32)
    for batch, (X, y) in enumerate(train_dataloader):

        # forward pass (model outputs raw logits)
        y_logits = model_2(X).squeeze()

        # convert logits to binary predictions (0 or 1)
        y_pred = torch.round(torch.sigmoid(y_logits))

        # calculate loss
        loss = loss_fn(y_logits, y)

        # optimizer zero grad
        optimizer_2.zero_grad()

        # backward pass
        loss.backward()

        # update parameters
        optimizer_2.step()

        # accumulate batch loss
        train_loss += loss.item()

        # accumulate correct predictions
        train_correct += (y_pred == y).sum().item()

        # accumulate total number of training examples (increases by 32 each time)
        train_total += len(y)

    # divide total train loss by length of train dataloader (average loss per batch for the epoch)
    train_loss /= len(train_dataloader)

    # append the training loss for the epoch to the train_losses list
    train_losses.append(train_loss)

    # write the training loss to file (used for TensorBoard)
    writer.add_scalar('Loss/Train', train_loss, epoch)

    # calculate training accuracy for the epoch
    train_accuracy = 100 * train_correct / train_total

    # append the accuracy for the epoch to the train_accur list
    train_accuracies.append(train_accuracy)

    # write the training accuracy to a file (used for TensorBoard)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

    # parameters to keep track of validation over epochs
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_accuracy = 0

    # set the model to evaluation model
    model_2.eval()

    # begin the inference procedure
    with torch.inference_mode():
        # loop through each batch of data in the validation set
        for batch, (X, y) in enumerate(val_dataloader):

            # forward pass
            y_logits = model_2(X).squeeze()

            # convert logits to binary predictions (0 or 1)
            y_pred = torch.round(torch.sigmoid(y_logits))

            # accumulate validation loss
            val_loss += loss_fn(y_logits, y).item()

            # accumulate correct predictions
            val_correct += (y_pred == y).sum().item()

            # accumulate the length of the batch size (increases by 32 each time)
            val_total += len(y)

        # check if the validation loss has improved for this epoch
        if val_loss < best_val_loss:
            # update the best validation loss
            best_val_loss = val_loss
            # reset the counter
            counter = 0
        # if validation loss did not improve, increase the counter
        else:
            counter += 1

        # if early stopping criteria has been met, break the loop
        if counter >= patience:
            print(f'Early stopping: No improvement in validation loss for {patience} epochs.')
            break

        # divide total validation loss by length of validation dataloader (average loss per batch for epoch)
        val_loss /= len(val_dataloader)

        # append the validation loss for the epoch to the train_losses list
        val_losses.append(val_loss)

        # write the validation loss to file (used for TensorBoard)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # Calculate validation accuracy for the epoch
        val_accuracy = 100 * val_correct / val_total

        # append the accuracy for the epoch to the train_accur list
        val_accuracies.append(val_accuracy)

        # convert y_pred from a tensor to a numpy array
        y_pred_numpy = y_pred.numpy()

        # convert true y label from tensor to numpy array
        y_true_numpy = y.numpy()

        # append the predictions for the batch to the y_pred_tracker list
        y_pred_tracker.extend(y_pred_numpy)

        # append the true labels for the batch to the y_true_tracker list
        y_true_tracker.extend(y_true_numpy)

        # write the validation accuracy to file (used for TensorBoard)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    # print training loss and accuracy for each epoch
    print(f'Train loss: {train_loss:.5f} | Training acc: {train_accuracy:.2f} %\n')

    # print validation loss and accuracy for each epoch
    print(f'Validation loss: {val_loss:.5f} |  Validation acc: {val_accuracy:.2f} %\n')

# mean training loss and accuracy
print(f'The mean training loss and accuracy over {epochs} epochs is {np.array(train_losses).mean().round(4)} \
    and {np.array(train_accuracies).mean().round(4)}, respectively.')

# mean validation loss and accuracy
print(f'The mean validation loss and accuracy over {epochs} epochs is {np.array(val_losses).mean().round(4)} \
    and {np.array(val_accuracies).mean().round(4)}.')

# plot training loss and validation loss
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(range(len(train_losses)), train_losses, label = 'Training Loss', color = 'blue')
plt.plot(range(len(val_losses)), val_losses, label = 'Validation Loss', color = 'red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

# plot training accuracy and validation accuracy
plt.subplot(2, 1, 2)
plt.plot(range(len(train_accuracies)), train_accuracies, label = 'Training Accuracy', color = 'blue')
plt.plot(range(len(val_accuracies)), val_accuracies, label = 'Validation Accuracy', color = 'red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

# show the above plots
plt.tight_layout()
plt.show()

# create confusion matrix
conf_matrix = confusion_matrix(y_true_tracker, y_pred_tracker)

# create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



"""

Neural Network Testing with Model 1

"""


# defining the loss function
loss_fn = nn.BCEWithLogitsLoss()

# parameters needed for early stopping
best_val_loss = float('inf')
epochs_without_improvement = 0
patience = 10

# set seed for reproducibility
torch.manual_seed(1024)

# initialize values to keep track of testing
test_loss = 0
test_correct = 0
test_total = 0
test_accuracy = 0
y_pred_tracker = []
y_true_tracker = []

# set model to evaluation mode
model_1.eval()

# begin the inference procedure
with torch.inference_mode():
    # loop through each batch of data in the test set
    for batch, (X, y) in enumerate(test_dataloader):

        # calculate the predicted logits
        y_logits = model_1(X).squeeze()

        # convert logits to binary predictions (0 or 1)
        y_pred = torch.round(torch.sigmoid(y_logits))

        # accumulate the loss
        test_loss += loss_fn(y_logits, y).item()

        # accumulate correct predictions
        test_correct += (y_pred == y).sum().item()

        # accumulate the the total of observations tested this batch
        test_total += len(y)

        # convert y_pred from a tensor to a numpy array
        y_pred_numpy = y_pred.numpy()

        # convert true y label from tensor to numpy array
        y_true_numpy = y.numpy()

        # append the predictions for the batch to the y_pred_tracker list
        y_pred_tracker.extend(y_pred_numpy)

        # append the true labels for the batch to the y_true_tracker list
        y_true_tracker.extend(y_true_numpy)

    # calculate testing loss
    test_loss /= len(test_dataloader)

    # calculate testing accuracy for the epoch
    test_accuracy = 100 * test_correct / test_total

# printing testing loss and accuracy
print(f'Test loss: {test_loss:.5f} | Testing acc: {test_accuracy:.4f}%\n')

# create confusion matrix
conf_matrix = confusion_matrix(y_true_tracker, y_pred_tracker)

# create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
