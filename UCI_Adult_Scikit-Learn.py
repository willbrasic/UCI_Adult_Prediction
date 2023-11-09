""" 

The University of Arizona
INFO 557
Homework 3

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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import learning_curve, validation_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# arrays are printed up to 6 digits and without scientific notation
np.set_printoptions(precision = 6, suppress = True)

# setting working directory
os.chdir(r'C:\Users\wbras\OneDrive\Desktop\GitHub\UCI_Adult_PyTorch_Scikit-Learn')

# setting plot stype to ggplot
plt.style.use('ggplot')


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

# the next few lines of code are to deal with our dataset having disproportionate classes
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

Building Logistic Regression Model and Making Plots

"""


# building logistic regression pipline
lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state = 1024))

# getting cross validation scores for logistic regression without penalty term
scores = cross_val_score(estimator = lr_pipeline,
                         X = X_train,
                         y = y_train,
                         cv = 5,
                         n_jobs = -1)
print(f'CV accuracy scores: {scores}')
print(f'CV accuracy has mean {np.mean(scores):.4f} with standard error '
      f'+/- {np.std(scores):.4f}')

# seeing how training evolves as training data grows; should converge as n gets large
train_sizes, train_scores, test_scores = learning_curve(estimator = lr_pipeline,
                                                        X = X_train,
                                                        y = y_train,
                                                        cv = 5,
                                                        n_jobs = -1,
                                                        random_state = 1024)

# computing the mean CV accuracy scores across the different sample sizes
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(train_sizes, train_mean,
         color = 'blue', marker = 'o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color = 'green', linestyle = '--',
         marker = 's', markersize = 5,
         label = 'Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.title('Learning Curve for Logistic Regression')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.79, 0.85])
plt.tight_layout()
plt.show()


"""

Building Lasso Logistic Regression Model and Making Plots

"""


# lasso logistic regression pipline
lr_lasso_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l1', 
                                                                       solver = 'liblinear',
                                                                       random_state = 1024))

# constructing the penalization parameter to search over
lr_param_grid = {'logisticregression__C': np.random.uniform(0.001, 100, size = 20)}

# random search for lasso C hyperparamter
lr_lasso_rs = RandomizedSearchCV(
      estimator = lr_lasso_pipeline,
      param_distributions = lr_param_grid,
      n_iter = 20,
      cv = 5,
      n_jobs = -1,
      verbose = 1,
      random_state = 1024
)

# getting the results from the random search procedure
lr_lasso_rs.fit(X_train, y_train)
print(f'Best hyperparameters found: {lr_lasso_rs.best_params_}')
print(f"Best accuracy: {lr_lasso_rs.best_score_:.4f}")

# lasso logistic regression pipline with the choosen C from the random search procedure
lr_lasso_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l1', 
                                                                       solver = 'liblinear', 
                                                                       C = lr_lasso_rs.best_params_['logisticregression__C'],
                                                                       random_state = 1024))

# getting cross validation scores for logistic regression with choosen C value from random search procedure
scores = cross_val_score(estimator = lr_lasso_pipeline,
                         X = X_train,
                         y = y_train,
                         cv = 5,
                         n_jobs = 1)
print(f'CV accuracy scores: {scores}')
print(f'CV accuracy has mean {np.mean(scores):.4f} with standard error '
      f'+/- {np.std(scores):.4f}')

# seeing how training evolves as training data grows; should converge as n gets large
train_sizes, train_scores, test_scores = learning_curve(estimator = lr_lasso_pipeline,
                                                        X = X_train,
                                                        y = y_train,
                                                        cv = 5,
                                                        n_jobs = -1,
                                                        random_state = 1024)

# results from learning curve
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

# plotting learning curve
plt.plot(train_sizes, train_mean,
         color = 'blue', marker = 'o',
         markersize = 5, label = 'Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color = 'green', linestyle = '--',
         marker = 's', markersize = 5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15, color='green')

plt.grid()
plt.title('Learning For Lasso Logistic Regression')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.79, 0.85])
plt.tight_layout()
plt.show()


"""

Building Ridge Logistic Regression Model and Making Plots

"""


# ridge logistic regression pipline
lr_ridge_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l2', 
                                                                       solver = 'liblinear',
                                                                       random_state = 1024))

# constructing the penalization parameter to search over
lr_param_grid = {'logisticregression__C': np.random.uniform(0.001, 100, size = 20)}

# random search for ridge C hyperparamter
lr_ridge_rs = RandomizedSearchCV(
      estimator = lr_ridge_pipeline,
      param_distributions = lr_param_grid,
      n_iter = 20,
      cv = 5,
      n_jobs = -1,
      verbose = 1,
      random_state = 1024
)

# getting the results from the random search procedure
lr_ridge_rs.fit(X_train, y_train)
print(f'Best hyperparameters found: {lr_ridge_rs.best_params_}')
print(f"Best accuracy: {lr_ridge_rs.best_score_:.4f}")

# ridge logistic regression pipline with the choosen C from the random search procedure
lr_ridge_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l2', 
                                                                       solver = 'liblinear', 
                                                                       C = lr_ridge_rs.best_params_['logisticregression__C'],
                                                                       random_state = 1024))

# getting cross validation scores for logistic regression with choosen C value from random search procedure
scores = cross_val_score(estimator = lr_ridge_pipeline,
                         X = X_train,
                         y = y_train,
                         cv = 5,
                         n_jobs = -1)
print(f'CV accuracy scores: {scores}')
print(f'CV accuracy has mean {np.mean(scores):.4f} with standard error '
      f'+/- {np.std(scores):.4f}')

# seeing how training evolves as training data grows; should converge as n gets large
train_sizes, train_scores, test_scores = learning_curve(estimator = lr_ridge_pipeline,
                                                        X = X_train,
                                                        y = y_train,
                                                        cv = 5,
                                                        n_jobs = 1,
                                                        random_state = 1024)

# results from learning curve
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# plotting learning curve
plt.plot(train_sizes, train_mean,
         color = 'blue', marker = 'o',
         markersize = 5, label = 'Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color = 'green', linestyle = '--',
         marker = 's', markersize = 5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15, color='green')

plt.grid()
plt.title('Learning Curve for Ridge Logistic Regression')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.79, 0.85])
plt.tight_layout()
plt.show()


"""

Plot of Ridge and Lasso Training and CV Accuracy for Differing Penalization Terms

"""


# lasso logistic regression pipline
lr_lasso_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l1', 
                                                                       solver='liblinear',
                                                                       random_state = 1024))

# ridge logistic regression pipline
lr_ridge_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l2', 
                                                                       solver='liblinear',
                                                                       random_state = 1024))

# range of penalization parameter 
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# constructing lasso validation curve for the penalization parameter
lr_lasso_train_scores, lr_lasso_test_scores = validation_curve(
                estimator = lr_lasso_pipeline, 
                X = X_train, 
                y = y_train, 
                param_name = 'logisticregression__C', 
                param_range = param_range,
                cv = 5,
                n_jobs = -1)

# constructing ridge validation curve for the penalization parameter
lr_ridge_train_scores, lr_ridge_test_scores = validation_curve(
                estimator = lr_ridge_pipeline, 
                X = X_train, 
                y = y_train, 
                param_name = 'logisticregression__C', 
                param_range = param_range,
                cv = 5,
                n_jobs = -1)

# results from lasso CV
lr_lasso_train_mean = np.mean(lr_lasso_train_scores, axis = 1)
lr_lasso_train_std = np.std(lr_lasso_train_scores, axis = 1)
lr_lasso_test_mean = np.mean(lr_lasso_test_scores, axis = 1)
lr_lasso_test_std = np.std(lr_lasso_test_scores, axis = 1)

# results from ridge CV
lr_ridge_train_mean = np.mean(lr_ridge_train_scores, axis = 1)
lr_ridge_train_std = np.std(lr_ridge_train_scores, axis = 1)
lr_ridge_test_mean = np.mean(lr_ridge_test_scores, axis = 1)
lr_ridge_test_std = np.std(lr_ridge_test_scores, axis = 1)

# specificying figure size
plt.figure(figsize = (10, 10))

# lasso validation curve
plt.subplot(2, 1, 1)
plt.plot(param_range, lr_lasso_train_mean, 
         color = 'blue', marker = 'o', 
         markersize = 5, label = 'Lasso Training accuracy')
plt.fill_between(param_range, 
                 lr_lasso_train_mean + lr_lasso_train_std,
                 lr_lasso_train_mean - lr_lasso_train_std, 
                 alpha = 0.15,color = 'blue')
plt.plot(param_range, lr_lasso_test_mean, 
         color = 'orange', linestyle = '--', 
         marker = 's', markersize = 5, 
         label = 'Lasso Validation accuracy')
plt.fill_between(param_range, 
                 lr_lasso_test_mean + lr_lasso_test_std,
                 lr_lasso_test_mean - lr_lasso_test_std, 
                 alpha = 0.15, color = 'orange')
plt.grid()
plt.xscale('log')
plt.title('Lasso Validation Curve')
plt.xlabel(r'Penalization Parameter (Inverse of $\lambda$)')
plt.ylabel('Accuracy')
plt.ylim([0.79, 0.83])

# ridge validation curve
plt.subplot(2, 1, 2)
plt.plot(param_range, lr_ridge_train_mean, 
         color = 'green', marker = 'o', 
         markersize = 5, label = 'Ridge Training accuracy')
plt.fill_between(param_range, 
                 lr_ridge_train_mean + lr_ridge_train_std,
                 lr_ridge_train_mean - lr_ridge_train_std, 
                 alpha = 0.15, color = 'green')
plt.plot(param_range, lr_ridge_test_mean, 
         color = 'purple', linestyle = '--', 
         marker = 's', markersize = 5, 
         label = 'Ridge Validation accuracy')
plt.fill_between(param_range, 
                 lr_ridge_test_mean + lr_ridge_test_std,
                 lr_ridge_test_mean - lr_ridge_test_std, 
                 alpha = 0.15, color = 'purple')
plt.grid()
plt.xscale('log')
plt.title('Ridge Validation Curve')
plt.xlabel(r'Penalization Parameter (Inverse of $\lambda$)')
plt.ylabel('Accuracy')
plt.ylim([0.81, 0.83])

# add a legend
plt.subplot(2, 1, 1)
plt.legend(loc = 'lower right')
plt.subplot(2, 1, 2)
plt.legend(loc = 'lower right')

# show the subplots
plt.tight_layout()
plt.grid(True)
plt.show()


"""

Building Elastic Net Logistic Regression Model and Making Plots

"""


# elastic logistic regression pipline
lr_enet_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'elasticnet', 
                                                                      solver = 'saga',
                                                                      random_state = 1024))

# constructing the penalization parameter to search over
lr_param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
                 'logisticregression__l1_ratio': [0.25, 0.5, 0.75]}

# random search for elastic net C and l1 ratio hyperparamters
lr_enet_rs = RandomizedSearchCV(
      estimator = lr_enet_pipeline,
      param_distributions = lr_param_grid,
      n_iter = 20,
      cv = 5,
      n_jobs = -1,
      verbose = 1,
      random_state = 1024
)

# getting the results from the random search procedure
lr_enet_rs.fit(X_train, y_train)
print(f'Best hyperparameters found: {lr_enet_rs.best_params_}')
print(f"Best accuracy: {lr_enet_rs.best_score_:.4f}")

# ridge logistic regression pipline with the choosen C from the random search procedure
lr_enet_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'elasticnet', 
                                                                      solver = 'saga', 
                                                                      C = lr_enet_rs.best_params_['logisticregression__C'],
                                                                      l1_ratio = lr_enet_rs.best_params_['logisticregression__l1_ratio'],
                                                                      random_state = 1024))

# getting cross validation scores for logistic regression with choosen C value from random search procedure
scores = cross_val_score(estimator = lr_enet_pipeline,
                         X = X_train,
                         y = y_train,
                         cv = 5,
                         n_jobs = -1)
print(f'CV accuracy scores: {scores}')
print(f'CV accuracy has mean {np.mean(scores):.4f} with standard error '
      f'+/- {np.std(scores):.4f}')

# seeing how training evolves as training data grows; should converge as n gets large
train_sizes, train_scores, test_scores = learning_curve(estimator = lr_enet_pipeline,
                                                        X = X_train,
                                                        y = y_train,
                                                        cv = 5,
                                                        n_jobs = 1,
                                                        random_state = 1024)

# results from learning curve
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

# ploting the learning curve
plt.plot(train_sizes, train_mean,
         color = 'blue', marker = 'o',
         markersize = 5, label = 'Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color = 'green', linestyle = '--',
         marker = 's', markersize = 5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15, color='green')

plt.grid()
plt.title('Learning Curve for Elastic Net Logistic Regression')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.79, 0.85])
plt.tight_layout()
plt.show()


"""

Building Random Forest and Making Plots

"""


# randomf forest pipeline pipline
rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state = 1024))

# constructing the parameter grid to search over
rf_param_grid = {
    # number of trees in the forest
    'randomforestclassifier__n_estimators': np.random.randint(250, 500, size = 20),
    
    # criterion for splitting
    'randomforestclassifier__criterion': ['gini', 'entropy'],
    
    # maximum depth of the tree
    'randomforestclassifier__max_depth': np.random.randint(25, 50, size = 20),
    
    # minimum number of samples required to split an internal node
    'randomforestclassifier__min_samples_split': np.random.randint(15, 30, size = 20),
    
    # minimum number of samples required to be at a leaf node
    'randomforestclassifier__min_samples_leaf': np.random.randint(1, 5, size = 20)
}

# random search for svc hyperparamters
rf_rs = RandomizedSearchCV(
    estimator = rf_pipeline,
    param_distributions = rf_param_grid,
    n_iter = 20,
    cv = 5,
    n_jobs = -1,
    verbose = 1,
    random_state = 1024
)

# getting the results from the random search procedure
rf_rs.fit(X_train, y_train)
print(f'Best hyperparameters found: {rf_rs.best_params_}')
print(f"Best accuracy: {rf_rs.best_score_:.4f}")

# random forest with choosen parameters from random search procedure
rf_pipeline = make_pipeline(StandardScaler(), 
                            RandomForestClassifier(n_estimators = rf_rs.best_params_[
                                                         'randomforestclassifier__n_estimators'],
                                                   criterion = rf_rs.best_params_[
                                                         'randomforestclassifier__criterion'],
                                                   max_depth = rf_rs.best_params_[
                                                         'randomforestclassifier__max_depth'],
                                                   min_samples_split = rf_rs.best_params_[
                                                         'randomforestclassifier__min_samples_split'],
                                                   min_samples_leaf = rf_rs.best_params_[
                                                         'randomforestclassifier__min_samples_leaf'],
                                                                        random_state = 1024))

# getting cross validation scores for random forest with parameters from random search procedure
scores = cross_val_score(estimator = rf_pipeline,
                         X=X_train,
                         y=y_train,
                         cv=5,
                         n_jobs=-1)
print(f'CV accuracy scores: {scores}')
print(f'CV accuracy has mean {np.mean(scores):.4f} with standard error '
      f'+/- {np.std(scores):.4f}')

# seeing how training evolves as training data grows; should converge as n gets large
train_sizes, train_scores, test_scores = learning_curve(estimator = rf_pipeline,
                                                        X = X_train,
                                                        y = y_train,
                                                        cv = 5,
                                                        n_jobs = 1,
                                                        random_state = 1024)

# results from learning curve
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

# plotting learning curve
plt.plot(train_sizes, train_mean,
         color = 'blue', marker = 'o',
         markersize = 5, label = 'Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha = 0.15, color = 'blue')

plt.plot(train_sizes, test_mean,
         color = 'green', linestyle = '--',
         marker = 's', markersize = 5,
         label = 'Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15, color = 'green')

plt.grid()
plt.title('Learning Curve for Random Forest')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.79, 0.85])
plt.tight_layout()
plt.show()


"""

Evaluating Model Using Random Forest on Test Data

"""


# fitting the model to obtain parameters
rf_pipeline.fit(X_train, y_train)

# create a heatmap of the confusion matrix
plt.figure(figsize = (8, 6))
sns.heatmap(confusion_matrix(y_test, rf_pipeline.predict(X_test)), annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# getting accuracy
accuracy_score(y_test, rf_pipeline.predict(X_test))










