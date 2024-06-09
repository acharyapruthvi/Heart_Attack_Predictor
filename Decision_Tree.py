import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import joblib  # for saving the model

# Importing the dataset
df_heart = pd.read_csv('DATA/heart.csv')
target = 'output'
# Creating one hot encoded features for working with non-tree based algorithms
string_col = df_heart.select_dtypes(include="object").columns
df_heart[string_col] = df_heart[string_col].astype("string")

# Creating one hot encoded features for working with non-tree based algorithms
df_tree = df_heart.apply(LabelEncoder().fit_transform)
df_tree.head()

# Creating the columns of features
feature_col_tree = df_tree.columns.to_list()
feature_col_tree.remove(target)

y = df_tree[target].values

# Parameters to test
n_splits_values = range(2, 10) # Number of splits to test
criterion_values = ['gini', 'entropy'] # Two criterion values to test

# Variables to store the best model information
best_score = 0
best_model = None
best_params = {}

# Looping through the parameters to test
for n_splits in n_splits_values:
    # Splitting the data into training and validation sets using Stratified K-Fold cross-validation
    kf = model_selection.StratifiedKFold(n_splits=n_splits)
    for c in criterion_values: # Looping through the criterion values
        roc_scores = [] # Saving the ROC-AUC scores for each fold
        for fold, (trn_, val_) in enumerate(kf.split(X=df_tree, y=y)): # Looping through the folds
            X_train = df_tree.loc[trn_, feature_col_tree] # Extracting the training data for the fold
            y_train = df_tree.loc[trn_, target] # Extracting the target variable for the training data
            
            X_valid = df_tree.loc[val_, feature_col_tree] # Extracting the validation data for the
            y_valid = df_tree.loc[val_, target] # Extracting the target variable for the
            
            clf = DecisionTreeClassifier(criterion=c) # Initializing the Decision Tree Classifier with the criterion value
            clf.fit(X_train, y_train) # Fitting the model on the training data
            y_pred = clf.predict(X_valid) # Predicting the target variable for the validation data
            # Printing the classification report
            print(f"The fold is: {fold} with n_splits={n_splits} and criterion={c}")
            print(classification_report(y_valid, y_pred))
            acc = roc_auc_score(y_valid, y_pred)
            roc_scores.append(acc)
            print(f"The ROC-AUC score for fold {fold+1}: {acc}")
        
        mean_roc_score = np.mean(roc_scores) # Calculating the mean ROC-AUC score for the folds
        print(f"Mean ROC-AUC score for n_splits={n_splits} and criterion={c}: {mean_roc_score}") # Calculating the mean ROC-AUC
        
        # Check if this is the best score
        if mean_roc_score > best_score:
            best_score = mean_roc_score
            best_model = clf
            best_params = {'n_splits': n_splits, 'criterion': c}

        # Plotting the ROC scores for each criterion
        plt.plot(roc_scores, label=f'n_splits={n_splits}, criterion={c}')

plt.legend()
plt.xlabel("Folds")
plt.ylabel("ROC-AUC Score")
plt.title("ROC-AUC Scores for Different n_splits and Criteria")
plt.show()

# Save the best model
if best_model:
    joblib.dump(best_model, 'best_decision_tree_model.joblib')
    print(f"Best model saved with params: {best_params} and ROC-AUC score: {best_score}")
else:
    print("No best model found")
