#import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set random seed
from pip._internal.utils.misc import tabulate

np.random.seed(42)

# Load data as csv
df = pd.read_csv("bc-norm.csv")

# View top 5 rows
df.head()

# Check if there are any null values
df.isnull().values.any()

# Remove null values
df = df.dropna()

# Check if there are any null values
df.isnull().values.any()

# Initialize X and y and folds
X = df.drop(columns="Class")
y = df["Class"]
folds = 10

# Import required library for resampling
from imblearn.under_sampling import RandomUnderSampler

# Instantiate Random Under Sampler
rus = RandomUnderSampler(random_state=42)

# Perform random under sampling
df_data, df_target = rus.fit_resample(X, y)

# Import required libraries for performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score),
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score),
           'f1_score':make_scorer(f1_score)}

# Import required libraries for machine learning classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#for both Gaussian (RBF) and polynomial kernel
from sklearn.svm import SVC
#for both sigmoid and ReLu
from sklearn.neural_network import MLPClassifier

# Instantiate the machine learning classifiers
knn_model = KNeighborsClassifier(n_neighbors = 5)
svc_g_model = SVC(kernel="rbf", max_iter=200)
svc_p_model = SVC(kernel="poly", degree=1, max_iter=200)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
dnn_s_model = MLPClassifier(max_iter=2000, activation="logistic", solver="sgd", batch_size="auto", learning_rate="adaptive")
dnn_r_model = MLPClassifier(max_iter=1400, activation="relu", solver="sgd", batch_size="auto", learning_rate="adaptive")

# Define the models evaluation function
def models_evaluation(X, y, folds):
    # Perform cross-validation to each machine learning classifier
    knn = cross_validate(knn_model, X, y, cv=folds, scoring=scoring)
    svc_g = cross_validate(svc_g_model, X, y, cv=folds, scoring=scoring)
    svc_p = cross_validate(svc_p_model, X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    dnn_s = cross_validate(dnn_s_model, X, y, cv=folds, scoring=scoring)
    dnn_r = cross_validate(dnn_r_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'K-Nearest Neighbors': [knn['test_accuracy'].mean(),
                                                                knn['test_precision'].mean(),
                                                                knn['test_recall'].mean(),
                                                                knn['test_f1_score'].mean()],

                                        'Support Vector Classifier (Gaussian Kernel)': [svc_g['test_accuracy'].mean(),
                                                                      svc_g['test_precision'].mean(),
                                                                      svc_g['test_recall'].mean(),
                                                                      svc_g['test_f1_score'].mean()],

                                        'Support Vector Classifier (Polynomial Kernel)': [svc_p['test_accuracy'].mean(),
                                                                      svc_p['test_precision'].mean(),
                                                                      svc_p['test_recall'].mean(),
                                                                      svc_p['test_f1_score'].mean()],

                                        'Decision Tree': [dtr['test_accuracy'].mean(),
                                                          dtr['test_precision'].mean(),
                                                          dtr['test_recall'].mean(),
                                                          dtr['test_f1_score'].mean()],

                                        'Random Forest': [rfc['test_accuracy'].mean(),
                                                          rfc['test_precision'].mean(),
                                                          rfc['test_recall'].mean(),
                                                          rfc['test_f1_score'].mean()],

                                       'Deep Neural Network (Sigmoid)': [dnn_s['test_accuracy'].mean(),
                                                                         dnn_s['test_precision'].mean(),
                                                                         dnn_s['test_recall'].mean(),
                                                                         dnn_s['test_f1_score'].mean()],

                                       'Deep Neural Network (ReLu)': [dnn_r['test_accuracy'].mean(),
                                                                                       dnn_r['test_precision'].mean(),
                                                                                       dnn_r['test_recall'].mean(),
                                                                                       dnn_r['test_f1_score'].mean()]},
                                       index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    #Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)

    #Return models performance metrics scores data frame
    return models_scores_table



# Run models_evaluation function
from openpyxl import Workbook
models_evaluation(X, y, 10).to_excel(r'/Users/rutwik/PycharmProjects/comparison of categorization methods_DM_SP21/plots\model_evaluation.xlsx', index=True, header=True)

