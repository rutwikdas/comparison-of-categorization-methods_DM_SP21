#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed
from sklearn.metrics import plot_roc_curve

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

# Split dataset into train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

# Import required library for classifier
from sklearn.svm import SVC

# Create Support Vector Classifier
svc_g_model = SVC(kernel="rbf", max_iter=200)

# Fit the classifier to the data
svc_g_model.fit(X_train,y_train)


# Check accuracy of our model on the test data
svc_g_model.score(X_test, y_test)

# Cross validation time
from sklearn.model_selection import cross_val_score

#train model with cv of 10
import time
start = time.time()
cv_scores = cross_val_score(svc_g_model, X, y, cv=10)
elapsed = time.time()-start

#print each cv score (accuracy) and average them
print("cv scores for 10-fold cross validation:")
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))
print("---%s seconds elapsed---" % elapsed)
svc_g_disp = plot_roc_curve(svc_g_model, X_test, y_test, name= "ROC Curve of SVC on test data\n (Gaussian Kernel, max iter=200)")
plt.show()