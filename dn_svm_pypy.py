# Importing the libraries
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.cross_validation import train_test_split


#=============================READING IN THE DATASET=================================================#
dataset = pd.read_csv('DM data set.csv')

#=============================SPLITTING THE DATASET=================================================#
y = dataset.iloc[:, [-1]].values
X = dataset.iloc[:, 0:19].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


#=============================FITTING SVM ESTIMATOR TO DATA=========================================#
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 42, probability=True)
classifier.fit(X_train, y_train)

#================================CROSS VALIDATION===================================================#
from sklearn.model_selection import GridSearchCV

Cs = [200, 280,285, 286, 287, 288, 289]
gammas = [0.000001, 0.00001]
param_grid = {'C': Cs, 'gamma' : gammas}

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(classifier, param_grid=param_grid, cv=4)

#================================TIMING THE TESTING AND TRAINING====================================#
startTime = time.time()
gm_cv.fit(X_train, y_train)
print ("Training took ", time.time()-startTime, " seconds")
startTime = time.time()
# Fit to the training set
y_pred = gm_cv.predict_proba(X_test)
#stop the timer
print("Testing took ", time.time()-startTime, " seconds")

#================================BEST PARAMS=========================================================#
r2 = gm_cv.score(X_test, y_test)
print("Tuned SVC C: {}".format(gm_cv.best_params_))
print("Tuned SVC Best Score: {}".format(gm_cv.best_score_))
print("Tuned SVC R squared: {}".format(r2))

#================================ROC CURVE==========================================================#
y_pred = gm_cv.predict_proba(X_test)
fpr, tpr , asdf = roc_curve(y_test, y_pred[:,1], pos_label = 1)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange',
label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#================================CONFUSION MATRIX====================================================#
y_pred_2 = gm_cv.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_2).ravel()

print(" ")
print("Confusion Matrix: \r")
print("--------------------------------------- ")
print("True Positive: ".ljust(12," ")+ str(tp).ljust(6," ") + "False Positive:",str(fp)) 
print("True Negative: ".ljust(12," ")+ str(tn).ljust(6," ") + "False Negative:",str(fn)) 
print("---------------------------------------")