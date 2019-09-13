# --------------
# import packages

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# code starts here

df = pd.read_csv(path)
X = df.drop('label', 1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)





# code ends here


# --------------
# code starts here
bandwidth = X_train['spectral_bandwidth']
sns.distplot(bandwidth)

zc_rate = X_train['zero_crossing_rate']
sns.distplot(zc_rate)

centroid = X_train['spectral_centroid']
sns.distplot(centroid)
# code ends here


# --------------
# import the packages
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# code starts here
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(accuracy_lr)

# code ends here


# --------------
#import the packages 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# code starts here
sc = SVC(random_state=9)
sc.fit(X_train, y_train)
y_pred_sc = sc.predict(X_test)
accuracy_sc = accuracy_score(y_test, y_pred_sc)
print(accuracy_sc)
rf = RandomForestClassifier(random_state=9)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(accuracy_rf)

# code ends here


# --------------
# import packages
from sklearn.ensemble import BaggingClassifier


# code starts here
bagging_clf = BaggingClassifier(base_estimator=lr, n_estimators=50, max_samples=100, bootstrap=True, random_state=9)
bagging_clf.fit(X_train, y_train)
y_pred_bag = bagging_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_bag)
print(accuracy)

# code ends here


# --------------
# import packages
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

# code starts here
nv = GaussianNB()
nv.fit(X_train, y_train)
voting_clf_soft = VotingClassifier(estimators =[('Logistic Regression', lr), ('Random Forest', rf), ('Naive Bayes', nv)], voting='soft')
voting_clf_soft.fit(X_train, y_train)
y_pred_soft = voting_clf_soft.predict(X_test)
accuracy_soft = accuracy_score(y_test, y_pred_soft)
print(accuracy_soft)
# code ends here


