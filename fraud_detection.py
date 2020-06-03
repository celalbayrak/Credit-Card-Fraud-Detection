# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:35:59 2020

@author: Celal
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.svm import SVC
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
#%%

df=pd.read_csv("C:/Users/Celal/Desktop/machine learning/proje/creditcard.csv")
#%%


x=df.drop("Class",axis=1)
y=df["Class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#%%



x_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
df_new=pd.concat([x_resampled,y_resampled],axis=1)
df_new =df_new.sample(frac=1)
x_resampled=df_new.drop("Class",axis=1)
y_resampled=df_new["Class"]

#%%

reg=LogisticRegression()
reg.fit(x_train,y_train)
a=confusion_matrix(y_test,reg.predict(x_test))


reg_weight=LogisticRegression(class_weight={0:0.2,1:0.8})
reg_weight.fit(x_train,y_train)
b=confusion_matrix(y_test,reg_weight.predict(x_test))

reg_resampled=LogisticRegression()
reg_resampled.fit(x_resampled,y_resampled)
c=confusion_matrix(y_test,reg_resampled.predict(x_test))

reg_weight_res=LogisticRegression(class_weight={0:0.2,1:0.8})
reg_weight_res.fit(x_resampled,y_resampled)
d=confusion_matrix(y_test,reg_weight_res.predict(x_test))
#%%

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
e=confusion_matrix(y_test,dt.predict(x_test))

#%%
dt_res=DecisionTreeClassifier()
dt_res.fit(x_resampled,y_resampled)
f=confusion_matrix(y_test,dt_res.predict(x_test))
#%%
dt_weight=DecisionTreeClassifier(class_weight={0:0.5008485,1:295.13601036})
dt_weight.fit(x_train,y_train)
g=confusion_matrix(y_test,dt_weight.predict(x_test))
#%%
dt_weight_res=DecisionTreeClassifier(class_weight={0:0.1,1:0.9})
dt_weight_res.fit(x_resampled,y_resampled)
h=confusion_matrix(y_test,dt_weight_res.predict(x_test))
#%%

#%%
def run_gridsearch(X, y, clf, param_grid, cv=5):
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv, scoring="roc_auc", verbose=4,n_jobs=-1)
    grid_search.fit(X, y)
    print("roc auc: ", grid_search.best_score_)
    print("Best Params: ", grid_search.best_params_)
    return 1
#%%
dt_test=DecisionTreeClassifier()
run_gridsearch(x_resampled,y_resampled,dt_test,param_grid,cv=3)
#%%

sorted(sklearn.metrics.SCORERS.keys())
#%% 0.6835
small_df=df.sample(n=100000)

x2=small_df.drop("Class",axis=1)
y2=small_df["Class"]
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2)
x_resampled2, y_resampled2 = SMOTE().fit_resample(x_train2, y_train2)

clf = SVC(gamma='auto')
clf.fit(x_resampled2,y_resampled2)
#%%

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
#%% knn
knn_param= {"n_neighbors":range(1,10,1)}
knn = KNeighborsClassifier()
run_gridsearch(x_resampled,y_resampled,knn,knn_param,cv=3)
#0.706 score
#%%knn best
knn_best=KNeighborsClassifier(n_neighbors=4)
knn_best.fit(x_resampled,y_resampled)
knn_pred=knn_best.predict(x_test)
#%% dtree
param_dtree = {"class_weight":[{0:0.1,1:0.9},{0:0.2,1:0.8},{0:0.3,1:0.7},{0:0.4,1:0.6},{0:0.5,1:0.5}]}
dtree=DecisionTreeClassifier()
run_gridsearch(x_resampled,y_resampled,dtree,param_dtree,cv=3)
#0.888 score
#%%dtree best
dtree_best=DecisionTreeClassifier(class_weight={0:0.2,1:0.8})
dtree_best.fit(x_resampled,y_resampled)
dtree_pred=dtree_best.predict(x_test)
#%%random forest
param_rnd={"class_weight":[{0:0.1,1:0.9},{0:0.2,1:0.8},{0:0.3,1:0.7},{0:0.4,1:0.6},{0:0.5,1:0.5}],"n_estimators":[2,4,8,16]}
#best rnd
#%%
rnd_tree=RandomForestClassifier(n_estimators=16,class_weight={0:0.2,1:0.8})
rnd_tree.fit(x_resampled,y_resampled)
#run_gridsearch(x_resampled,y_resampled,rnd_tree,param_rnd,cv=3)
rnd_pred=rnd_tree.predict(x_test)
#0.894 score
#%%
n_inputs = x_resampled.shape[1]

nn= Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
nn.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#%%
nn.fit(x_resampled, y_resampled, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)
#%%neural network best 0.91
nn_pred=nn.predict_classes(x_test,batch_size=200,verbose=2)
#%%
import matplotlib.pyplot as plt
df.Class.value_counts()[0]
plt.bar(('0','1'), [227448,227448], align='center', alpha=1.0)
plt.xticks(np.arange(2), ('0','1'))
plt.ylabel('Number of Instances')
plt.title('Fraud')

plt.show()
#%%
from sklearn.model_selection import validation_curve
param_range = range(1,20)
# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(nn,
                                             x_resampled,
                                             y_resampled,
                                             param_name="epoch",
                                             param_range=param_range,
                                             cv=2,
                                             scoring="roc_auc",
                                             n_jobs=-1)


# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Neural Network")
plt.xlabel("epoch")
plt.ylabel("Roc Auc")
plt.tight_layout()
plt.legend(loc="best")
plt.show()