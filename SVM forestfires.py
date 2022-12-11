# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 19:42:44 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("forestfires.csv")
df.shape
df.dtypes
df.head()

df.isnull().sum()

#========================================================
# finding duplicate rows and columns
df.duplicated()
df.duplicated().sum()
df = df.drop_duplicates()
df.shape

df.columns.duplicated()
df.columns.duplicated().sum()

#===========================================================
# Data visualization
# Boxplots

import matplotlib.pyplot as plt

def plot_boxplot(df,ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()
    
plot_boxplot(df,"FFMC")
plot_boxplot(df,"DMC")
plot_boxplot(df,"DC")
plot_boxplot(df,"ISI")
plot_boxplot(df,"temp")
plot_boxplot(df,"RH")
plot_boxplot(df,"wind")
plot_boxplot(df,"rain")
plot_boxplot(df,"area")

def outliers(df,ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    ls = df.index[(df[ft]< lower_bound) | (df[ft]> upper_bound)]
    return ls
index_list =[]
for feature in ["FFMC","DMC","DC","ISI","temp","RH","wind","rain","area"]:
    index_list.extend(outliers(df,feature))

index_list

def remove(df,ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df

df_cleaned = remove(df,index_list)
df_cleaned .shape

#===================================================
# lableEncoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_cleaned["month"] = LE.fit_transform(df_cleaned["month"])
df_cleaned["day"] = LE.fit_transform(df_cleaned["day"])
df_cleaned["size_category"] = LE.fit_transform(df_cleaned["size_category"])

df_cleaned.dtypes
df_cleaned.shape

# spliting the varaibles as X and Y

X = df_cleaned.iloc[:,0:30]
Y = df_cleaned['size_category']

#===================================================
# scatter plot
# histogram
import seaborn as sns
sns.pairplot(df_cleaned)

#=============================================================
# spliting test and train
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=33)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

#======================================================
# model fitting
# Linear function
from sklearn.svm import SVC
clf = SVC(kernel="linear",C=1.0)
clf.fit(X_train,Y_train)
Y_pred_train = clf.predict(X_train)
Y_pred_test = clf.predict(X_test)

# metrics
from sklearn.metrics import accuracy_score, confusion_matrix

cm = confusion_matrix(Y_train, Y_pred_train)
cm1 = confusion_matrix(Y_test, Y_pred_test)
cm
cm1

print('Training Accuracy :',accuracy_score(Y_train, Y_pred_train).round(3))
print('Testing Accuracy :',accuracy_score(Y_test, Y_pred_test).round(3))
#==============================================================
# Polynamial function
from sklearn.svm import SVC
clf = SVC(kernel="poly",degree=6)
clf.fit(X_train,Y_train)
Y_pred_train = clf.predict(X_train)
Y_pred_test = clf.predict(X_test)

# metrics
from sklearn.metrics import accuracy_score, confusion_matrix

cm = confusion_matrix(Y_train, Y_pred_train)
cm1 = confusion_matrix(Y_test, Y_pred_test)
cm
cm1

print('Training Accuracy :',accuracy_score(Y_train, Y_pred_train).round(3))
print('Testing Accuracy :',accuracy_score(Y_test, Y_pred_test).round(3))

#==============================================================
# Radial basis function
from sklearn.svm import SVC
clf = SVC(kernel="rbf",gamma=20)
clf.fit(X_train,Y_train)    
Y_pred_train = clf.predict(X_train)
Y_pred_test = clf.predict(X_test)

# metrics
from sklearn.metrics import accuracy_score, confusion_matrix

cm = confusion_matrix(Y_train, Y_pred_train)
cm1 = confusion_matrix(Y_test, Y_pred_test)
cm
cm1

print('Training Accuracy :',accuracy_score(Y_train, Y_pred_train).round(3))
print('Testing Accuracy :',accuracy_score(Y_test, Y_pred_test).round(3))

#=========================================================
from sklearn.metrics import log_loss
log_loss(Y_train, Y_pred_train) 
#=========================================================

# comparing the all functions polynamial function is best















