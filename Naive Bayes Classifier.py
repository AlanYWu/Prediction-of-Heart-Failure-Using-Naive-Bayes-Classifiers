#!/usr/bin/env python
# coding: utf-8
# Record running time
import time
start = time.time()

import pandas as pd

# Data Processing and loading
data_ca = pd.read_csv("./Data/data_change_heart_disease_categorical.csv")
data_nu = pd.read_csv("./Data/data_change_heart_disease_numerical.csv")

# Test train split and store into dictionaries
from sklearn.model_selection import train_test_split
def split(df,test_size=0.2):
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=43)
    data_dic = {}
    data_dic["X_train"]=X_train
    data_dic["X_test"]=X_test
    data_dic["y_train"]=y_train
    data_dic["y_test"]=y_test
    return data_dic
data_ca = split(data_ca)
data_nu = split(data_nu)

# Change datatype for categorical data
data_ca["X_train"]["FastingBS"] = data_ca["X_train"]["FastingBS"].apply(str)
data_ca["X_test"]["FastingBS"] = data_ca["X_test"]["FastingBS"].apply(str)

#Calculation of numerical data parameters
prediction_nu = {}
def feature_extract_nu(X_train,y_train):
    features = X_train.columns
    HD_idx = y_train=="Heart disease"
    Normal_idx = y_train=="Normal"
    prediction={}
    for i in features:
        feature_HD = X_train.loc[HD_idx,i]
        feature_Normal = X_train.loc[Normal_idx,i]

        prediction[i]=pd.DataFrame()
        prediction[i]["Heart disease"]=[feature_HD.mean(),feature_HD.std()]
        prediction[i]["Normal"]=[feature_Normal.mean(),feature_Normal.std()]
        prediction[i].index=["mean","sd"]
    return prediction
prediction_nu = feature_extract_nu(data_nu["X_train"],data_nu["y_train"])

# Calculation of categorical data parameters
prediction_ca = {}
def feature_extract_ca(X_train,y_train):
    prediction={}
    features = X_train.columns
    HD_idx = y_train=="Heart disease"
    Normal_idx = y_train=="Normal"
    for i in features:
        feature_list = X_train[i].unique()

        # Heart disease
        data=X_train.loc[HD_idx,i]
        total_count = data.count()
        category_prob = []
        for k in feature_list:
            a = data==k
            category_count = a.sum()
            category_prob.append(category_count/total_count)
        prediction[i]=pd.DataFrame()
        prediction[i]["Heart disease"]=category_prob
        prediction[i].index = feature_list

        # Normal
        data=X_train.loc[Normal_idx,i]
        total_count = data.count()
        category_prob = []
        for k in feature_list:
            a = data==k
            category_count = a.sum()
            category_prob.append(category_count/total_count)
        prediction[i]["Normal"]=category_prob
        prediction[i].index = feature_list

    return prediction
prediction_ca = feature_extract_ca(data_ca["X_train"],data_ca["y_train"])          

#Combine prediction parameters
prediction = {}
prediction.update(prediction_ca)
prediction.update(prediction_nu)

# Predict numerical data
from scipy.stats import norm
feature_list = data_nu["X_test"].columns
data_row = data_nu["X_test"].iloc[1,:]

def find_probability_nu(data_row):
    # Heart disease
    status = "Heart disease"
    prob_heart = 1
    for i in feature_list:
        data = data_row[i]
        mean = prediction[i]["Heart disease"]["mean"]
        sd = prediction[i]["Heart disease"]["sd"]
        prob_heart *= norm.cdf(data,loc=mean, scale=sd)
    # Normal
    status = "Normal"
    prob_norm = 1
    for i in feature_list:
        data = data_row[i]
        mean = prediction[i]["Normal"]["mean"]
        sd = prediction[i]["Normal"]["sd"]
        prob_norm *= norm.cdf(data,loc=mean, scale=sd)

    prob=pd.DataFrame()
    prob["Heart disease"]=[prob_heart]
    prob["Normal"]=[prob_norm]
    prob.index=[data_row.name] #Add the original data index
    return prob
find_probability_nu(data_row)

probability_nu = pd.DataFrame()
for i in range(len(data_nu["X_test"])):
    data_row = data_nu["X_test"].iloc[i,:]
    probability_nu = pd.concat([probability_nu,find_probability_nu(data_row)],axis=0)

# Predict numerical data
data_row=data_ca["X_test"].iloc[0,:]
feature_list = data_ca["X_test"].columns

def find_probability_ca(data_row):
    # Heart disease
    prob_heart = 1
    for i in range(len(feature_list)):
        prob_heart *= prediction[feature_list[i]]["Heart disease"][data_row[i]]
    # Normal
    prob_norm = 1
    for i in range(len(feature_list)):
        prob_norm *= prediction[feature_list[i]]["Normal"][data_row[i]]
        
    prob=pd.DataFrame()
    prob["Heart disease"]=[prob_heart]
    prob["Normal"]=[prob_norm]
    prob.index=[data_row.name] #Add the original data index
    return prob

find_probability_ca(data_row)

probability_ca = pd.DataFrame()
for i in range(len(data_ca["X_test"])):
    data_row = data_ca["X_test"].iloc[i,:]
    probability_ca = pd.concat([probability_ca,find_probability_ca(data_row)],axis=0)

# Combine the probability
probability = pd.DataFrame()
for i in ["Heart disease","Normal"]:
    probability[i] = probability_ca.loc[:,i]*probability_nu.loc[:,i]

# Return results
prediction = pd.DataFrame()
prediction["Result"]= probability.idxmax(axis=1)
prediction = prediction.iloc[:,0]

# Compare the results and return probability
result = prediction == data_ca["y_test"]
accuracy = result.sum()/data_ca["y_test"].__len__()
print("The accuracy for Naive Bayes Classifier is", round(accuracy, ndigits=4))

#Print running time
end = time.time()
print("The running time is:",round(end-start,ndigits=3),"s")

# http://www.codeinword.com/ Code generation
