#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:42:02 2025

@author: vaibhavkale
"""
import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,precision_recall_fscore_support

print("Program is Starting")
print()
start_time = time.time()

## load the dataset

df1 = pd.read_csv("data/case_study1.xlsx - case_study1.csv")
df2 = pd.read_csv("data/case_study2.xlsx - case_study2.csv")

## remove nulls

df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

df2 = df2.drop(columns_to_be_removed,axis=1)

for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]
    
## looking for the common columns in both the datasets for merging the datasets.
for i in list(df1.columns):
    if i in list(df2.columns):
        print(i)
        
## merge the datasets on common column

df = pd.merge(df1, df2,'inner',left_on=['PROSPECTID'],right_on=['PROSPECTID'])


## check how many columns are categorical

for i in df.columns:
    if df[i].dtype == 'object':
        print(i)
        
## Chi Square Test
for i in ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']:
    contingency_table = pd.crosstab(df[i], df['Approved_Flag'])
    chi2, pval, dof, expected = chi2_contingency(contingency_table)
    print(i, '----', pval)
    
## Since all the pvalues are less thab 0.05 we will consider all the categorical variables

## VIF Calculation.
numerical_columns = []

for i in df.columns:
    if df[i].dtype != 'object' and i not in ['Approved_Flag','PROSPECTID']:
        numerical_columns.append(i)


## Multicollinearity vs Correlation

### Multicolinearity -- Predictability of each feature by using other features.
## Correlation -- it is linear relationships between columns.
#$ in convex functions , correlation gives misleading values.

## VIF Sequentially check

vif_data = df[numerical_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0,total_columns):
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index,'-----',vif_value)
    
    if vif_value <= 6:
        columns_to_be_kept.append(numerical_columns[i])
        column_index = column_index + 1 
    else:
        vif_data = vif_data.drop([numerical_columns[i]],axis=1)
        
## check annova for columns_to_be_kept

from scipy.stats import f_oneway

columns_to_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df['Approved_Flag'])
    
    group_p1 = [value for value,group in zip(a,b) if group == 'P1']
    group_p2 = [value for value,group in zip(a,b) if group == 'P2']
    group_p3 = [value for value,group in zip(a,b) if group == 'P3']
    group_p4 = [value for value,group in zip(a,b) if group == 'P4']
    
    f_statistics, p_value = f_oneway(group_p1,group_p2,group_p3,group_p4)
    
    if p_value <= 0.05:
        columns_to_kept_numerical.append(i)
        
## feature Selection is done for categoricala and numerical columns

## listing all the final features
features = columns_to_kept_numerical + ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']
df = df[features + ['Approved_Flag']]





df['MARITALSTATUS'].unique()
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()

## by observing the type of outputs we can conclude we can only apply lableencoding to Education column

## Lable Encoding to EDUCATION.
## Ordinal Feature -- EDUCATION
## SSC             : 1
## 12th            : 2
## Graduate        : 3
## Undergraduate   : 3
## Post-Graduate   : 4
## Others          : 1
## Professional    : 3

## Others had to be verified by the business and user

df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']] = 1 
df.loc[df['EDUCATION'] == '12TH',['EDUCATION']] = 2 
df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']] = 3 
df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']] = 3 
df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']] = 4 
df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']] = 1 
df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']] = 3 

df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()

df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2'],dtype='uint8')

df_encoded.info()

## Machine Learning Model fitting

## data Processing

x = df_encoded.drop(['Approved_Flag'],axis=1)
y = df_encoded['Approved_Flag']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

## 1. RandomForest

rf_classifier = RandomForestClassifier(n_estimators=200,random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f"Accuracy Score : {accuracy}")
print()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['P1','P2','P3','P4']):
    print(f"Class {v}: ")
    print(f"Precision : {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"f1_Score : {f1_score[i]}")
    print()

# 2. xgboost

import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size=0.2,random_state=42)

xgb_classifier.fit(x_train,y_train)
y_pred = xgb_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print()
print(f"Accuracy Score : {accuracy}")
print()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['P1','P2','P3','P4']):
    print(f"Class {v}: ")
    print(f"Precision : {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"f1_Score : {f1_score[i]}")
    print()
    
# 3. Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

dt_model = DecisionTreeClassifier(max_depth=20,min_samples_split=10)
dt_model.fit(x_train,y_train)
y_pred = dt_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print()
print(f"Accuracy Score : {accuracy}")
print()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['P1','P2','P3','P4']):
    print(f"Class {v}: ")
    print(f"Precision : {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"f1_Score : {f1_score[i]}")
    print()
    

## as xgboost show the best results out of all the models choosen.
## so we will select xgboost and perform hyperparameter tuning on it to get the 
## better accuracy.

## hyperparameter tuning for xgboost
## define the hyperparameters

param_grid = {
    'colsample_bytree' : [0.1,0.3,0.5,0.7,0.9],
    'learning_rate' : [0.001,0.01,0.1,1],
    'max_depth' : [3,5,8,10],
    'alpha' : [1,10,100],
    'n_estimators' : [10,50,100]
    }

index = 0

answers_grid = {
    'combination' : [],
    'train_Accuracy' : [],
    'test_Accuracy' : [],
    'colsample_bytree' : [],
    'learning_rate' : [],
    'max_depth' : [],
    'alpha' : [],
    'n_estimators' : []
    }

## loop throguh each combination of hyperparameter

for colsample_bytree in param_grid['colsample_bytree']:
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            for alpha in param_grid['alpha']:
                for n_estimators in param_grid['n_estimators']:
                    
                    index = index + 1 
                    
                    ## define and train the xgboost model
                    model = xgb.XGBClassifier(objective='multi:softmax',
                                              num_class=4,
                                              colsample_bytree=colsample_bytree,
                                              learning_rate=learning_rate,
                                              max_depth=max_depth,
                                              alpha=alpha,
                                              n_estimators=n_estimators)
                    
                    
                    x = df_encoded.drop(['Approved_Flag'],axis=1)
                    y = df_encoded['Approved_Flag']
                    
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)

                    x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size=0.2,random_state=42)

                    model.fit(x_train,y_train)
                    
                    ## predict on training and testing data
                    y_train_pred = model.predict(x_train)
                    y_test_pred = model.predict(x_test)
                    
                    ## test and train accuracy
                    train_accuracy = accuracy_score(y_train,y_train_pred)
                    test_accuracy = accuracy_score(y_test,y_test_pred)
                    
                    # Include into the lists
                    answers_grid ['combination']   .append(index)
                    answers_grid ['train_Accuracy']    .append(train_accuracy)
                    answers_grid ['test_Accuracy']     .append(test_accuracy)
                    answers_grid ['colsample_bytree']   .append(colsample_bytree)
                    answers_grid ['learning_rate']      .append(learning_rate)
                    answers_grid ['max_depth']          .append(max_depth)
                    answers_grid ['alpha']              .append(alpha)
                    answers_grid ['n_estimators']       .append(n_estimators)
                    
                    
                    # Print results for this combination
                    print(f"Combination {index}")
                    print(f"colsample_bytree: {colsample_bytree}, learning_rate: {learning_rate}, max_depth: {max_depth}, alpha: {alpha}, n_estimators: {n_estimators}")
                    print(f"Train Accuracy: {train_accuracy:.2f}")
                    print(f"Test Accuracy : {test_accuracy :.2f}")
                    print("-" * 30)
                    

## predict for unseen data
a3 = pd.read_excel('data/Unseen_Dataset.xlsx')
cols_in_df = list(df.columns)
cols_in_df.pop(42)

df_unseen = a3[cols_in_df]

df_unseen['MARITALSTATUS'].unique()
df_unseen['EDUCATION'].unique()
df_unseen['GENDER'].unique()
df_unseen['last_prod_enq2'].unique()
df_unseen['first_prod_enq2'].unique()

## Lable Encoding to EDUCATION.
## Ordinal Feature -- EDUCATION
## SSC             : 1
## 12th            : 2
## Graduate        : 3
## Undergraduate   : 3
## Post-Graduate   : 4
## Others          : 1
## Professional    : 3

df_unseen.loc[df_unseen['EDUCATION'] == 'SSC',['EDUCATION']] = 1 
df_unseen.loc[df_unseen['EDUCATION'] == '12TH',['EDUCATION']] = 2 
df_unseen.loc[df_unseen['EDUCATION'] == 'GRADUATE',['EDUCATION']] = 3 
df_unseen.loc[df_unseen['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']] = 3 
df_unseen.loc[df_unseen['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']] = 4 
df_unseen.loc[df_unseen['EDUCATION'] == 'OTHERS',['EDUCATION']] = 1 
df_unseen.loc[df_unseen['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']] = 3 


df_unseen['EDUCATION'].value_counts()
df_unseen['EDUCATION'] = df_unseen['EDUCATION'].astype(int)
df_unseen.info()
                    

df_encoded_unseen = pd.get_dummies(df_unseen,columns=['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2'],dtype='uint8')
df_encoded_unseen.info()

model = xgb.XGBClassifier(objective='multi:softmax',
                          num_class=4,
                          colsample_bytree=0.9,
                          learning_rate=1,
                          max_depth=3,
                          alpha=10,
                          n_estimators=100)

model.fit(x_train,y_train)
y_pred_unseen = model.predict(df_encoded_unseen)
a3['Target_Variable'] = y_pred_unseen

a3['Target_Variable'].value_counts()

a3.to_excel('data/Final_Predictions.xlsx',index=False)


# print runtime
end_time = time.time()

elapsed_time = end_time - start_time
print("Total run time of the program- " + str(round(elapsed_time,2)) + 'sec')
