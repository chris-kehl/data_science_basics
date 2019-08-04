#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:26:16 2019

@author: chriskehl
"""

# using Naive Bayes to test which wine has the most alcohol and sugar red or white

# import packages
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import statsmodels.tools.tools as stattools

# load the test and traiining sets
wine_tr = pd.read_csv("/Users/chriskehl/Library/CloudStorage/iCloud Drive/Documents/data_files/data_science/data_sets/website_data_sets/wine_flag_training.csv")

wine_test = pd.read_csv("/Users/chriskehl/Library/CloudStorage/iCloud Drive/Documents/data_files/data_science/data_sets/website_data_sets/wine_flag_test.csv")

# contingency tables for Alcohol
t1 = pd.crosstab(wine_tr['Type'], wine_tr['Alcohol_flag'])
t1['Total'] = t1.sum(axis=1)
t1.loc['Total'] = t1.sum()
t1

# contingency tables for Sugar
t2 = pd.crosstab(wine_tr['Type'], wine_tr['Sugar_flag'])
t2['Total'] = t1.sum(axis=1)
t2.loc['Total'] = t1.sum()
t2

# plot of alcohol content
t1_plot = pd.crosstab(wine_tr['Alcohol_flag'], wine_tr['Type'])
t1_plot.plot(kind='bar', stacked = True)

# plot of sugar content
t2_plot = pd.crosstab(wine_tr['Sugar_flag'], wine_tr['Type'])
t2_plot.plot(kind='bar', stacked = True)

# Run the Naive Bayes Algorithm

X_Alcohol_ind = np.array(wine_tr['Alcohol_flag'])
(X_Alcohol_ind, X_Alcohol_ind_dict) = stattools.categorical(X_Alcohol_ind, drop=True, dictnames = True)
X_Alcohol_ind = pd.DataFrame(X_Alcohol_ind)
X_Sugar_ind = np.array(wine_tr['Sugar_flag'])
(X_Sugar_ind, X_Sugar_ind_dict) = stattools.categorical(X_Sugar_ind, drop=True, dictnames = True)
X_Sugar_ind = pd.DataFrame(X_Sugar_ind)
X = pd.concat((X_Alcohol_ind, X_Sugar_ind), axis = 1)

# target variable
Y = wine_tr['Type']

# Naive Bayes Algorithm
nb_01 = MultinomialNB().fit(X, Y)


# generate the prediction
X_Alcohol_ind_test = np.array(wine_test['Alcohol_flag'])
(X_Alcohol_ind_test, X_Alcohol_ind_dict_test) = stattools.categorical(X_Alcohol_ind_test, drop=True, dictnames = True)
X_Alcohol_ind_test = pd.DataFrame(X_Alcohol_ind_test)
X_Sugar_ind_test = np.array(wine_test['Sugar_flag'])
(X_Sugar_ind_test, X_Sugar_ind_dict_test) = stattools.categorical(X_Sugar_ind_test, drop=True, dictnames = True)
X_Sugar_ind_test = pd.DataFrame(X_Sugar_ind_test)
X_test = pd.concat((X_Alcohol_ind_test, X_Sugar_ind_test), axis = 1)
 
Y_predicted = nb_01.predict(X_test)
 
ypred = pd.crosstab(wine_test['Type'], Y_predicted, rownames = ['Actual'], colnames = ['Predicted'])
ypred['Total'] = ypred.sum(axis=1); ypred.loc['Total'] = ypred.sum(); ypred
 
