# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:51:19 2020

@author: SAMRAH SOHA
"""
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


bank=pd.read_csv("E:/Assisgnments/Logistic Regression/bank_data.csv")
bank=bank.drop('pdays',axis=1)
bank=bank.drop('previous',axis=1)
#exploratory data analysis
bank.describe()
bank.columns
#model building
#model 1
var=bank.iloc[:,:29]
Model1=sm.logit('y~var',data= bank).fit()
Model1.summary()
Model1.summary2()      #AIC-22695 

#probabilities of variables are greater than 0.05 so we will remove some variables and construct the model
#model2
var2=bank.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,14,15,17,18,19,20,21,22,23,24,25,26,27]]
Model2=sm.logit('y~var2',data =bank).fit()
Model2.summary()
Model2.summary2() # AIC = 11295


#as AIC value for model2 is low we are going to consider model 2
print(np.exp(Model2.params)) #as logistic regression is function of odds
#prediction
pred=Model2.predict(bank.iloc[:,1:]) 
bank["bank_pred"]=pred# creating new column


#confusion matrix
C_Table=confusion_matrix(bank.y,bank.bank_pred>0.5)
C_Table
accuracy = (39012+1703)/(39012+910+3586+1703) 
accuracy  # 90 %

#ROC CURVE AND AUC
from sklearn import metrics
fpr,tpr,threshold = metrics.roc_curve(bank.y, bank.bank_pred)
#PLOT OF ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr) #area under curve 0.89 which is excellent value for cutoff value of 0.5
roc_auc

