# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:22:51 2020

@author: bodda
"""


import pandas as pd
import numpy as np

data=pd.read_csv(r'E:\letsupgrade AI_ML\day25\titanic.csv')
data=data.drop(['PassengerId','Name','Ticket','Cabin','Age','Fare'],axis=1)

data['Embarked']=data['Embarked'].astype(str)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Sex']=le.fit_transform(data['Sex'])
data['Embarked']=le.fit_transform(data['Embarked'])

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

def train_model(x_train,x_test,y_train,y_test):
    
    classifier=GaussianNB()
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    
    ac=accuracy_score(y_test,y_pred)
    
    print(confusion_matrix(y_test,y_pred))
    
    
    print(ac)
    
    
for i in data.columns:
    print("DV",i)
    x=data.drop([i],axis=1)
    y=data[i]
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
    train_model( x_train,x_test,y_train,y_test)
    
    

    