# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:34:39 2020

@author: bodda
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


data=pd.read_csv('E:/letsupgrade AI_ML/day 11/general_data.csv')
data=data.drop_duplicates()
print(data.head())

#checking for null values

print(data.isnull().sum())

data=data.dropna(axis=0)
print(data.columns)
print(data.isnull().sum())
print(data.dtypes)
print(data.describe())

# Reassign target
data.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)
# Drop useless feat
data = data.drop(columns=['StandardHours', 
                          'EmployeeCount', 
                          'Over18','EmployeeID'
                        ])
print(data.head())
Attrition_people=data[data['Attrition']==1]
k=Attrition_people.describe()
non_Att_p=data[data['Attrition']==0]
f=non_Att_p.describe()

col=['Age','DistanceFromHome','JobLevel','MonthlyIncome','PercentSalaryHike','YearsSinceLastPromotion']
print("MEDIAN and MEAN OF CHURNED AND STAYED ")
for i in col:
    print("churned :",str(i))
    print(Attrition_people[i].median(),Attrition_people[i].mean())
    print("Stayed :",str(i))
    print(non_Att_p[i].median(),non_Att_p[i].mean())
#r,p=pearsonr(data.Education,data.Attrition)
#print(r)
plt.figure(figsize=(30,30))
corr=data.corr()

sns.heatmap(corr,annot=True)
plt.show()

'''
CONCLUSIONS

FROM THE HEATMAP WE CAN TELL THE ATTRITION IS POSITIVELY CORRELATED WITH PERCENT SALARY HIKE AND NUMCOMPANIES WORKED
AND NEGATIVELY CORRELATED WITH ALL OTHER VARIABLES CONSIDERED  AND THE VARIABLES ARE HAVING WEAK CORRELATION

BY COMPARING THE F AND K DATAFRAMES WE CAN GET DESCRIPTIVE ANALYSIS OF DATA
EMPLOYEE WITH AGE MEAN OF 33 ARE LIKELY TO CHURN AND DISTANCE FROM HOME >7 LIKELY TO CHURN
'''
