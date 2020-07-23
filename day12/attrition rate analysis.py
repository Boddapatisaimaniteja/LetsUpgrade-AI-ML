# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:39:51 2020

@author: bodda
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


data=pd.read_csv('E:/letsupgrade AI_ML/day 11/general_data.csv')
data=data.drop_duplicates()
print(data.head())
print(data.columns)

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
from scipy.stats import mannwhitneyu
col=['Age','DistanceFromHome','JobLevel','MonthlyIncome','PercentSalaryHike','YearsSinceLastPromotion','YearsAtCompany','YearsWithCurrManager','TotalWorkingYears','NumCompaniesWorked']

p_values=[]
for i in col:
    stats,p=mannwhitneyu(Attrition_people[i],non_Att_p[i])
    p_values.append(p)
 
'''    
INFERENCES 
    1:since Age p value is less 0.05,there is no significance effect on attrition with age
    2:since Distance from home p values >0.05,there is significance effect on attrition with  Distance from home
    3:since JobLevel p values >0.05,there is significance effect on attrition with  JobLevel
    4:since MonthlyIncome p values >0.05,there is significance effect on attrition with  MonthlyIncome
    5:since PercentSalaryHike p value is less 0.05,there is no significance effect on attrition with PercentSalaryHike
    6:since YearsSinceLastPromotion p value is less 0.05,there is no significance effect on attrition with YearsSinceLastPromotion
    7:since YearsAtCompany p value is less 0.05,there is no significance effect on attrition with YearsAtCompany
    8:since YearsWithCurrManager p value is less 0.05,there is no significance effect on attrition with YearsWithCurrManager
    9:since TotalWorkingYears p value is less 0.05,there is no significance effect on attrition with TotalWorkingYears
    10:since NumCompaniesWorked p value is less 0.05,there is no significance effect on attrition with NumCompaniesWorked
    conclusion:
        the people who are having less distance to the company, will have high chance to stay and joblevel is high,curning may decrease and increase in monthlyincome may decrease the attrition rate
    solution:
        the people from long distamce to company ,increase in monthly income to those people may decrease attrition rate and if the joblevel is increased to the capable people ,may reduce the attrtition rate
        
'''
