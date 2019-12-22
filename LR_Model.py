# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:33:53 2019

@author: Aman Pandey
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import os

class Linear_Regression():
    def __init__(self, x_data,y_data):
        self.x_data = x_data
        self.y_data = y_data
      #  self.country_stats = self.prep_data()
        
        
#    def prep_data(self):
#        combi = self.x_data.append(self.y_data)
#        return combi
#    
#    def visual(self):
#        print(self.country_stats.head())
#        self.country_stats.plot(x = "2015", y = 'Value')
#        plt.show()
        
    def predict(self, x_new = [[7587]]):
      #  self.country_stats.dropna()
        lin_reg_model = sklearn.linear_model.LinearRegression()
        X = np.c_[self.x_data["2015"]]
        y = np.c_[self.y_data["Value"]]
   #     print(X,y)
     #   X.reshape(-1,1)
       # y.reshape(-1,1)
        where = np.where(np.isnan(X))
   #     print(where)
     #   print(y.shape)
        for i in where:
           print(i)
           X[i] = np.nan_to_num(X[i])
       #    print(X[i])
        where2 = np.where(np.isnan(y))
        print(where2)
        print(X.shape,y.shape)
#        for j in where2:
#         #   print(j)
#            y[j]= np.nan_to_num(y[j])
#      #  for i,j in where:
       #     np.nan_to_num(X(i,j)) 
               
               
        
       # y=y.replace(np.nan,0)
      #  print(y)
        lin_reg_model.fit(X[:190],y[:190])
        print(lin_reg_model.coef_,lin_reg_model.intercept_)
        y_new = lin_reg_model.predict(x_new)
        return y_new


if __name__ == '__main__':
    oecd = pd.read_csv('lifesat\oecd_bli_2015.csv',thousands = ',')
    gdp = pd.read_csv('lifesat\gdp_per_capita.csv', thousands = ',', delimiter = '\t',encoding = 'latin1', na_values ="n/a")
  #  gdp =gdp.fillna(gdp.mean(), inplace = True)
   # gdp= gdp.dropna(how = 'any', axis =0)
    
  #  oecd = oecd.fillna(oecd.mean(), inplace = True)
  #  oecd = oecd.dropna(how = 'any', axis =0)
    print(gdp.head())
    lr = Linear_Regression(gdp,oecd)
 #   lr.visual()
    print(lr.predict())