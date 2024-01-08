# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:26:18 2022

@author: HP
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df = pd.read_csv("U00.csv", delimiter=';', decimal=",")

df = shuffle(df)

import warnings
warnings.filterwarnings("ignore")

# Correlation check
df_cor=df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]]

c=df_cor.corr()

#############################################################################
###############   PYCARET   ##################################################

from pycaret.regression import *

models = [
    'ridge',
    'lightgbm',
    'lasso',
    'lr',
    'rf',
    'mlp'
    ]

data = df.sample(frac=0.85, random_state=886)
data_unseen = df.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)


exp_reg101 = setup(data = data, target = 'U0_VALUE', html=False, silent=True, session_id=199)
best_model = compare_models(sort = 'MAE')

import warnings
warnings.filterwarnings("ignore")


ET = create_model('et')
print(ET)
#tuned_ET = tune_model(ET, n_iter=100, optimize="Accuracy")      
#print(tuned_ET)
plot_model(ET, plot = 'residuals')   
plot_model(ET, plot = 'cooks') 
plot_model(ET, plot='feature')    
plot_model(ET, plot='rfe')   
plot_model(ET, plot = 'error')    
plot_model(ET, plot = 'vc')    
plot_model(ET, plot = 'learning')    
plot_model(ET, plot = 'manifold') 
plot_model(ET, plot = 'parameter') 
interpret_model(ET)

unseen_predictions = predict_model(ET, data=df)   
unseen_predictions = unseen_predictions.reset_index(drop=True) 


unseen_predictions.head()    
from pycaret.utils import check_metric
check_metric(unseen_predictions.U0_VALUE, unseen_predictions.Label, 'MAE')

save_model(ET,'Extra_Trees_Regressor_PIM_11012023')

plt.plot(range(0,100),unseen_predictions.iloc[:100,-1], unseen_predictions.iloc[:100,-2])
plt.title('Predictions vs Real results')
plt.xlabel('observation')
plt.ylabel('U0_Value')
plt.legend(['Predicted result','Real result'], loc='upper right')
plt.show()

