import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pmdarima as pm 
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle

from process_data_ import * 

#####################################

def arima_model(df,target_column,prediction_days,train_days,max_p,max_q,max_d,seasonal,m):
        
    prediction_size=prediction_days*24
    train_size=train_days*24
    train, test = df.iloc[-train_size-prediction_size:-prediction_size], df.iloc[-prediction_size:]
    
    exog_train =train.drop(columns=[target_column])
     
    model = pm.auto_arima(train['Temperature (C)'], 
                          exogenous=exog_train,
                          m=m,               # frequency of series                      
                          seasonal=seasonal,  # TRUE if seasonal series
                          test='adf',         # use adftest to find optimal 'd'
                         # start_p=2, start_q=0,
                         # start_d=1,
                          max_p=max_p, max_q=max_q, max_d=max_d,
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True)
    
    # Summary of the best modeltest
    
  #  print(model.summary())
    
    with open("arima_model.pkl", "wb") as f:
        pickle.dump(model, f)

    forecast = model.predict(prediction_size)
        
    return forecast

def main(prediction_days, train_days, target_column='Temperature (C)',cor_threshold=0.4, max_p=None, max_q=None, max_d=3, seasonal=False, m=24):
   
    path_filename = 'interview_dataset.csv'
    df = read_and_prepare_df(path_filename,target_column, cor_threshold=cor_threshold)
 
    forecast = arima_model(df,target_column,prediction_days,train_days,max_p,max_q,max_d,seasonal,m)
  #  forecast = pd.DataFrame({'timestamp': forecast.index, 'forecast_value': forecast.values})
    
    
    return forecast


if __name__ == "__main__":
    main(prediction_days = 2,train_days = 120)