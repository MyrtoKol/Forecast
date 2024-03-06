import pmdarima as pm 
import pickle 
from keras.models import load_model  # Import load_model function
import sys
import matplotlib.pyplot as plt  # Add matplotlib for plotting
from sklearn.metrics import mean_squared_error

import os

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))

from process_data_ import * 

def load_arima_model():
    with open("arima_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_lstm_model(folder):
    from keras.models import load_model
    model = load_model("lstm_model.h5")
    return model

def arima_predict(model, exog_test, prediction_size):
    forecast = model.predict(prediction_size, exogenous=exog_test)
    return forecast

def lstm_predict(model, X_test_scaled, scaler):
    forecast = model.predict(X_test_scaled)
    forecast_actual = scaler.inverse_transform(forecast)
    return forecast_actual

def main(method,forecast_days,train_days,target_column):
     
     filename = 'interview_dataset.csv'
     df = read_and_prepare_df(filename,target_column='Temperature (C)', cor_threshold=0.4,show_correlation_matrix=True)
    
     if method == 'arima':
        # ARIMA Model
        arima_model = load_arima_model()
        
        prediction_size=forecast_days*24
        train_size=train_days*24

        train, test = df.iloc[-train_size-prediction_size:-prediction_size], df.iloc[-prediction_size:]
        exog_test = test.drop(columns=[target_column])
          

        arima_forecast = arima_predict(arima_model, exog_test, prediction_size)
        
        print(arima_forecast)
        mse = mean_squared_error(test[target_column],arima_forecast)
        
        print('Test MSE: %.3f' % mse)
        
        # Plotting
        plt.figure(figsize=(20,10))
        plt.plot(train.iloc[-500:].index, train.iloc[-500:][target_column], label='Past data')
        plt.plot(test.index, test[target_column], label='Actual')
        plt.plot(arima_forecast.index, arima_forecast, color='red', label='Forecast')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.title('ARIMA Forecast with Exogenous Variables')
        plt.legend()
        plt.show()
        plt.text(0.5, 0.95, 'Test MSE: %.3f' % mse, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig(f'arima_visualize.png',bbox_inches='tight')

        
        return arima_forecast
   
     elif method == 'lstm':
        
        train_days=90
       
        lstm_model = load_lstm_model(folder)
        data, target = prepare_sequences_for_lstm(df, target_column, forecast_days, train_days)
        _, X_test_scaled, _, y_test_scaled,scaler=train_test_split_and_normalize(data, target, test_size_percentage=0.2)
        lstm_forecast = lstm_predict(lstm_model, X_test_scaled, scaler)
        
        y_test=scaler.inverse_transform(y_test_scaled)
        
        mse = mean_squared_error(y_test[-3:-1, :].flatten(),lstm_forecast[-3:-1, :].flatten())
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-len(y_test[-3:-1,:].flatten()):],y_test[-3:-1, :].flatten(), label='Actual', color='blue')
        plt.plot(df.index[-len(lstm_forecast[-3:-1,:].flatten()):],lstm_forecast[-3:-1, :].flatten(), label='Forecast', color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (C)')
        plt.title('Actual vs Forecasted Temperatures')
        plt.legend()
        plt.text(0.5, 0.95, 'MSE: {:.2f}%'.format(mse), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        plt.show()
        plt.savefig(f'lstm_visualize_days={train_days}.png',bbox_inches='tight')

        return lstm_forecast
    
     else:
        print("Invalid method specified. Please choose either 'arima' or 'lstm'.")
        return None

if __name__ == "__main__":
    
    forecast = main(method='arima',forecast_days=2,train_days=120,target_column='Temperature (C)')
    print(forecast)
