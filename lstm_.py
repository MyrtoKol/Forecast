from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import os

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))

from process_data_ import * 

def lstm_model(df,target_column,past_days,forecast_days, epochs, batch_size, learning_rate,units):
    # Define optimizer
    
    data,target=prepare_sequences_for_lstm(df, target_column, forecast_days, past_days)
    
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,scaler=train_test_split_and_normalize(data, target, test_size_percentage=0.2)
    optimizer = Adam(learning_rate=learning_rate)
    
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(LSTM(units=units))
    model.add(Dense(units=y_train_scaled.shape[1]))  # Units should match the number of features in y_train_scaled
    model.compile(optimizer=optimizer, loss='mse')

    # Train model
    model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    # Save model
    model.save(f"lstm_model.h5")
   
    # from tensorflow.keras.models import load_model 
    # model =load_model(f"{folder}/lstm_model.h5")
    
    # Evaluate model
  #  loss = model.evaluate(X_test_scaled, y_test_scaled)
    
    forecast = model.predict(X_test_scaled)
    forecast_actual = scaler.inverse_transform(forecast)
    

    return forecast_actual

def main(target_column,forecast_days,epochs,batch_size,learning_rate,units,past_days,cor_threshold=0.4):
    
    path_filename = 'interview_dataset.csv'

    df = read_and_prepare_df(path_filename,target_column, cor_threshold)
 
         
    lstm_forecast = lstm_model(df,target_column,past_days,forecast_days, epochs, batch_size, learning_rate,units)


if __name__ == "__main__":
    
    main(target_column='Temperature (C)',forecast_days = 2,epochs=10,batch_size=32,learning_rate=0.001,units=50,past_days=90)
    