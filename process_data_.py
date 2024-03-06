import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def categorical_columns_to_numeric(df):

    string_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for column in string_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
     #   label_encoders[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return df,string_columns

def find_maximum_interval_of_missing_values(df):
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    missing_dates = date_range[~date_range.isin(df.index)]
    
    if len(missing_dates) > 0:
        missing_intervals = []
        current_interval = 0
        
        for i in range(1, len(missing_dates)):
            current_interval += (missing_dates[i] - missing_dates[i-1]).total_seconds() / 3600
            if (missing_dates[i] - missing_dates[i-1]).total_seconds() / 3600 > 1:
                missing_intervals.append(current_interval)
                current_interval = 0
        
        if current_interval > 0:
            missing_intervals.append(current_interval)
        
        if missing_intervals:
            print("Maximum interval of missing values:", max(missing_intervals), "hours")
            precent=round(max(missing_intervals)/len(df)*100,1)
            print(f"Maximum {precent}% missing")
        else:
            print("No intervals of missing values greater than 1 hour found.")
    else:
        print("No missing dates found.")
    
 
    
def index_and_fix_missing_rows(df):
    
    df = df.sort_index()
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    df = df.reindex(date_range)
    
    return df


def fix_nans(df,columns_to_mode_fill):
    
    interpolated_df=df.copy()
    for column in df.columns:
        if column in columns_to_mode_fill:
            mode_value = df[column].mode().iloc[0]  # Calculate mode
            interpolated_df[column].fillna(mode_value, inplace=True)
        else:
            interpolated_df[column] = df[column].interpolate(method='linear')
    
    return interpolated_df


def drop_columns_with_no_variance(df):
    return df.loc[:, df.var() > 0]

def drop_low_correlation_columns(df, threshold, target_column):
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()

    # Identify columns with correlation less than abs(threshold)
    columns_to_drop = correlation_matrix[abs(correlation_matrix[target_column]) < threshold].index

    # Drop columns from the DataFrame
    df_filtered = df.drop(columns=columns_to_drop)

    return df_filtered


def plot_correlation_matrix(df,save=True):
    correlation_matrix = df.corr()
    def annotate_heatmap(data, **kwargs):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j + 0.5, i + 0.5, f'{data.iloc[i, j]:.2f}', ha='center', va='center', **kwargs)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot_kws={"size": 10}, annot=False)
    annotate_heatmap(correlation_matrix, size=10)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    if save:
        plt.savefig(f'corr_matrix.png',bbox_inches='tight')


def prepare_sequences_for_lstm(df, target_column, forecast_days, past_days):
    
    sequence_length = past_days*24
    forecast_size=forecast_days*24
    
    data = []
    target = []
    for i in range(len(df) - sequence_length - forecast_size):  # Exclude forecast size for testing
        data.append(df.iloc[i:i + sequence_length].values)
        target.append(df[target_column].iloc[i + sequence_length:i + sequence_length + forecast_size].values)
    data = np.array(data)
    target = np.array(target)
    
    return data, target

def train_test_split_and_normalize(data, target, test_size_percentage):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size_percentage, shuffle=False)
    
    # Normalize data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,scaler


def read_and_prepare_df(path_filename,target_column,cor_threshold,show_correlation_matrix=False,smoothing=True,window_size=120):
       
    df0 = pd.read_csv(path_filename)
    df0['Formatted Date'] = pd.to_datetime(df0['Formatted Date'])
    df0.set_index('Formatted Date', inplace=True)
    
     
    find_maximum_interval_of_missing_values(df0)

    df = index_and_fix_missing_rows(df0)
    df, string_columns = categorical_columns_to_numeric(df)
    df = drop_columns_with_no_variance(df)
    
    if smoothing:
        smoothed_df = df.rolling(window=window_size, min_periods=1).mean()
    
    df = fix_nans(df, columns_to_mode_fill=string_columns)


    if show_correlation_matrix:
        plot_correlation_matrix(df)
    
    df = df.drop(columns=['Apparent Temperature (C)']) #get rid of this column, as it could be a potential leak
 
    df = drop_low_correlation_columns(df,cor_threshold, target_column)

    df = df.loc[:'2016-11-01']  #crop the least month

     
    return df
