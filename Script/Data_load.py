import pandas as pd
import os 

csv_path = './Data/Original/'
pkl_path = './Data/Pickle/'

def data_load(file_name,flag):
    if flag == 0:
        df = pd.read_csv(csv_path + str(file_name) +'.csv')
        df.to_pickle(pkl_path + str(file_name) +'.pkl')
    elif flag == 1:
        df = pd.read_pickle(pkl_path  + str(file_name) +'.pkl')
        
    return df

def check_data(df):
    print(f"Data Shape {df.shape}")
    print('NaN percentage')
    print(df.isna().mean())
    print('Unique percentage')
    print(df.nunique() / len(df))
    display(df.describe())
    display(df.describe(include = 'O'))
    display(df.head(5))
    