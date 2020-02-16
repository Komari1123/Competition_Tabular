import os 
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def pre_cat(df_train , df_test , cat_list , prepro_type):
    
    if prepro_type == 'one-hot':
        df_train = pd.get_dummies(df_train, columns = cat_list)
        df_test  = pd.get_dummies(df_test , columns = cat_list)
        
    if prepro_type == 'label':
        for c in cat_list:
            le = LabelEncoder()
            le.fit(df_train[c])
            df_train[c] = le.transform(df_train[c])
            df_test[c]  = le.transform(df_test[c])
            
    if prepro_type == 'freq':
        for c in cat_list:
            freq = df_train[c].value_counts()
            df_train[c] = df_train[c].map(freq)
            df_test[c]  = df_test[c].map(freq)
        
        
    return df_train , df_test