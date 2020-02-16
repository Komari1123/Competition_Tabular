import os
import pandas as pd
import numpy as np
# 評価指標をimportする
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
import warnings
warnings.filterwarnings("ignore")
import random as rn
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils


tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def model_nn(train_df , target , cv_ope , cv_num):
    score_list = [] 
    models = []
    if cv_ope == 'kf':
        kf = KFold(n_splits = cv_num, shuffle=True,random_state = 0)
    
    if cv_ope == 'stkf':
        kf = StratifiedKFold(n_splits = cv_num, shuffle=True,random_state = 0)
    
    if cv_ope == 'gkf':
        kf = GroupKFold(n_splits = cv_num, shuffle=True,random_state = 0)
        #enumerate(kf.split(train_df , target , group_key))

    for fold_, (train_index, valid_index) in tqdm(enumerate(kf.split(train_df,target))): 
        print(f'fold{fold_ + 1} start')
        train_x , valid_x = train_df.iloc[train_index] , train_df.iloc[valid_index]
        train_y , valid_y  = target[train_index] , target[valid_index]
        
        
        train_x = train_df.iloc[train_index]
        valid_x = train_df.iloc[valid_index]
        train_y = target[train_index] 
        valid_y = target[valid_index] 

        model = Sequential()
        model.add(Dense(200, activation = 'relu',input_shape = (train_x.shape[1],)))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(50, activation =  'relu'))
        model.add(Dense(1))

        model.compile(Adam(lr=5e-3), loss = "mean_absolute_error")
        early_stopping = EarlyStopping(patience=200, verbose=1,mode='auto')


        history = model.fit(train_x, train_y, batch_size=None, 
                            epochs=20000, verbose= 0,callbacks=[early_stopping],validation_data=(valid_x, valid_y))


        score_list.append(r2_score(valid_y, model.predict(valid_x))) 
        print(r2_score(valid_y, model.predict(valid_x)))
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        models.append(model)  
        print(f'fold{fold_ + 1} end\n' )

    print('平均 : Score ', round(np.mean(score_list),5))
    
    
        
    return models