import os
import pandas as pd
import numpy as np
# 評価指標をimportする
from sklearn.metrics import r2_score
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold

def model_xgb(train_df , target , params , cv_ope , cv_num):
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
        
        dtrain = xgb.DMatrix(train_x,label = train_y)
        dvalid = xgb.DMatrix(valid_x,label = valid_y)
        
        evals = [(dtrain,'train'),(dvalid,'eval')]
        evals_result = {}
        
        xgbt = xgb.train(params, 
                        dtrain,
#                         num_boost_round = 10000,
                        early_stopping_rounds = 200,
                        evals = evals,
                        evals_result = evals_result,
                        verbose_eval = 100 ,
                       )
        
        score_list.append(r2_score(valid_y, xgbt.predict(xgb.DMatrix(valid_x)))) 
        models.append(xgbt)  
        print(f'fold{fold_ + 1} end\n' )

    print('平均 : Score ', round(np.mean(score_list),5))
    
    importance_0 = models[0].get_score(importance_type = 'total_gain')
    importance_df = pd.DataFrame(importance_0.values(),index =importance_0.keys() )

    for m in models[1:]:
        s = m.get_score(importance_type = 'total_gain')
        df = pd.DataFrame(s.values(),index = s.keys())
        importance_df= pd.merge(importance_df,df,how = 'outer',left_index = True ,right_index = True)

    importance_df_mean = pd.DataFrame(importance_df.mean(axis=1), columns=['importance']).fillna(0)
    importance_df_std = pd.DataFrame(importance_df.std(axis=1), columns=['importance']).fillna(0)

    importance = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True ,suffixes=['_mean', '_std'])
    
    
    importance_plot = importance.iloc[0:10].sort_values('importance_mean', ascending =True)
    plt.figure(figsize=(7, 10))
    plt.barh(importance_plot.index,importance_plot['importance_mean'],alpha = 0.4,color = "blue",label = 'imp_mean')
    plt.barh(importance_plot.index,importance_plot['importance_std'],alpha = 0.4,color = "red", label = 'impor_std')
    plt.xlabel("Feature Importance")
    plt.legend()
    plt.show()
    
    display(importance)
        
    return models

