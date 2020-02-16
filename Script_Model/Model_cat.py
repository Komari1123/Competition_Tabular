import os
import pandas as pd
import numpy as np
# 評価指標をimportする
from sklearn.metrics import r2_score
from catboost import CatBoost
from catboost import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold

def model_cat(train_df , target , params , cv_ope , cv_num):
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
        
        train_pool = Pool(train_x, label = train_y)
        valid_pool = Pool(valid_x, label = valid_y)
        
        model = CatBoost(params)
        model.fit(train_pool,
                  eval_set = [valid_pool],
                  use_best_model = True,
                  verbose_eval = 100,
                 )
        

        score_list.append(r2_score(valid_y, model.predict(valid_pool
#                                                          prediction_type = 'Class'
                                                         ))) 
        models.append(model)  
        print(f'fold{fold_ + 1} end\n' )

    print('平均 : Score ', round(np.mean(score_list),5))
    
    fe_importance = np.zeros((len(train_df.columns), cv_num))
    for fold_, gbm in enumerate(models): 
        importance_ = gbm.get_feature_importance()
        fe_importance[:,fold_] = importance_
        

    importance_mean = np.mean(fe_importance,axis = 1)
    importance_std  = np.std(fe_importance,axis = 1)

    
    importance = pd.DataFrame([importance_mean,importance_std], columns = train_df.columns, 
                              index =['importance_mean','importance_std']).T.sort_values('importance_mean', ascending =False)
    importance_plot = importance.iloc[0:10].sort_values('importance_mean', ascending =True)
    plt.figure(figsize=(7, 10))
    plt.barh(importance_plot.index,importance_plot['importance_mean'],alpha = 0.4,color = "blue",label = 'imp_mean')
    plt.barh(importance_plot.index,importance_plot['importance_std'],alpha = 0.4,color = "red", label = 'impor_std')
    plt.xlabel("Feature Importance")
    plt.legend()
    plt.show()
    
    display(importance)
        
    return models

