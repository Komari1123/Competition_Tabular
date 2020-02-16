import pandas as pd
import os

def get_grpby_features(data, agg_name_list, categorical_columns, numeric_columns):
    fe_name_prefix = 'Groupby_'
    features = pd.DataFrame()
    total = data

    for agg_name in agg_name_list:
        for cat_col in categorical_columns:
            for num_col in numeric_columns:
                agg = total.groupby(cat_col)[num_col].agg(agg_name)
                features[fe_name_prefix + f'{num_col}_{agg_name}_grpby_{cat_col}'] = total[cat_col].map(agg)
                features[fe_name_prefix + f'{num_col}_{agg_name}_grpby_{cat_col}_ratio'] = total[num_col] / total[cat_col].map(agg)
                features[fe_name_prefix + f'{num_col}_{agg_name}_grpby_{cat_col}_diff'] = total[num_col] - total[cat_col].map(agg)

    train_features = features
    return train_features