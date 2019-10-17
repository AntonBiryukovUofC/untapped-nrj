import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.make_dataset import DATE_COLUMNS, CAT_COLUMNS
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

project_dir = Path(__file__).resolve().parents[2]

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def main(input_file_path, output_file_path, tgt='Oil_norm'):
    input_file_name = os.path.join(input_file_path, 'train_final.pck')
    output_file_name = os.path.join(output_file_path, f'models_lgbm_{tgt}.pck')

    df = pd.read_pickle(input_file_name).sample(frac=0.2)
    model = LGBMRegressor(num_leaves=12, learning_rate=0.1, n_estimators=300, reg_lambda=20, reg_alpha=20,
                          objective='mae')
    cv = KFold(n_splits=5)
    models = []
    scores = []
    y = df.loc[~df[tgt].isna(),tgt]
    X = df.loc[~df[tgt].isna(),:].drop(['Oil_norm', 'Gas_norm', 'Water_norm', 'EPAssetsId'],axis=1)


    for train_index, test_index in cv.split(X):
        X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train, categorical_feature=CAT_COLUMNS)
        score = mean_absolute_error(y_val, model.predict(X_val))
        logging.info(f' Score = {score}')
        models.append(model)
        scores.append(score)
    with open(output_file_name, 'wb') as f:
        pickle.dump(models, f)
    logging.info(scores)

if __name__ == '__main__':
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, 'data', 'final')
    output_file_path = os.path.join(project_dir, 'models')
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    main(input_file_path, output_file_path)
