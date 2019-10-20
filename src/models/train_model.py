import logging
import os
import pickle
import random
from pathlib import Path
from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.make_dataset import DATE_COLUMNS, CAT_COLUMNS
from lightgbm import LGBMRegressor, LGBMModel
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

project_dir = Path(__file__).resolve().parents[2]

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
# exclude_cols = ['FinalDrillDate', 'RigReleaseDate', 'SpudDate','UWI']
exclude_cols = ['FinalDrillDate', 'RigReleaseDate', 'SpudDate', 'UWI']


class LogLGBM(LGBMRegressor):

    def fit(self, X, Y, **kwargs):
        y_train = np.log(Y)
        super(LogLGBM, self).fit(X, y_train, **kwargs)

        return self

    def predict(self, X):
        preds = super(LogLGBM, self).predict(X)
        preds = np.exp(preds)
        return preds


def main(input_file_path, output_file_path, tgt='Oil_norm', n_splits=5):
    input_file_name = os.path.join(input_file_path, 'Train_final.pck')
    input_file_name_test = os.path.join(input_file_path, 'Test_final.pck')
    input_file_name_val = os.path.join(input_file_path, 'Validation_final.pck')

    output_file_name = os.path.join(output_file_path, f'models_lgbm_{tgt}.pck')

    df = pd.read_pickle(input_file_name).drop(exclude_cols, axis=1)
    df_test = pd.read_pickle(input_file_name_test)
    df_val = pd.read_pickle(input_file_name_val).drop(exclude_cols, axis=1)

    ids = df_test['EPAssetsId']
    ids_uwi = df_test['UWI']

    df_test = df_test.drop(exclude_cols, axis=1)

    cv = KFold(n_splits=n_splits, shuffle=True)
    models = []
    scores = []
    scores_dm = []
    y = df.loc[~df[tgt].isna(), tgt]
    X = df.loc[~df[tgt].isna(), :].drop(['Oil_norm', 'Gas_norm', 'Water_norm', 'EPAssetsId', '_Normalized`IP`BOE/d'],
                                        axis=1)
    X_test = df_test.copy().drop('EPAssetsId', axis=1)

    X_holdout, y_holdout = df_val.loc[~df_val[tgt].isna(), :].drop(
        ['Oil_norm', 'Gas_norm', 'Water_norm', 'EPAssetsId', '_Normalized`IP`BOE/d'],
        axis=1), df_val.loc[~df_val[tgt].isna(), tgt]

    preds_test = np.zeros((n_splits, df_test.shape[0]))
    preds_holdout = np.zeros((n_splits, df_val.shape[0], 3))

    for k, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]

        # model = LGBMRegressor(num_leaves=16, learning_rate=0.1, n_estimators=300, reg_lambda=30, reg_alpha=30,
        # objective='mae',random_state=123)

        model = LogLGBM(num_leaves=16, learning_rate=0.05, n_estimators=600, reg_lambda=20, reg_alpha=20,
                        objective='mae', random_state=123, feature_fraction=0.7)
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        geom_mean = gmean(y_train)
        dm = DummyRegressor(strategy='constant', constant=geom_mean)

        model.fit(X_train, y_train, categorical_feature=CAT_COLUMNS)
        # model.fit(X_train, y_train)
        dm.fit(X_train, y_train)

        score = mean_absolute_error(y_holdout, model.predict(X_holdout))
        score_dm = mean_absolute_error(y_val, dm.predict(X_val))

        # logging.info(f' Score = {score}')
        models.append(model)
        scores.append(score)
        scores_dm.append((score_dm))
        preds_test[k, :] = model.predict(X_test).reshape(1, -1)
        preds_holdout[k, :] = model.predict(X_holdout).reshape(1, -1)

    with open(output_file_name, 'wb') as f:
        pickle.dump(models, f)
    logging.info(scores)
    logging.info(f'Mean scores LGBM = {np.mean(scores)}')
    logging.info(f'Mean scores Dummy = {np.mean(scores_dm)}')

    preds_df = pd.DataFrame({'EPAssetsID': ids, 'UWI': ids_uwi, tgt: preds_test.mean(axis=0)})
    return preds_df


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, 'data', 'final')
    output_file_path = os.path.join(project_dir, 'models')
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    preds_oil = main(input_file_path, output_file_path, tgt='Oil_norm')
    preds_gas = main(input_file_path, output_file_path, tgt='Gas_norm')
    preds_water = main(input_file_path, output_file_path, tgt='Water_norm')

    df = preds_oil
    df_merge_list = [preds_gas, preds_water]
    for x in df_merge_list:
        df = pd.merge(df, x.drop('UWI', axis=1), on='EPAssetsID')
    submission = df.sort_values('EPAssetsID')
    submission.columns = ['EPAssetsId', 'UWI', '_Normalized`IP`(Oil`-`Bbls)', '_Normalized`IP`Gas`(Boe/d)',
                          '_Normalized`IP`(Water`-`Bbls)']
    submission.to_csv(os.path.join(input_file_path, 'submission_lgbm.txt'), index=False)
# EPAssetsID,UWI,Oil_norm,Gas_norm,Water_norm
