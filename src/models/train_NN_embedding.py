import logging
import os
import pickle
import eli5

from pathlib import Path
from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd
from src.data.make_dataset import CAT_COLUMNS
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from src.models.NNPredictor import NNPredictor

project_dir = Path(__file__).resolve().parents[2]

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
# exclude_cols = ['FinalDrillDate', 'RigReleaseDate', 'SpudDate','UWI']
exclude_cols = ['FinalDrillDate', 'RigReleaseDate', 'SpudDate', 'UWI','OSArea','OSDeposit','SurveySystem']
cat_cols_new = CAT_COLUMNS
cat_cols_new = list(set(cat_cols_new) - (set(cat_cols_new) & set(exclude_cols)))

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
    X_all = df.loc[~df[tgt].isna(), :].drop(
        ['Oil_norm', 'Gas_norm', 'Water_norm', 'EPAssetsId', '_Normalized`IP`BOE/d'],
        axis=1)
    X_test = df_test.copy().drop('EPAssetsId', axis=1)

    X_holdout, y_holdout = df_val.loc[~df_val[tgt].isna(), :].drop(
        ['Oil_norm', 'Gas_norm', 'Water_norm', 'EPAssetsId', '_Normalized`IP`BOE/d'],
        axis=1), df_val.loc[~df_val[tgt].isna(), tgt]

    preds_test = np.zeros((n_splits, df_test.shape[0]))
    preds_holdout = np.zeros((n_splits, X_holdout.shape[0]))

    for k, (train_index, test_index) in enumerate(cv.split(X_all, y)):
        X_train, X_val = X_all.iloc[train_index, :], X_all.iloc[test_index, :]

        # model = LGBMRegressor(num_leaves=16, learning_rate=0.1, n_estimators=300, reg_lambda=30, reg_alpha=30,
        # objective='mae',random_state=123)
        logging.info(f'Creating a NNPredictor with {cat_cols_new}')
        predictor_obj = NNPredictor(categorical_features=cat_cols_new,
                                    numerical_features=X_train.columns.difference(cat_cols_new),
                                    data=X_train)

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        geom_mean = gmean(y_train)
        dm = DummyRegressor(strategy='constant', constant=geom_mean)
        proc_X_train, proc_X_holdout, proc_X_test = predictor_obj.preprocess_data(X_train, X_holdout, X_test)

        predictor_obj.fit(x=proc_X_train, y=y_train.values, batch_size=16, epochs=5)
        # model.fit(X_train, y_train)
        dm.fit(X_train, y_train)

        score = mean_absolute_error(y_holdout, predictor_obj.predict(x=X_holdout))
        score_dm = mean_absolute_error(y_val, dm.predict(X_val))

        # logging.info(f' Score = {score}')
        models.append(predictor_obj)
        scores.append(score)
        scores_dm.append(score_dm)
        logging.info(f'Holdout score = {score}')
        preds_test[k, :] = predictor_obj.predict(x=proc_X_test).reshape(1, -1)
        preds_holdout[k, :] = predictor_obj.predict(x=proc_X_holdout).reshape(1, -1)

    with open(output_file_name, 'wb') as f:
        pickle.dump(models, f)
    logging.info(scores)
    logging.info(f'Mean scores LGBM = {np.mean(scores)}')
    logging.info(f'Mean scores Dummy = {np.mean(scores_dm)}')

    preds_df = pd.DataFrame({'EPAssetsID': ids, 'UWI': ids_uwi, tgt: preds_test.mean(axis=0)})
    preds_df_val = pd.DataFrame({tgt: preds_holdout.mean(axis=0),
                                 'gt': y_holdout})
    score_holdout = mean_absolute_error(preds_df_val['gt'], preds_df_val[tgt])
    logging.info(f'Final score on holdout: {score_holdout}')
    print(eli5.format_as_dataframe(eli5.explain_weights(model)))

    return preds_df, score_holdout


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, 'data', 'final')
    output_file_path = os.path.join(project_dir, 'models')
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    preds_oil, score_holdout_oil = main(input_file_path, output_file_path, tgt='Oil_norm')
    preds_gas, score_holdout_gas = main(input_file_path, output_file_path, tgt='Gas_norm')
    preds_water, score_holdout_water = main(input_file_path, output_file_path, tgt='Water_norm')
    logging.info(f'Scores are: oil {score_holdout_oil}, gas {score_holdout_gas} water {score_holdout_water}')
    df = preds_oil
    df_merge_list = [preds_gas, preds_water]
    for x in df_merge_list:
        df = pd.merge(df, x.drop('UWI', axis=1), on='EPAssetsID')
    submission = df.sort_values('EPAssetsID')
    submission.columns = ['EPAssetsId', 'UWI', '_Normalized`IP`(Oil`-`Bbls)', '_Normalized`IP`Gas`(Boe/d)',
                          '_Normalized`IP`(Water`-`Bbls)']
    submission.to_csv(os.path.join(input_file_path, 'submission_lgbm.txt'), index=False)
# EPAssetsID,UWI,Oil_norm,Gas_norm,Water_norm
