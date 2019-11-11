import sys
from timeit import default_timer as timer

from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin

sys.path.insert(0, '/home/jovyan/anton/power_repos/pg_model')

import csv
import logging
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from hyperopt import base
base.have_bson = False

project_dir = Path(__file__).resolve().parents[2]

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
# exclude_cols = ['FinalDrillDate', 'RigReleaseDate', 'SpudDate','UWI']

exclude_cols_oil = ["UWI", "CompletionDate", 'DaysDrilling',
                    'DrillMetresPerDay',
                    'GroundElevation',
                    'HZLength',
                    'LengthDrill',
                    'Municipality',
                    #'Pool',
                    'SurfaceOwner',
                    '_Fracture`Stages',
                    'final_timediff',
                    'lic_timediff',
                    'rrd_timediff',
                    'st_timediff','Confidential','SurfAbandonDate']

exclude_cols_gas = ['ConfidentialReleaseDate', "UWI", "CompletionDate",
                    'CurrentOperator',
                    'DaysDrilling',
                    'DrillMetresPerDay',
                    'DrillingContractor',
                    'FinalDrillDate',
                    'KBElevation',
                    'LengthDrill',
                    'LicenceDate',
                    'Municipality',
                    'Pool',
                    'ProjectedDepth',
                    'RigReleaseDate',
                    'SpudDate',
                    'StatusSource',
                    'SurfaceOwner',
                    'TVD',
                    'TotalDepth',
                    'UnitName',
                    '_Fracture`Stages',
                    'cf_timediff',
                    'final_timediff',
                    'rrd_timediff',
                    'st_timediff','Confidential','SurfAbandonDate']
exclude_cols_water = ['ConfidentialReleaseDate',"UWI", "CompletionDate",
                      'DaysDrilling',
                      'DrillMetresPerDay',
                      'FinalDrillDate',
                      'GroundElevation',
                      'HZLength',
                      'KBElevation',
                      'LaheeClass',
                      'LicenceDate',
                      'Licensee',
                      'ProjectedDepth',
                      'SpudDate',
                      'TotalDepth',
                      '_Fracture`Stages',
                      'cf_timediff',
                      'final_timediff',
                      'lic_timediff',
                      'rrd_timediff',
                      'st_timediff','Confidential','SurfAbandonDate']

exclude_cols_dict = {'Oil_norm': exclude_cols_oil,
                     'Gas_norm': exclude_cols_gas,
                     'Water_norm': exclude_cols_water}

class LogLGBM(LGBMRegressor):

    def fit(self, X, Y, **kwargs):
        y_train = np.log(Y)
        super(LogLGBM, self).fit(X, y_train, **kwargs)

        return self

    def predict(self, X):
        preds = super(LogLGBM, self).predict(X)
        preds = np.exp(preds)
        return preds


def get_score(model, X, y, X_val, y_val):
    model.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    score = mean_absolute_error(y_val, model.predict(X_val))
    print(score)
    return score


def main(input_file_path, tgt='Oil_norm'):
    note = "LGBM"
    input_file_name = os.path.join(input_file_path, 'Train_final.pck')
    input_file_name_val = os.path.join(input_file_path, 'Validation_final.pck')
    exclude_cols = exclude_cols_dict[tgt]
    df = pd.read_pickle(input_file_name).drop(exclude_cols, axis=1)
    df_val = pd.read_pickle(input_file_name_val).drop(exclude_cols, axis=1)

    y = df.loc[~df[tgt].isna(), tgt]
    X = df.loc[~df[tgt].isna(), :].drop(['Oil_norm', 'Gas_norm', 'Water_norm', 'EPAssetsId', '_Normalized`IP`BOE/d'],
                                        axis=1)

    X_holdout = df_val.loc[~df_val[tgt].isna(), :].drop(
        ['Oil_norm', 'Gas_norm', 'Water_norm', 'EPAssetsId', '_Normalized`IP`BOE/d'],
        axis=1)
    y_holdout = df_val.loc[~df_val[tgt].isna(), tgt]

    # Prep output files for hyperopt for performance tracking:
    trials = Trials()

    space = {
        "min_data_in_leaf": hp.uniform("min_data_in_leaf", 1, 40),
        'num_leaves': hp.quniform('num_leaves', 30, 128, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.1, 0.9),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.1, 1.0),
        'lambda_l1': hp.uniform('lambda_l1', 0.1, 10),
        'lambda_l2': hp.uniform('lambda_l2', 0.1, 10)

    }

    fpath = f'{project_dir}/models/{note}_{tgt}_feats_final_Trials.pkl'
    fpath_csv = f'{project_dir}/models/{note}_{tgt}_feats_final_Trials.csv'
    # File to save first results
    of_connection = open(fpath_csv, 'w')
    writer = csv.writer(of_connection)
    # Write the headers to the file
    writer.writerow(['loss', *list(space.keys()), 'train_time'])
    of_connection.close()


    def objective(params):
        params = {
            "min_data_in_leaf": int(params['min_data_in_leaf']),
            "num_leaves": int(params['num_leaves']),
            "feature_fraction": "{:.3f}".format(params['feature_fraction']),
            "bagging_fraction": '{:.3f}'.format(params['bagging_fraction']),
            "lambda_l1": params['lambda_l1'],
            "lambda_l2": params['lambda_l2']

        }
        m = LogLGBM(learning_rate=0.05, n_estimators=500,
                    objective='mse', random_state=123, **params)

        start_time = timer()
        score = get_score(m, X, y, X_holdout, y_holdout)
        run_time = timer() - start_time
        # Write to the csv file ('a' means append)
        of_connection = open(fpath_csv, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([score, *list(params.values()), run_time])
        of_connection.close()
        print("Score {:.3f} params {}".format(score, params))
        return score

    best_lgbm = fmin(fn=objective,
                     space=space,
                     algo=tpe.suggest,
                     max_evals=250, trials=trials)

    losses = [trials.trials[i]['result']['loss'] for i in range(len(trials.trials))]
    params = pd.DataFrame(trials.vals)
    params['loss'] = losses
    params.sort_values('loss', inplace=True)
    with open(fpath, 'wb') as f:
        pickle.dump(trials, f)
    df_hyperparams = pd.read_csv(fpath_csv).sort_values('loss')
    df_hyperparams.to_csv(fpath_csv)


if __name__ == '__main__':
    # Read data in
    input_file_path = os.path.join(project_dir, 'data', 'final')
    output_file_path = os.path.join(project_dir, 'models')
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    main(input_file_path, tgt='Water_norm')
    main(input_file_path, tgt='Gas_norm')
    main(input_file_path, tgt='Oil_norm')
