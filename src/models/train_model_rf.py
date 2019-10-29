import logging
import os
import pickle
import random
import eli5

from pathlib import Path
from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.make_dataset import DATE_COLUMNS, CAT_COLUMNS
from lightgbm import LGBMRegressor, LGBMModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PowerTransformer,FunctionTransformer
project_dir = Path(__file__).resolve().parents[2]


# def mean_log(preds):
#     lpreds =np.log1p(preds)
#     mean_preds = np.mean(lpreds,axis=0)
#     final_preds = np.expm1(mean_preds)
#     return final_preds
def mean_log(preds):
    final_preds = gmean(preds,axis=0)
    return final_preds


exclude_cols_oil = ["UWI", "CompletionDate", 'DaysDrilling',
                    'DrillMetresPerDay',
                    'GroundElevation',
                    'HZLength',
                    'LengthDrill',
                    'Municipality',
                    'Pool',
                    'SurfaceOwner',
                    '_Fracture`Stages',
                    'final_timediff',
                    'lic_timediff',
                    'rrd_timediff',
                    'st_timediff',
                    'timediff',
                    'SurfAbandonDate']

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
                    'st_timediff','timediff',
                    'SurfAbandonDate']
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
                      'st_timediff',
                      'timediff',
                      'SurfAbandonDate'
                      ]

exclude_cols_dict = {'Oil_norm': exclude_cols_oil,
                     'Gas_norm': exclude_cols_gas,
                     'Water_norm': exclude_cols_water}

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

threshold_dict = {}

class LogRF(RandomForestRegressor):
    def __init__(self, target=None, **kwargs):
        super().__init__(**kwargs)
        if target == "Oil_norm":
            self.target_scaler = PowerTransformer(method='box-cox', standardize=False)
        elif target == 'Gas_norm':
            self.target_scaler = FunctionTransformer(func=np.log1p,inverse_func=np.expm1)
        elif target == 'Water_norm':
            self.target_scaler = FunctionTransformer(func=np.log1p,inverse_func=np.expm1)


    def fit(self, X, Y, **kwargs):
        #y_train = np.log1p(Y)
        self.target_scaler.fit(Y.values.reshape(-1,1)+1)
        y_train = pd.Series(self.target_scaler.transform(Y.values.reshape(-1,1)+1).reshape(-1,))
        super(RandomForestRegressor, self).fit(X, y_train, **kwargs)

        return self

    def predict(self, X):
        preds = super(RandomForestRegressor, self).predict(X).reshape(-1, 1)
        preds = self.target_scaler.inverse_transform(preds)-1
        return preds[:,0]

condition_dict = {'Oil_norm':[0,1e5],'Gas_norm':[0.02,1e5],'Water_norm':[0.01,1e5]}

def main(input_file_path, output_file_path, tgt="Oil_norm", interim_file_path=None, n_splits=11):


    condition = condition_dict[tgt]


    input_file_name = os.path.join(input_file_path, "Train_final.pck")
    input_file_name_test = os.path.join(input_file_path, "Test_final.pck")
    input_file_name_val = os.path.join(input_file_path, "Validation_final.pck")
    exclude_cols = exclude_cols_dict.get(tgt)
    output_file_name = os.path.join(output_file_path, f"models_lgbm_{tgt}.pck")

    df = pd.read_pickle(input_file_name).drop(exclude_cols, axis=1)
    df_test = pd.read_pickle(input_file_name_test)
    df_val = pd.read_pickle(input_file_name_val).drop(exclude_cols, axis=1)
    df_all = pd.concat([df, df_val], axis=0)


    ids = df_test["EPAssetsId"].copy()

    ids_uwi = df_test["UWI"].copy()

    df_test = df_test.drop(exclude_cols, axis=1)

    cv = KFold(n_splits=n_splits, shuffle=False)
    models = []
    scores = []
    scores_dm = []

    y = df_all.loc[~df_all[tgt].isna(), tgt]
    id_X =df_all.loc[~df_all[tgt].isna(),["EPAssetsId"]]

    X = df_all.loc[~df_all[tgt].isna(), :].drop(
        ["Oil_norm", "Gas_norm", "Water_norm", "EPAssetsId", "_Normalized`IP`BOE/d"],
        axis=1,
    )
    # Filter large vals
    # condition = y < threshold_dict[tgt]
    # X = X.loc[condition,:]
    # y = y.loc[condition]

    X_test = df_test.copy().drop("EPAssetsId", axis=1)

    preds_test = np.zeros((n_splits, df_test.shape[0]))
    preds_holdout = []
    y_true = []
    id_list=[]

    np.random.seed(123)

    best_params = pd.read_csv(os.path.join(output_file_path, f'LGBM_{tgt}_feats_final_Trials.csv')).head(20)
    datasets = {}
    for k, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_holdout = X.iloc[train_index, :], X.iloc[test_index, :]
        print(X_train.shape)
        id_X_holdout = id_X.iloc[test_index]


        # model = LGBMRegressor(num_leaves=16, learning_rate=0.1, n_estimators=300, reg_lambda=30, reg_alpha=30,
        # objective='mae',random_state=123)
        if tgt == 'Oil_norm':
            params = best_params.iloc[k, :].to_dict()
        else:
            params = best_params.iloc[0, :].to_dict()

        model = LogRF(target=tgt,
        max_depth=17,
        n_estimators=200
        )
        y_train, y_holdout = y.iloc[train_index], y.iloc[test_index]
        idx = (y_train > condition[0]) & (y_train < condition[1])
        X_train, y_train = X_train.loc[idx,:],y_train.loc[idx]
        # Calculate a fill value:
        #target_log_mean = np.median(np.log(y_train[y_train > 0]))
        #target_fill_val = np.exp(target_log_mean)
        target_fill_val = 0
        y_train = y_train.fillna(value=target_fill_val)
        logging.info(f'Filling {tgt} with {target_fill_val}')

        y_holdout = y_holdout.fillna(value=0)
        geom_mean = gmean(y_train)
        dm = DummyRegressor(strategy="constant", constant=geom_mean)
        X_train = X_train.fillna(-999)
        X_holdout = X_holdout.fillna(-999)
        X_test = X_test.fillna(-999)

        model.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        dm.fit(X_train, y_train)
        y_pred_holdout = model.predict(X_holdout)
        score = mean_absolute_error(y_holdout, y_pred_holdout)
        score_dm = mean_absolute_error(y_holdout, dm.predict(X_holdout))

        # logging.info(f' Score = {score}')
        models.append(model)
        scores.append(score)

        scores_dm.append((score_dm))
        logger.warning(f"Holdout score = {score}")
        preds_test[k, :] = model.predict(X_test)
        preds_holdout.append(y_pred_holdout.reshape(1, -1))
        y_true.append(y_holdout.values.reshape(1, -1))
        id_list.append(id_X_holdout.values.reshape(1,-1))

        print(mean_absolute_error(y_holdout.values.reshape(1, -1), y_pred_holdout.reshape(1, -1)))

    with open(output_file_name, "wb") as f:
        pickle.dump(models, f)
    logger.info(scores)
    logger.info(f"Mean scores LGBM = {np.mean(scores)}")
    logger.info(f"Mean scores Dummy = {np.mean(scores_dm)}")

    # preds_df = pd.DataFrame(
    #     {"EPAssetsID": ids, "UWI": ids_uwi, tgt: preds_test.mean(axis=0)}
    # )
    preds_df = pd.DataFrame(
        {"EPAssetsID": ids, "UWI": ids_uwi, tgt: mean_log(preds_test)}
    )
    n_points = np.hstack(y_true).shape[0]
    preds_df_val = pd.DataFrame({tgt: np.hstack(preds_holdout)[0, :], f"gt_{tgt}": np.hstack(y_true)[0, :],
                                 'EPAssetsId': np.hstack(id_list)[0, :]})
    logger.warning(f"Final scores on holdout: {np.mean(scores)} +- {np.std(scores)}")
    logger.warning(f"Final scores on full holdout: {mean_absolute_error(preds_df_val[f'gt_{tgt}'], preds_df_val[tgt])}")

    print(eli5.format_as_dataframe(eli5.explain_weights(model, top=60)))

    return preds_df, preds_df_val, np.mean(scores)


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "final")
    output_file_path = os.path.join(project_dir, "models")
    interim_file_path = os.path.join(project_dir, "data", "interim")
    os.makedirs(interim_file_path, exist_ok=True)
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    preds_gas, preds_val_gas, score_holdout_gas = main(
        input_file_path, output_file_path, "Gas_norm", interim_file_path
    )

    preds_oil, preds_val_oil, score_holdout_oil = main(
        input_file_path, output_file_path, "Oil_norm", interim_file_path
    )

    preds_water, preds_val_water, score_holdout_water = main(
        input_file_path, output_file_path, "Water_norm", interim_file_path
    )

    df_oof = pd.concat([preds_val_oil, preds_val_gas, preds_val_water], axis=1)
    df_oof.to_pickle(os.path.join(input_file_path, "RF_OOF.pck"))

    logger.warning(
        f"Scores are: oil {score_holdout_oil}, gas {score_holdout_gas} water {score_holdout_water}"
    )
    df = preds_oil
    df_merge_list = [preds_gas, preds_water]
    for x in df_merge_list:
        df = pd.merge(df, x.drop("UWI", axis=1), on="EPAssetsID")
    submission = df.sort_values("EPAssetsID")
    submission.to_pickle(os.path.join(input_file_path, "submission_rf.pck"))

    submission.columns = [
        "EPAssetsId",
        "UWI",
        "_Normalized`IP`(Oil`-`Bbls)",
        "_Normalized`IP`Gas`(Boe/d)",
        "_Normalized`IP`(Water`-`Bbls)",
    ]
    submission.to_csv(os.path.join(input_file_path, "submission_rf.txt"), index=False)
# EPAssetsID,UWI,Oil_norm,Gas_norm,Water_norm
