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
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

project_dir = Path(__file__).resolve().parents[2]


# exclude_cols = ['FinalDrillDate', 'RigReleaseDate', 'SpudDate','UWI']
exclude_cols = ["FinalDrillDate", "RigReleaseDate", "SpudDate", "UWI"]
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

class LogLGBM(LGBMRegressor):
    def fit(self, X, Y, **kwargs):
        y_train = np.log(Y)
        super(LogLGBM, self).fit(X, y_train, **kwargs)

        return self

    def predict(self, X):
        preds = super(LogLGBM, self).predict(X)
        preds = np.exp(preds)
        return preds


def main(input_file_path, output_file_path, tgt="Oil_norm", n_splits=5):
    input_file_name = os.path.join(input_file_path, "Train_final.pck")
    input_file_name_test = os.path.join(input_file_path, "Test_final.pck")
    input_file_name_val = os.path.join(input_file_path, "Validation_final.pck")

    output_file_name = os.path.join(output_file_path, f"models_lgbm_{tgt}.pck")

    df = pd.read_pickle(input_file_name).drop(exclude_cols, axis=1)
    df_test = pd.read_pickle(input_file_name_test)
    df_val = pd.read_pickle(input_file_name_val).drop(exclude_cols, axis=1)
    df_all = pd.concat([df, df_val], axis=0)

    ids = df_test["EPAssetsId"]

    ids_uwi = df_test["UWI"]

    df_test = df_test.drop(exclude_cols, axis=1)

    cv = KFold(n_splits=n_splits, shuffle=True)
    models = []
    scores = []
    scores_dm = []

    y = df_all.loc[~df_all[tgt].isna(), tgt]
    X = df_all.loc[~df_all[tgt].isna(), :].drop(
        ["Oil_norm", "Gas_norm", "Water_norm", "EPAssetsId", "_Normalized`IP`BOE/d"],
        axis=1,
    )
    X_test = df_test.copy().drop("EPAssetsId", axis=1)

    preds_test = np.zeros((n_splits, df_test.shape[0]))
    preds_holdout = np.zeros((n_splits, X.shape[0]))
    np.random.seed(123)
    fr = np.random.uniform(low=-0.1,high=0.1,size = n_splits)
    lam = np.random.uniform(low=-5, high=5, size=n_splits)
    alpha = np.random.uniform(low=1, high=5, size=n_splits)
    leaves = np.random.uniform(low=-15, high=10, size=n_splits).astype(int)


    for k, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_holdout = X.iloc[train_index, :], X.iloc[test_index, :]

        # model = LGBMRegressor(num_leaves=16, learning_rate=0.1, n_estimators=300, reg_lambda=30, reg_alpha=30,
        # objective='mae',random_state=123)

        model = LogLGBM(
            num_leaves=32 + leaves[k],
            bagging_fraction=0.5 + fr[k],
            learning_rate=0.04,
            n_estimators=3500,
            reg_lambda=15 + lam[k],
            reg_alpha=5 + alpha[k],
            objective="mae",
            random_state=123,
            feature_fraction=0.7,
        )
        y_train, y_holdout = y.iloc[train_index], y.iloc[test_index]
        geom_mean = gmean(y_train)
        dm = DummyRegressor(strategy="constant", constant=geom_mean)

        model.fit(X_train, y_train, categorical_feature=CAT_COLUMNS,
                  eval_set=(X_holdout, y_holdout),
                  early_stopping_rounds=50,
                  verbose=200
                  )
        # model.fit(X_train, y_train)
        dm.fit(X_train, y_train)

        score = mean_absolute_error(y_holdout, model.predict(X_holdout))
        score_dm = mean_absolute_error(y_holdout, dm.predict(X_holdout))

        # logging.info(f' Score = {score}')
        models.append(model)
        scores.append(score)

        scores_dm.append((score_dm))
        logger.warning(f"Holdout score = {score}")
        preds_test[k, :] = model.predict(X_test).reshape(1, -1)
        preds_holdout[k, test_index] = model.predict(X_holdout).reshape(1, -1)

    with open(output_file_name, "wb") as f:
        pickle.dump(models, f)
    logger.info(scores)
    logger.info(f"Mean scores LGBM = {np.mean(scores)}")
    logger.info(f"Mean scores Dummy = {np.mean(scores_dm)}")

    preds_df = pd.DataFrame(
        {"EPAssetsID": ids, "UWI": ids_uwi, tgt: preds_test.mean(axis=0)}
    )
    preds_df_val = pd.DataFrame({tgt: preds_holdout.mean(axis=0), "gt": y})
    logger.warning(f"Final scores on holdout: {np.mean(scores)} +- {np.std(scores)}")
    print(eli5.format_as_dataframe(eli5.explain_weights(model)))

    return preds_df, np.mean(scores)


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "final")
    output_file_path = os.path.join(project_dir, "models")
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    preds_oil, score_holdout_oil = main(
        input_file_path, output_file_path, tgt="Oil_norm"
    )
    preds_gas, score_holdout_gas = main(
        input_file_path, output_file_path, tgt="Gas_norm"
    )
    preds_water, score_holdout_water = main(
        input_file_path, output_file_path, tgt="Water_norm"
    )
    logger.warning(
        f"Scores are: oil {score_holdout_oil}, gas {score_holdout_gas} water {score_holdout_water}"
    )
    df = preds_oil
    df_merge_list = [preds_gas, preds_water]
    for x in df_merge_list:
        df = pd.merge(df, x.drop("UWI", axis=1), on="EPAssetsID")
    submission = df.sort_values("EPAssetsID")
    submission.columns = [
        "EPAssetsId",
        "UWI",
        "_Normalized`IP`(Oil`-`Bbls)",
        "_Normalized`IP`Gas`(Boe/d)",
        "_Normalized`IP`(Water`-`Bbls)",
    ]
    submission.to_csv(os.path.join(input_file_path, "submission_lgbm.txt"), index=False)
# EPAssetsID,UWI,Oil_norm,Gas_norm,Water_norm
