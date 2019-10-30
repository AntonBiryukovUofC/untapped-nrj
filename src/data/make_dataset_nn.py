# -*- coding: utf-8 -*-
import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
import category_encoders as ce

import pandas_profiling as pdp

pd.set_option("display.width", 1800)
pd.set_option("display.max_columns", 10)

# fix the seed for reproducibility:
random.seed(123)
np.random.seed(123)
COLS_TO_KEEP = (
    "EPAssetsId,UWI,CurrentOperator,"
    "WellType,"
    "Formation,Field,Pool,"
    "Surf_Longitude,Surf_Latitude,BH_Longitude,BH_Latitude,"
    "GroundElevation,KBElevation,TotalDepth,LaheeClass,"
    "DrillingContractor,SpudDate,RigReleaseDate,DaysDrilling,DrillMetresPerDay,TVD,"
    "WellProfile,ProjectedDepth,"
    "_Max`Prod`(BOE),"
    "_Fracture`Stages,"
    "Confidential,SurfaceOwner,_Open`Hole,CompletionDate,Agent,ConfidentialReleaseDate,StatusDate,SurfAbandonDate,FinalDrillDate,"
    "Licensee,StatusSource,CurrentOperatorParent,LicenceDate,Municipality,OSArea,OSDeposit,UnitName,_Completion`Events"
)
CAT_COLUMNS = [
    "CurrentOperator",
    "WellType",
    "Formation",
    "Field",
    "Pool",
    "LaheeClass",
    "DrillingContractor",
    "WellProfile",
    "Confidential",
    "SurfaceOwner",
    "_Open`Hole",
    "Agent",
    "Licensee",
    "StatusSource",
    "CurrentOperatorParent",
    "Municipality", "_Fracture`Stages",
    "OSDeposit",
    "OSArea",
    "UnitName",
]
DATE_COLUMNS = [
    "ConfidentialReleaseDate",
    "SurfAbandonDate",
    "SpudDate",
    "StatusDate",
    "LicenceDate",
    "FinalDrillDate",
    "RigReleaseDate",
]  # DATE_COLUMNS = []
COUNT_COLUMNS = []

project_dir = Path(__file__).resolve().parents[2]
cols = COLS_TO_KEEP.split(",")
all_cols = CAT_COLUMNS + COUNT_COLUMNS + DATE_COLUMNS

in_cols = [c in cols for c in all_cols]
if not (all(in_cols)):
    logging.error(in_cols)
    logging.error(all_cols[in_cols.index(False)])

    raise ValueError("Check your categorical columns and cols to keep!")


def read_table(input_file_path, logger, output_file_path, suffix):
    output_filepath_df = os.path.join(output_file_path, f"{suffix}_df_nn.pck")
    output_filepath_misc = os.path.join(output_file_path, f"{suffix}_misc.pck")
    logger.info(f"making final data set from raw data {suffix}")
    feature_df = pd.read_csv(
        os.path.join(input_file_path, f"Header - {suffix.lower()}.txt")
    )
    test_wells = pd.read_csv(
        os.path.join(input_file_path, "regression_sample_submission_test.txt")
    )["EPAssetsId"]
    cols = COLS_TO_KEEP.split(",")

    if suffix == "Test":
        inds = feature_df["EPAssetsId"].isin(test_wells)
        df_full = feature_df.loc[inds, cols]
        df_full["HZLength"] = df_full["TotalDepth"] - df_full["TVD"]

    # report = pdp.ProfileReport(df_full)
    # report.to_file(os.path.join(output_file_path, f"{suffix}_profile.html"))

    if suffix in ["Train", "Validation"]:
        target_df = pd.read_csv(
            os.path.join(input_file_path, f"Viking - {suffix}.txt")
        ).drop("UWI", axis=1)
        target_df.rename(
            columns={
                "_Normalized`IP`(Oil`-`Bbls)": "Oil_norm",
                "_Normalized`IP`Gas`(Boe/d)": "Gas_norm",
                "_Normalized`IP`(Water`-`Bbls)": "Water_norm",
            },
            inplace=True,
        )
        df_full = pd.merge(feature_df, target_df, on="EPAssetsId")

        # report = pdp.ProfileReport(df_full)
        # report.to_file(os.path.join(output_file_path, f"{suffix}_profile.html"))

        df_full = df_full[cols + target_df.drop("EPAssetsId", axis=1).columns.tolist()]
        df_full["HZLength"] = df_full["TotalDepth"] - df_full["TVD"]

        l = list(set(feature_df["EPAssetsId"]) & set(target_df["EPAssetsId"]))
        logger.info(f"intersection len: {len(l)}")

    # Clip to max of 35 days of drilling (?)
    # df_full["DaysDrilling"] = np.clip(df_full["DaysDrilling"], a_min=None, a_max=35)

    logger.info(f"Shape feature = {df_full.shape} {suffix}")
    for c in DATE_COLUMNS:
        logger.info(f"to DT: {c}")
        df_full[c] = pd.to_datetime(df_full[c])
        df_full[c] = (df_full[c] - pd.to_datetime("1970-01-01")).dt.total_seconds() / 1000

    logger.info(f"Shape full = {df_full.shape} {suffix}")
    return df_full, output_filepath_df, output_filepath_misc


def preprocess_table(input_file_path, output_file_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    encoders = {}
    logger = logging.getLogger(__name__)

    df_full_train, output_filepath_df_train, output_filepath_misc_train = read_table(
        input_file_path, logger, output_file_path, suffix="Train"
    )

    df_full_test, output_filepath_df_test, output_filepath_misc_test = read_table(
        input_file_path, logger, output_file_path, suffix="Test"
    )

    df_full_val, output_filepath_df_val, output_filepath_misc_val = read_table(
        input_file_path, logger, output_file_path, suffix="Validation"
    )

    # Label encode categoricals
    for cat in CAT_COLUMNS:
        logger.info(f"to category: {cat}")
        df_full_train[cat] = df_full_train[cat].astype(str)
        df_full_test[cat] = df_full_test[cat].astype(str)
        df_full_val[cat] = df_full_val[cat].astype(str)

    CALC_COUNT_COLUMNS = []
    # Assign clusters:
    ncomp = 23
    cl = GaussianMixture(n_components=ncomp).fit(
        df_full_train[["Surf_Longitude", "Surf_Latitude"]]
    )

    # Assign cluster probs:
    cols_probs = [f"GM{i}" for i in range(ncomp)]
    df_full_train[cols_probs] = pd.DataFrame(
        data=cl.predict_proba(df_full_train[["Surf_Longitude", "Surf_Latitude"]]),
        columns=cols_probs,
        index=df_full_train.index,
    )
    df_full_test[cols_probs] = pd.DataFrame(
        data=cl.predict_proba(df_full_test[["Surf_Longitude", "Surf_Latitude"]]),
        columns=cols_probs,
        index=df_full_test.index,
    )
    df_full_val[cols_probs] = pd.DataFrame(
        data=cl.predict_proba(df_full_val[["Surf_Longitude", "Surf_Latitude"]]),
        columns=cols_probs,
        index=df_full_val.index,
    )

    df_full_train["cluster"] = cl.predict(
        df_full_train[["Surf_Longitude", "Surf_Latitude"]]
    )
    df_full_test["cluster"] = cl.predict(
        df_full_test[["Surf_Longitude", "Surf_Latitude"]]
    )
    df_full_val["cluster"] = cl.predict(
        df_full_val[["Surf_Longitude", "Surf_Latitude"]]
    )

    # Standard-scale numerical columns:


    df_to_fit_le = pd.concat([df_full_train, df_full_val], axis=0)[df_full_test.columns]
    cols_numerical =df_to_fit_le.columns.difference(CAT_COLUMNS+COUNT_COLUMNS+['cluster','UWI','EPAssetsId'])
    scaler = StandardScaler()
    scaler.fit(df_to_fit_le[cols_numerical])
    df_to_fit_le[cols_numerical] =scaler.transform(df_to_fit_le[cols_numerical])



    # Label encode categoricals
    ohe_encoder = ce.OneHotEncoder(
        return_df=True, cols=CAT_COLUMNS + COUNT_COLUMNS + ["cluster"], verbose=1
    )

    # Encode train and test with LE
    ohe_encoder.fit(df_to_fit_le)
    df_full_train_ohe = ohe_encoder.transform(
        df_full_train[df_full_test.columns]
    )
    df_full_test_ohe = ohe_encoder.transform(df_full_test)
    df_full_val_ohe = ohe_encoder.transform(
        df_full_val[df_full_test.columns]
    )
    # Attach OHE to originals:
    df_full_train = pd.concat([df_full_train[['UWI','EPAssetsId','Oil_norm','Gas_norm','Water_norm']],df_full_train_ohe ],axis=1)
    df_full_val = pd.concat([df_full_val[['UWI','EPAssetsId','Oil_norm','Gas_norm','Water_norm']], df_full_val_ohe], axis=1)
    df_full_test = pd.concat([df_full_test[['UWI','EPAssetsId']], df_full_test_ohe],
                            axis=1)

    # Clip values:

    for c in ["GroundElevation", "TotalDepth", "ProjectedDepth", "_Max`Prod`(BOE)"]:
        l, u = (
            np.nanquantile(df_full_train[c], 0.01),
            np.nanquantile(df_full_train[c], 0.98),
        )

        df_full_train[c] = np.clip(df_full_train[c], l, u)
        df_full_test[c] = np.clip(df_full_test[c], l, u)
        df_full_val[c] = np.clip(df_full_val[c], l, u)

        logging.info(f"Clipped {c} at {l} -- {u}")

    #
    print(df_full_train.shape)
    print(df_full_test.shape)
    print(df_full_val.shape)
    # Encode test:

    misc = {}
    misc["encoder_dict"] = encoders
    # profile = feature_df.profile_report(title=f'Pandas Profiling Report for {suffix}')
    # profile.to_file(output_file=os.path.join(project_dir, f"output_{suffix}.html"))


    df_full_train.to_pickle(output_filepath_df_train)
    df_full_test.to_pickle(output_filepath_df_test)
    df_full_val.to_pickle(output_filepath_df_val)

    with open(output_filepath_misc_train, "wb") as f:
        pickle.dump(misc, f)

    return 0


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_filepath = os.path.join(project_dir, "data", "raw")
    output_filepath = os.path.join(project_dir, "data", "processed")
    preprocess_table(input_filepath, output_filepath)
