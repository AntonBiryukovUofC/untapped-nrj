# -*- coding: utf-8 -*-
import logging
import os
import pickle
import random
from pathlib import Path
import tqdm
import numpy as np
import pandas as pd
import utm
from category_encoders import OrdinalEncoder
from sklearn.neighbors.ball_tree import BallTree
from sklearn.preprocessing import LabelEncoder
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
    "Licensee,LicenceNumber,StatusSource,CurrentOperatorParent,LicenceDate,Municipality,OSArea,OSDeposit,UnitName"
)
CAT_COLUMNS = [
    "CurrentOperator",
    "CurrentOperatorParent",
    "Licensee",
    "WellType",
    "Formation",
    "Field",
    "Pool",
    "LaheeClass",
    "DrillingContractor",
    "WellProfile",
    "_Fracture`Stages",
    "_Open`Hole",
    "Confidential",
    "SurfaceOwner",
    "Agent",
    "StatusSource",
    "Municipality"
]
DATE_COLUMNS = [
    "ConfidentialReleaseDate",
    "SurfAbandonDate",
    "SpudDate",
    "StatusDate",
    "LicenceDate",
    "FinalDrillDate",
    "RigReleaseDate",
]# DATE_COLUMNS = []
COUNT_COLUMNS = ["OSDeposit", "OSArea", "UnitName"]

project_dir = Path(__file__).resolve().parents[2]
cols = COLS_TO_KEEP.split(",")
all_cols = CAT_COLUMNS + COUNT_COLUMNS + DATE_COLUMNS

in_cols = [c in cols for c in all_cols]
if not (all(in_cols)):
    logging.error(in_cols)
    logging.error(all_cols[in_cols.index(False)])

    raise ValueError("Check your categorical columns and cols to keep!")


def rule(row):
    lat, long, _, _ = utm.from_latlon(row["Surf_Latitude"], row["Surf_Longitude"])
    return pd.Series({"lat": lat, "long": long})


def read_table(input_file_path, logger, output_file_path, suffix):
    output_filepath_df = os.path.join(output_file_path, f"{suffix}_df.pck")
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
        feature_df = feature_df.loc[inds, cols]

        df_full = feature_df

    if suffix in ["Train", "Validation"]:
        feature_df = feature_df.loc[:, cols]

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
        l = list(set(feature_df["EPAssetsId"]) & set(target_df["EPAssetsId"]))
        logger.info(f"intersection len: {len(l)}")

    df_full["HZLength"] = df_full["TotalDepth"] - df_full["TVD"]
    # Clip to max of 35 days of drilling (?)
    #df_full["DaysDrilling"] = np.clip(df_full["DaysDrilling"], a_min=None, a_max=35)

    logger.info(f"Shape feature = {df_full.shape} {suffix}")
    for c in DATE_COLUMNS:
        logger.info(f"to DT: {c}")
        df_full[c] = pd.to_datetime(df_full[c])

    # Convert coordinates to UTM

    df_full = df_full.merge(df_full.apply(rule, axis=1), left_index=True, right_index=True)
    df_full['lat'] = df_full['lat'] - 2.9e5
    df_full['long'] = df_full['long'] - 5.55e6

    logger.info(f"Shape full = {df_full.shape} {suffix}")
    return df_full, output_filepath_df, output_filepath_misc


def calculate_agg_statistics(tree, X, df,radius=3000):
    mean_boe=0
    mean_water=0
    mean_gas=0
    mean_oil=0
    std_boe=0
    std_water=0
    std_gas=0
    std_oil=0
    nwells=0
    df_list=[]
    for i  in tqdm.trange(df.shape[0]):
        row=df.iloc[i,:]
        inds = tree.query_radius(row[['lat','long']].values.reshape(1,-1),r=radius)[0]
        data=X.iloc[inds,:]
        mean_boe = np.nanmean(data['_Max`Prod`(BOE)'])
        mean_water = np.nanmean(data['Water_norm'])
        mean_gas = np.nanmean(data['Gas_norm'])
        mean_oil = np.nanmean(data['Oil_norm'])
        std_boe = np.nanstd(data['_Max`Prod`(BOE)'])
        std_water = np.nanstd(data['Water_norm'])
        std_gas = np.nanstd(data['Gas_norm'])
        std_oil = np.nanstd(data['Oil_norm'])
        nwells = data.shape[0]
        tmp = pd.DataFrame({'mean_boe':mean_boe,'mean_water':mean_water,'mean_gas':mean_gas,'mean_oil':mean_oil,
                            "std_boe":std_boe,"std_water":std_water,"std_gas":std_gas,'std_oil':std_oil,'n_wells':nwells},index=[i])
        df_list.append(tmp)
    result = pd.concat(df_list,axis=0)
    result.index=df.index

    result = pd.concat([result,df],axis=1)
    return result


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
    df_to_fit_le = pd.concat([df_full_train, df_full_val], axis=0)[df_full_test.columns]

    # Label encode categoricals
    label_encoder = ce.CountEncoder(return_df=True, cols=CAT_COLUMNS, verbose=1, normalize=True)
    count_encoder = ce.CountEncoder(
        return_df=True, cols=COUNT_COLUMNS + CALC_COUNT_COLUMNS, verbose=1, normalize=True
    )

    # Encode train and test with LE
    label_encoder.fit(df_to_fit_le)
    df_full_train[df_full_test.columns] = label_encoder.transform(
        df_full_train[df_full_test.columns]
    )
    df_full_test = label_encoder.transform(df_full_test)
    df_full_val[df_full_test.columns] = label_encoder.transform(
        df_full_val[df_full_test.columns]
    )
    # Encode train and test with CE
    count_encoder.fit(df_to_fit_le)
    df_full_train[df_full_test.columns] = count_encoder.transform(
        df_full_train[df_full_test.columns]
    )
    df_full_test = count_encoder.transform(df_full_test)
    df_full_val[df_full_test.columns] = count_encoder.transform(
        df_full_val[df_full_test.columns]
    )
    # Encode aggregate statistics using BallTree:
    X = pd.concat([df_full_train[['lat', 'long']], df_full_val[['lat', 'long']]], axis=0).values
    # Build a tree:
    tree = BallTree(X)
    # Calculate aggregate statistics using tree:
    X_to_get_data = pd.concat([df_full_train, df_full_val], axis=0)
    #
    # df_full_train = calculate_agg_statistics(tree,X_to_get_data,df_full_train)
    # df_full_val = calculate_agg_statistics(tree, X_to_get_data, df_full_val)
    # df_full_test = calculate_agg_statistics(tree, X_to_get_data, df_full_test)



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
