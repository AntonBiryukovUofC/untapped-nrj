# -*- coding: utf-8 -*-
import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ["Unknown"])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = [
                    "Unknown" if x == unique_item else x for x in new_data_list
                ]

        return self.label_encoder.transform(new_data_list)


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
    "DrillingContractor,SpudDate,FinalDrillDate,RigReleaseDate,DaysDrilling,DrillMetresPerDay,TVD,"
    "WellProfile,ProjectedDepth,"
    "_Max`Prod`(BOE),"
    "_Fracture`Stages,"
    "Confidential,SurfaceOwner,_Open`Hole,CompletionDate,Agent,ConfidentialReleaseDate,StatusDate,SurfAbandonDate,"
    "Licensee,LicenceNumber,StatusSource,CurrentOperatorParent,LicenseDate,Municipality,OSArea,OSDeposit,"
    "PSACAreaCode,UnitName,_Completion`Events"
)
CAT_COLUMNS = [
    "CurrentOperator",
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
    "Municipality",
    "CurrentOperatorParent"
]
DATE_COLUMNS = [
    "ConfidentialReleaseDate",
    "SurfAbandonDate",
    "SpudDate",
    "StatusDate",
    "LicenseDate",
    "CompletionDate",
    "FinalDrillDate",
    "RigReleaseDate",
]# DATE_COLUMNS = []
COUNT_COLUMNS = ["LicenceNumber", "OSDeposit", "OSArea", "PSACAreaCode", "UnitName"]

project_dir = Path(__file__).resolve().parents[2]
cols = COLS_TO_KEEP.split(",")
all_cols = CAT_COLUMNS + COUNT_COLUMNS + DATE_COLUMNS

in_cols = [c in cols for c in all_cols]
if not (all(in_cols)):
    logging.error(in_cols)
    logging.error(all_cols[in_cols.index(False)])

    raise ValueError("Check your categorical columns and cols to keep!")

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
    df_full["DaysDrilling"] = np.clip(df_full["DaysDrilling"], a_min=None, a_max=35)
    logger.info(f"Shape feature = {df_full.shape} {suffix}")
    for c in DATE_COLUMNS:
        logger.info(f"to DT: {c}")
        df_full[c] = pd.to_datetime(df_full[c])

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

    # Label encode categoricals
    label_encoder = ce.OrdinalEncoder(return_df=True, cols=CAT_COLUMNS, verbose=1)
    count_encoder = ce.CountEncoder(
        return_df=True, cols=COUNT_COLUMNS, verbose=1, handle_unknown=999
    )

    df_to_fit_le = pd.concat([df_full_train, df_full_val], axis=0)[df_full_test.columns]
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
