import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.make_dataset import DATE_COLUMNS, CAT_COLUMNS

project_dir = Path(__file__).resolve().parents[2]


def build_features(input_file_path, output_file_path, suffix="Train"):
    input_filename = os.path.join(input_file_path, f"{suffix}_df.pck")
    output_file_name = os.path.join(output_file_path, f"{suffix}_final.pck")

    df = pd.read_pickle(input_filename)
    df.loc[df["Surf_Longitude"] > -70, "Surf_Longitude"] = np.nan

    for col in DATE_COLUMNS:
        df[f'month_{col}'] = df[col].dt.month
        df[f'year_{col}'] = df[col].dt.year
        df[f'day_{col}'] = df[col].dt.day
        df[col] = (df[col] - pd.to_datetime("1970-01-01")).dt.total_seconds()
    df["timediff"] = df["SpudDate"] - df["SurfAbandonDate"]
    df['depth'] = df['TVD'] - df['GroundElevation']
    df['is_op'] = (df['Licensee']==df['CurrentOperator'])
    df["st_timediff"] = df["SpudDate"] - df["StatusDate"]
    df["cf_timediff"] = df["ConfidentialReleaseDate"] - df["SpudDate"]
    df["lic_timediff"] = df["LicenseDate"] - df["SpudDate"]
    df["LengthDrill"] = df["DaysDrilling"] * df["DrillMetresPerDay"]
    df["comp_timediff"] = df["CompletionDate"] - df["SpudDate"]
    df["final_timediff"] = df["FinalDrillDate"] - df["SpudDate"]
    df["rrd_timediff"] = df["RigReleaseDate"] - df["SpudDate"]

    df.to_pickle(output_file_name)
    return df


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "processed")
    output_file_path = os.path.join(project_dir, "data", "final")
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    df_train = build_features(input_file_path, output_file_path, suffix="Train")
    df_test = build_features(input_file_path, output_file_path, suffix="Test")
    df_val = build_features(input_file_path, output_file_path, suffix="Validation")
