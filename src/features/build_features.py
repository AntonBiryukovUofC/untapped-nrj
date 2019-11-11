import logging
import os
import pickle
import random
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.make_dataset import DATE_COLUMNS, CAT_COLUMNS
import utm
project_dir = Path(__file__).resolve().parents[2]


def rule(row):
    lat, long,_,_= utm.from_latlon(row["Surf_Latitude"], row["Surf_Longitude"], 45, 'K')
    return pd.Series({"lat": lat, "long": long})

def distance(s_lat, s_lng, e_lat, e_lng):
    # approximate radius of earth in km
    R = 6373.0

    s_lat = s_lat * np.pi / 180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat) / 2) ** 2 + np.cos(s_lat) * np.cos(e_lat) * np.sin((e_lng - s_lng) / 2) ** 2

    return 2 * R * np.arcsin(np.sqrt(d))


def build_features(input_file_path, output_file_path, suffix="Train"):
    input_filename = os.path.join(input_file_path, f"{suffix}_df.pck")
    output_file_name = os.path.join(output_file_path, f"{suffix}_final.pck")
    df = pd.read_pickle(input_filename)
    #df.loc[df["Surf_Longitude"] > -70, "Surf_Longitude"] = np.nan
    # for col in ['RigReleaseDate','SpudDate']:
    #     df[f'{col}_month']=df[col].dt.month
    #     df[f'{col}_year'] = df[col].dt.year
    #     df[f'{col}_day'] = df[col].dt.day
    # df.loc[df["Surf_Longitude"] > -70, "Surf_Longitude"] = np.nan

    df['RigReleaseDate_days_till_monthend'] = 31 - df['RigReleaseDate'].dt.day
    df['FinalDrillDate_days_till_monthend'] = 31 - df['FinalDrillDate'].dt.day

    #df['BOE_average'] = df['_Max`Prod`(BOE)'] / df['RigReleaseDate_days_till_monthend']
    #df['BOE_fd_av'] = df['_Max`Prod`(BOE)'] / df['FinalDrillDate_days_till_monthend']
    df['SpudDate_dt'] = df['SpudDate']
    for col in DATE_COLUMNS:
        df[col] = (df[col] - pd.to_datetime("1950-01-01")).dt.total_seconds()
    # All possible diff interactions of dates:
    # for i in range(len(DATE_COLUMNS)):
    #     for j in range(i + 1, len(DATE_COLUMNS)):
    #         l=DATE_COLUMNS[i]
    #         r=DATE_COLUMNS[j]
    #         df[f'{l}_m_{r}'] = df[l] - df[r]

    df["timediff"] = df["SpudDate"] - df["SurfAbandonDate"]
    df["st_timediff"] = df["SpudDate"] - df["StatusDate"]
    df["cf_timediff"] = df["ConfidentialReleaseDate"] - df["SpudDate"]
    df["lic_timediff"] = df["LicenceDate"] - df["SpudDate"]
    df["final_timediff"] = df["FinalDrillDate"] - df["SpudDate"]
    df["rrd_timediff"] = df["RigReleaseDate"] - df["SpudDate"]

    #df['is_na_completion_date'] = pd.to_datetime(df['CompletionDate']).isna()
    df["LengthDrill"] = df["DaysDrilling"] * df["DrillMetresPerDay"]
    #df["DepthDiff"] = df["ProjectedDepth"]/ df["TVD"]
    #df["DepthDiffLD"] = df["ProjectedDepth"] - df["LengthDrill"]
    #df["TDPD"] = df["ProjectedDepth"] - df["TotalDepth"]
    #df['LicenceNumber_nchar']=df['LicenceNumber'].astype(str).str.count("[a-zA-Z]")
    #df['LicenceNumber_ndig'] = df['LicenceNumber'].astype(str).str.count("[0-9]")

    #df['is_na_BH'] = df['BH_Latitude'].isna() | df['BH_Longitude'].isna()
    df.drop(['OSArea','OSDeposit'], axis=1, inplace=True)

   # # TODO Haversine, azimuth:
    df['haversine_Length'] = distance(df['Surf_Latitude'], df['Surf_Longitude'], df['BH_Latitude'], df['BH_Longitude'])
    #df['azi_proxy'] = np.arctan((df['Surf_Latitude'] - df['BH_Latitude'])/(df['Surf_Longitude'] - df['BH_Longitude']))
    df.drop(['BH_Latitude', 'BH_Longitude','LicenceNumber'], axis=1, inplace=True)
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
