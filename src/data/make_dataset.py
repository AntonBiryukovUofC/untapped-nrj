# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import random
import numpy as np
import pandas_profiling as pp
import pickle
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.width', 1800)
pd.set_option('display.max_columns',10)


# fix the seed for reproducibility:
random.seed(123)
np.random.seed(123)
COLS_TO_KEEP = "EPAssetsId,CurrentOperator," \
           "LicenseeID,WellTypeStandardised," \
           "Formation,Field,Pool,SurveySystem," \
           "Surf_Longitude,Surf_Latitude," \
           "BH_Longitude,BH_Latitude," \
           "GroundElevation,KBElevation,TotalDepth,LaheeClass,OSArea," \
           "OSDeposit,DrillingContractor,SpudDate,FinalDrillDate,RigReleaseDate,DaysDrilling,DrillMetresPerDay,TVD," \
           "WellProfile,ProjectedDepth," \
           "_Max`Prod`(BOE)," \
           "_Fracture`Stages,_Completion`Events"
CAT_COLUMNS =['LicenseeID','CurrentOperator','WellTypeStandardised','Formation','Field','Pool','SurveySystem',
              'LaheeClass','OSArea','OSDeposit','DrillingContractor','WellProfile']
DATE_COLUMNS = ['FinalDrillDate','RigReleaseDate','SpudDate']
project_dir = Path(__file__).resolve().parents[2]




def main(input_filepath, output_filepath,suffix = 'Train'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    encoders = {}
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    target_df = pd.read_csv(os.path.join(input_filepath,f'Viking - {suffix}.txt'),nrows=20)
    feature_df = pd.read_csv(os.path.join(input_filepath, f'Header - {suffix.lower()}.txt'))
    output_filepath_df = os.path.join(output_filepath,f'{suffix}_df.pck')
    output_filepath_misc = os.path.join(output_filepath, f'{suffix}_misc.pck')

    # subset cols:
    cols =COLS_TO_KEEP.split(',')
    feature_df = feature_df.loc[:, cols]
    feature_df['HZLength'] = feature_df['TotalDepth'] - feature_df['TVD']
    # Clip to max of 35 days of drilling (?)
    feature_df['DaysDrilling'] = np.clip(feature_df['DaysDrilling'],a_min=None,a_max=35)
    for c in DATE_COLUMNS:
        logger.info(f'to DT: {c}')
        feature_df[c] = pd.to_datetime(feature_df[c])
    for cat in CAT_COLUMNS:
        logger.info(f'to category: {cat}')
        label_encoder = LabelEncoder()
        feature_df[cat] = label_encoder.fit_transform(feature_df[cat].astype(str))
        encoders[cat] = label_encoder


    misc = {}
    misc['encoder_dict'] = encoders
    #profile = feature_df.profile_report(title=f'Pandas Profiling Report for {suffix}')
    #profile.to_file(output_file=os.path.join(project_dir, f"output_{suffix}.html"))

    feature_df.to_pickle(output_filepath_df)
    with open(output_filepath_misc,'wb') as f:
        pickle.dump(misc,f)

    return 0


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_filepath = os.path.join(project_dir, 'data', 'raw')
    output_filepath = os.path.join(project_dir, 'data', 'processed')
    df = main(input_filepath, output_filepath)
