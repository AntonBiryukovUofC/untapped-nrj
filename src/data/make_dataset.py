# -*- coding: utf-8 -*-
import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import LabelEncoder
import numpy as np


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
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
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
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

pd.set_option('display.width', 1800)
pd.set_option('display.max_columns', 10)

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
CAT_COLUMNS = ['LicenseeID', 'CurrentOperator', 'WellTypeStandardised', 'Formation', 'Field', 'Pool', 'SurveySystem',
               'LaheeClass', 'OSArea', 'OSDeposit', 'DrillingContractor', 'WellProfile']
DATE_COLUMNS = ['FinalDrillDate', 'RigReleaseDate', 'SpudDate']
project_dir = Path(__file__).resolve().parents[2]


def read_table(input_file_path, logger, output_file_path, suffix):
    output_filepath_df = os.path.join(output_file_path, f'{suffix}_df.pck')
    output_filepath_misc = os.path.join(output_file_path, f'{suffix}_misc.pck')
    logger.info('making final data set from raw data')
    feature_df = pd.read_csv(os.path.join(input_file_path, f'Header - {suffix.lower()}.txt'))
    cols = COLS_TO_KEEP.split(',')
    feature_df = feature_df.loc[:, cols]
    feature_df['HZLength'] = feature_df['TotalDepth'] - feature_df['TVD']
    # Clip to max of 35 days of drilling (?)
    feature_df['DaysDrilling'] = np.clip(feature_df['DaysDrilling'], a_min=None, a_max=35)
    for c in DATE_COLUMNS:
        logger.info(f'to DT: {c}')
        feature_df[c] = pd.to_datetime(feature_df[c])
    if suffix == 'Train':
        target_df = pd.read_csv(os.path.join(input_file_path, f'Viking - {suffix}.txt')).drop('UWI', axis=1)
        target_df.rename(columns={'_Normalized`IP`(Oil`-`Bbls)': 'Oil_norm',
                                  '_Normalized`IP`Gas`(Boe/d)': 'Gas_norm',
                                  '_Normalized`IP`(Water`-`Bbls)': 'Water_norm'}, inplace=True)
        df_full = pd.merge(feature_df, target_df, on='EPAssetsId')
    else:
        df_full = feature_df

    return df_full, output_filepath_df, output_filepath_misc


def preprocess_table(input_file_path, output_file_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    encoders = {}
    logger = logging.getLogger(__name__)

    df_full_train, output_filepath_df_train, output_filepath_misc_train = read_table(
        input_file_path, logger,
        output_file_path, suffix='Train')

    df_full_test, output_filepath_df_test, output_filepath_misc_test = read_table(
        input_file_path, logger,
        output_file_path, suffix='Test')

    # Label encode categoricals
    for cat in CAT_COLUMNS:
        logger.info(f'to category: {cat}')
        df_full_train[cat] = df_full_train[cat].astype(str)
        label_encoder = LabelEncoderExt()
        label_encoder.fit(df_full_train[cat])
        # Encode train and test
        df_full_train[cat] = label_encoder.transform(df_full_train[cat])
        df_full_test[cat] = label_encoder.transform(df_full_test[cat])
        encoders[cat] = label_encoder
        logger.info(f'{cat}: {np.sort(df_full_train[cat].unique())}')
    #
    print(df_full_train.head())
    # Encode test:



    misc = {}
    misc['encoder_dict'] = encoders
    # profile = feature_df.profile_report(title=f'Pandas Profiling Report for {suffix}')
    # profile.to_file(output_file=os.path.join(project_dir, f"output_{suffix}.html"))

    df_full_train.to_pickle(output_filepath_df_train)
    df_full_test.to_pickle(output_filepath_df_test)
    with open(output_filepath_misc_train, 'wb') as f:
        pickle.dump(misc, f)

    return 0


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_filepath = os.path.join(project_dir, 'data', 'raw')
    output_filepath = os.path.join(project_dir, 'data', 'processed')
    preprocess_table(input_filepath, output_filepath)
