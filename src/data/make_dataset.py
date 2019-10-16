# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import random
import numpy as np
pd.set_option('display.width', 1800)
pd.set_option('display.max_columns',10)


# fix the seed for reproducibility:
random.seed(123)
np.random.seed(123)
COLS_TO_KEEP = "EPAssetsId,Province,LicenceNumber,UWI,CurrentOperator,CurrentOperatorParent,CurrentOperatorID,Licensee," \
           "LicenseeParentCompany,LicenseeID,WellTypeStandardised," \
           "Formation,Field,Pool,SurveySystem,Surf_Location,Surf_Township," \
           "Surf_Meridian,Surf_Range,Surf_Section,Surf_LSD,Surf_Longitude,Surf_Latitude,Surf_TownshipRange," \
           "Surf_QuarterUnit,Surf_Unit,Surf_Block,Surf_NTSMapSheet,Surf_Series,Surf_Area,Surf_Sheet," \
           "Surf_QuarterSection,BH_Location," \
           "BH_Longitude,BH_Latitude," \
           "GroundElevation,KBElevation,TotalDepth,LaheeClass,Confidential,SurfaceOwner,OSArea," \
           "OSDeposit,DrillingContractor,SpudDate,FinalDrillDate,RigReleaseDate,DaysDrilling,DrillMetresPerDay,TVD," \
           "WellProfile,ProjectedDepth," \
           "StatusDate,StatusSource,UnitID,UnitName,UnitFlag,Municipality,CompletionDate,Agent,_Max`Prod`(BOE)," \
           "_Fracture`Stages,_Open`Hole,_Completion`Events "
# TODO
# TODO: LicenseeID / LicenceNumbre - can be encoded by counting the entries
# TODO: Operator - count-encoded ?
# TODO:

def main(input_filepath, output_filepath,suffix = 'Train'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    target_df = pd.read_csv(os.path.join(input_filepath,f'Viking - {suffix}.txt'),nrows=20)
    feature_df = pd.read_csv(os.path.join(input_filepath, f'Header - {suffix.lower()}.txt'),nrows=2000)


    #df = pd.read_table(target_df,nrows=20)
    print(target_df.head())
    print(feature_df['SurveySystem'].unique())

    df=None
    return df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_filepath = os.path.join(project_dir, 'data', 'raw')
    output_filepath = os.path.join(project_dir, 'data', 'processed')
    df = main(input_filepath, output_filepath)
