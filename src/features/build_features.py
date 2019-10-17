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


def main(input_filepath, output_filepath):
    input_filename = os.path.join(input_filepath, 'Train_df.pck')
    df = pd.read_pickle(input_filename)
    for col in DATE_COLUMNS:
        df[col] = (df[col] - pd.to_datetime('1970-01-01')).dt.total_seconds()
    print(df.sample(10))
    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_filepath = os.path.join(project_dir, 'data', 'processed')
    output_filepath = os.path.join(project_dir, 'data', 'final')
    os.makedirs(input_filepath, exist_ok=True)
    os.makedirs(output_filepath, exist_ok=True)

    df = main(input_filepath, output_filepath)
