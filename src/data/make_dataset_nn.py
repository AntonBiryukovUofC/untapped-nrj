import os;
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from src.data.make_dataset import COLS_TO_KEEP,DATE_COLUMNS,COUNT_COLUMNS,CAT_COLUMNS, read_table

project_dir = Path(__file__).resolve().parents[2]
cols = COLS_TO_KEEP.split(",")
all_cols = CAT_COLUMNS + COUNT_COLUMNS + DATE_COLUMNS
cols_to_select = list(set(cols) - set(['UWI','EPAssetsId']) - set(CAT_COLUMNS)- set(COUNT_COLUMNS))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    for c in CAT_COLUMNS+COUNT_COLUMNS:
        df[c] = df[c].fillna('NA')
    return df

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto()
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        return model.predict(X_test)[:, 0]

def main(input_file_path, output_file_path, suffix):
   # items = []
   # for c in CAT_COLUMNS+COUNT_COLUMNS:
   #     items.append()
   output_filepath_df = os.path.join(output_file_path, f"{suffix}_df.pck")
   output_filepath_misc = os.path.join(output_file_path, f"{suffix}_misc.pck")
   feature_df = pd.read_csv(
       os.path.join(input_file_path, f"Header - {suffix.lower()}.txt")
   )

   vectorizer = make_union(
        ColumnSelector(cols = cols_to_select),
        on_field(CAT_COLUMNS+COUNT_COLUMNS,
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
   y_scaler = StandardScaler()
   with timer('process train'):
        train = feature_df
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(train))
        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        del train
   with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]

if __name__ == '__main__':
    input_filepath = os.path.join(project_dir, "data", "raw")
    output_filepath = os.path.join(project_dir, "data", "processed")
    suffix='Train'
    main(input_filepath,output_filepath,suffix)