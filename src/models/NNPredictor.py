import pandas as pd
from category_encoders import OrdinalEncoder
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler, MinMaxScaler,FunctionTransformer
import logging
import numpy as np

logger = logging.getLogger((__name__))
EPS = 1e-5


# noinspection PyPep8Naming
class NNPredictor:
    def __init__(self, categorical_features, numerical_features, data, **kwargs):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoder = OrdinalEncoder(
            cols=self.categorical_features, return_df=True, handle_unknown="value"
        )
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.model = None
        self.data = data
        self.inputs = []
        self.embeddings = []

        # for c in self.categorical_features:
        #     unique_values = data[c].unique()
        #     logger.info(f"{c} -- {unique_values}")

        self.build_full_network(data, **kwargs)

    def _build_embedding_layer(self, category, cat_cardinality=5, emb_dim=3):
        input_cat = Input(shape=(1,))
        embedding = Embedding(
            cat_cardinality,
            emb_dim,
            input_length=1,
            name=f'{category.replace("`", "").replace("_", "")}',
        )(input_cat)
        embedding = Reshape(target_shape=(emb_dim,))(embedding)
        self.inputs.append(input_cat)
        self.embeddings.append(embedding)

    def _build_numeric_layer(self, n_units):
        input_num = Input(shape=(len(self.numerical_features),))
        dense_num = Dense(n_units)(input_num)
        dense_num = Dense(n_units // 2)(dense_num)
        dense_num = Dense(n_units // 2)(dense_num)
        self.inputs.append(input_num)
        self.embeddings.append(dense_num)

    def build_full_network(
        self, data: pd.DataFrame, optimizer=SGD(lr=0.001), n_layers=[32, 16]
    ):
        # Create the categorical embeddings first:
        for c in self.categorical_features:
            cardinality = data[c].unique().shape[0]
            self._build_embedding_layer(
                c,
                cat_cardinality=cardinality + 1,
                emb_dim=min(50, (cardinality + 1) // 2),
            )
        # Build numerical layer:
        self._build_numeric_layer(n_units=20)

        m = Concatenate()(self.embeddings)
        m = Dense(n_layers[0], activation="relu")(m)
        #   m = Dropout(rate = 0.2)(m)
        m = Dense(n_layers[1], activation="relu")(m)
        # m = Dropout(rate = 0.2)(m)
        m = Dense(4, activation="relu")(m)
        m = Dense(4, activation="relu")(m)
        #      m = Dropout(rate = 0.2)(m)
        output = Dense(1, activation="linear")(m)

        model = Model(self.inputs, output)
        model.compile(loss="mae", optimizer=optimizer)
        self.model = model

    def fit(self, x, y, **kwargs):
        y = np.log(y + EPS).reshape(-1, 1)
        self.target_scaler.fit(y)
        y = self.target_scaler.transform(y)

        self.model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):

        y = self.model.predict(x, **kwargs)
        y = self.target_scaler.inverse_transform(y)
        y = np.exp(y) - EPS
        return y

    def preprocess_data(self, X_train, X_val, X_test):

        input_list_train = []
        input_list_val = []
        input_list_test = []

        X_train[self.categorical_features] = X_train[self.categorical_features].astype(
            str
        )
        X_val[self.categorical_features] = X_val[self.categorical_features].astype(str)
        X_test[self.categorical_features] = X_test[self.categorical_features].astype(
            str
        )

        # Fit encoder
        # the cols to be embedded: rescaling to range [0, # values)

        for c in self.categorical_features:
            raw_vals = np.unique(X_train[c])
            val_map = {}
            for i in range(len(raw_vals)):
                val_map[raw_vals[i]] = i
            input_list_train.append(X_train[c].map(val_map).values)
            input_list_val.append(X_val[c].map(val_map).fillna(len(raw_vals)).values)
            input_list_test.append(X_test[c].map(val_map).fillna(len(raw_vals)).values)

        for c in self.numerical_features:
            mu = np.nanmean(X_train[c])
            X_train[c] = X_train[c].fillna(mu)
            X_test[c] = X_test[c].fillna(mu)
            X_val[c] = X_val[c].fillna(mu)

        # Fit scaler
        self.scaler.fit(X_train[self.numerical_features])
        X_train[self.numerical_features] = self.scaler.transform(
            X_train[self.numerical_features]
        )
        X_test[self.numerical_features] = self.scaler.transform(
            X_test[self.numerical_features]
        )
        X_val[self.numerical_features] = self.scaler.transform(
            X_val[self.numerical_features]
        )

        input_list_train.append(X_train[self.numerical_features].values)
        input_list_val.append(X_val[self.numerical_features].values)
        input_list_test.append(X_test[self.numerical_features].values)

        return input_list_train, input_list_val, input_list_test


# noinspection PyPep8Naming
class NNPredictorNumerical:
    def __init__(self, numerical_features, data, **kwargs):
        self.numerical_features = numerical_features
        self.scaler = StandardScaler()
        self.target_scaler = FunctionTransformer(func=np.log1p,inverse_func=np.expm1())

        self.model = None
        self.data = data
        self.inputs = []
        self.build_full_network(**kwargs)

    def build_full_network(self, optimizer=SGD(lr=0.001)):
        # Create the categorical embeddings first:
        input_num = Input(shape=(len(self.numerical_features),))

        dense_num = Dense(256, activation="relu")(input_num)
        m = Dropout(rate=0.2)(dense_num)
        dense_num = Dense(128, activation="relu")(m)
        m = Dropout(rate=0.2)(dense_num)
        dense_num = Dense(64, activation="relu")(m)
        m = Dense(16, activation="relu")(dense_num)
        m = Dropout(rate=0.2)(m)
        m = Dense(8, activation="relu")(m)
        m = Dropout(rate=0.2)(m)

        m = Dense(4, activation="relu")(m)
        output = Dense(1, activation="linear")(m)

        model = Model(input_num, output)
        model.compile(loss="mae", optimizer=optimizer)
        self.model = model

    def fit(self, x, y, **kwargs):
        y = y.reshape(-1, 1)
        self.target_scaler.fit(y)
        y = self.target_scaler.transform(y)

        self.model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        y = self.model.predict(x, **kwargs)
        y = self.target_scaler.inverse_transform(y)
        return y

    def preprocess_data(self, X_train, X_val, X_test):
        input_list_train = []
        input_list_val = []
        input_list_test = []

        for c in self.numerical_features:
            mu = np.nanmean(X_train[c])
            X_train[c] = X_train[c].fillna(mu)
            X_test[c] = X_test[c].fillna(mu)
            X_val[c] = X_val[c].fillna(mu)

        # Fit scaler
        self.scaler.fit(X_train[self.numerical_features])
        X_train[self.numerical_features] = self.scaler.transform(
            X_train[self.numerical_features]
        )
        X_test[self.numerical_features] = self.scaler.transform(
            X_test[self.numerical_features]
        )
        X_val[self.numerical_features] = self.scaler.transform(
            X_val[self.numerical_features]
        )

        input_list_train.append(X_train[self.numerical_features].values)
        input_list_val.append(X_val[self.numerical_features].values)
        input_list_test.append(X_test[self.numerical_features].values)

        return input_list_train, input_list_val, input_list_test
