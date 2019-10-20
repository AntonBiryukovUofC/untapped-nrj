import pandas as pd
from category_encoders import OrdinalEncoder
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger((__name__))


class NNPredictor():
    def __init__(self, categorical_features, numerical_features, data):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoder = OrdinalEncoder(cols=self.categorical_features, return_df=True, handle_unknown='value')
        self.scaler = StandardScaler()
        self.model = None
        self.data = data
        self.inputs = []
        self.embeddings = []

        for c in self.categorical_features:
            vals = data[c].unique()
            logger.info(f'{c} -- {vals}')

        self.build_full_network(data)

    def _build_embedding_layer(self, category, cat_cardinality=5, emb_dim=3):
        input_cat = Input(shape=(1,))
        embedding = Embedding(cat_cardinality, emb_dim, input_length=1,
                              name=f'{category.replace("`", "").replace("_", "")}')(input_cat)
        embedding = Reshape(target_shape=(emb_dim,))(embedding)
        self.inputs.append(input_cat)
        self.embeddings.append(embedding)

    def _build_numeric_layer(self, n_units):
        input_num = Input(shape=(len(self.numerical_features),))
        dense_num = Dense(n_units)(input_num)
        self.inputs.append(input_num)
        self.embeddings.append(dense_num)

    def build_full_network(self, data: pd.DataFrame):
        # Create the categorical embeddings first:
        for c in self.categorical_features:
            cardinality = data[c].unique().shape[0]
            self._build_embedding_layer(c, cat_cardinality=cardinality, emb_dim=min(50, (cardinality + 1) // 2))
        # Build numerical layer:
        self._build_numeric_layer(n_units=64)

        m = Concatenate()(self.embeddings)
        m = Dense(64, activation='relu')(m)
        m = Dropout(0.2)(m)
        m = Dense(32, activation='relu')(m)
        m = Dropout(0.2)(m)
        m = Dense(32, activation='relu')(m)
        m = Dropout(0.2)(m)
        output = Dense(1, activation='linear')(m)

        model = Model(self.inputs, output)
        model.compile(loss='mse', optimizer='adam')
        self.model = model

    def fit(self, **kwargs):
        self.model.fit(**kwargs)

    def predict(self, **kwargs):
        self.model.predict(**kwargs)

    def preprocess_data(self, X_train, X_val, X_test):


        input_list_train = []
        input_list_val = []
        input_list_test = []

        X_train[self.categorical_features] = X_train[self.categorical_features].astype(str)
        X_val[self.categorical_features] = X_val[self.categorical_features].astype(str)
        X_test[self.categorical_features] = X_test[self.categorical_features].astype(str)


        # Fit encoder
        self.encoder.fit(X_train)
        X_train = self.encoder.transform(X_train) - 1
        X_val = self.encoder.transform(X_val) - 1
        X_test = self.encoder.transform(X_test) - 1

        X_train = X_train.fillna(-1)
        X_test = X_test.fillna(-1)
        X_val = X_val.fillna(-1)

        # Fit scaler
        self.scaler.fit(X_train[self.numerical_features])
        X_train[self.numerical_features] = self.scaler.transform(X_train[self.numerical_features])
        X_test[self.numerical_features] = self.scaler.transform(X_test[self.numerical_features])
        X_val[self.numerical_features] = self.scaler.transform(X_val[self.numerical_features])

        # the cols to be embedded: rescaling to range [0, # values)
        for c in self.categorical_features:
            input_list_train.append(X_train[c].values)
            input_list_val.append(X_val[c].values)
            input_list_test.append(X_test[c].values)

        input_list_train.append(X_train[self.numerical_features].values)
        input_list_val.append(X_val[self.numerical_features].values)
        input_list_test.append(X_test[self.numerical_features].values)

        return input_list_train, input_list_val, input_list_test
