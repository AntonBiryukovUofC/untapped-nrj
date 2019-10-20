from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from category_encoders import OrdinalEncoder
from keras.layers.embeddings import Embedding
import pandas as pd
import numpy as np


class NNPredictor():
    def __init__(self, categorical_features, numerical_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoder = OrdinalEncoder(cols=self.categorical_features, return_df=True, handle_unknown='return_nan')
        self.model = None
        self.inputs = []
        self.embeddings = []
        self.build_full_network()

    def _build_embedding_layer(self, category, cat_cardinality=5, emb_dim=3):
        input_cat = Input(shape=(1,))
        embedding = Embedding(cat_cardinality, emb_dim, input_length=1, name=f'{category}')(input_cat)
        embedding = Reshape(target_shape=(emb_dim,))(embedding)
        self.inputs.append(input_cat)
        self.embeddings.append(embedding)

    def _build_numeric_layer(self, n_units):
        input_num = Input(shape=(len(self.numerical_features, )))
        dense_num = Dense(n_units)(input_num)
        self.inputs.append(dense_num)
        self.embeddings.append(dense_num)

    def build_full_network(self, data: pd.DataFrame):
        # Create the categorical embeddings first:
        for c in self.categorical_features:
            cardinality = data[c].unique().shape[0]
            self.build_embedding_layer(c, cat_cardinality=cardinality, emb_dim=min(50, (cardinality + 1) // 2))
        # Build numerical layer:
        self.build_numeric_layer(n_units=64)

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

    def fit(self,**kwargs):
        self.model.fit(**kwargs)
    def predict(self,**kwargs):
        self.model.fit(**kwargs)
    def preprocess_data(self, X_train, X_val, X_test):

        input_list_train = []
        input_list_val = []
        input_list_test = []
        self.encoder.fit(X_train)
        X_train = self.encoder.transform(X_train)
        X_val = self.encoder.transform(X_val)
        X_test = self.encoder.transform(X_test)

        # the cols to be embedded: rescaling to range [0, # values)
        for c in self.categorical_features:
            input_list_train.append(X_train[c].values)
            input_list_val.append(X_val[c].fillna(0).values)
            input_list_test.append(X_test[c].fillna(0).values)

        input_list_train.append(X_train[self.numerical_features].values)
        input_list_val.append(X_val[self.numerical_features].values)
        input_list_test.append(X_test[self.numerical_features].values)

        return input_list_train, input_list_val, input_list_test
