import os
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam, SGD
from keras import Model
from pathlib import Path

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_model():
    inp = Input(shape=(3,))
    l1 = Dense(16, activation="relu")(inp)
    d1 = Dropout(rate=0.2)(l1)
    l2 = Dense(8, activation="relu")(d1)
    d2 = Dropout(rate=0.2)(l2)
    l3 = Dense(8, activation="relu")(d2)
    l4 = Dense(3)(l3)
    model = Model(inp, l4)
    model.compile(loss="mae", optimizer=Adam(lr=5e-3))
    return model


project_dir = Path(__file__).resolve().parents[2]
input_file_path = os.path.join(project_dir, "data", "final")
output_file_path = os.path.join(project_dir, "models")

df = pd.read_pickle(os.path.join(input_file_path, "OOF.pck"))
feats = ["Oil_norm", "Gas_norm", "Water_norm"]
targets = [f"gt_{x}" for x in feats]

X = df[feats].dropna()
Y = df.loc[X.index, targets]
Y_scaled = Y.copy()
train_index, test_index = np.arange(0, 4544), np.arange(4544, 5744)
X_train = X.iloc[train_index, :]
X_test = X.iloc[test_index, :]
Y_train = Y.iloc[train_index, :]
Y_test = Y.iloc[test_index, :]


scaler_x = StandardScaler()
scaler_y = {}

scaler_x.fit(X_train)
X_scaled_train = scaler_x.transform(X_train)
X_scaled_test = scaler_x.transform(X_test)

for t in targets:
    scaler_y[t] = MinMaxScaler(feature_range=(-1, 1))
    scaler_y[t].fit(Y[t].values.reshape(-1, 1))
    Y_scaled[t] = scaler_y[t].transform(Y[t].values.reshape(-1, 1))
Y_scaled_train = Y_scaled.iloc[train_index, :]
Y_scaled_test = Y_scaled.iloc[test_index, :]

model = create_model()
model.fit(X_scaled_train, Y_scaled_train, validation_split=0.2, batch_size=1, epochs=15)

y_preds_test = pd.DataFrame(data=model.predict(X_scaled_test), columns=targets)
for t in targets:
    y_preds_test[t] = scaler_y[t].inverse_transform(
        y_preds_test[t].values.reshape(-1, 1)
    )
    score = mean_absolute_error(Y_test[t], y_preds_test[t])
    print(f"Score for  {t} is {score}")
