import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch



def data_read(file_csv):
    df = pd.read_csv(file_csv)
    df = df.replace('-',np.nan)
    df = df.fillna(0)
    X = df.drop(['REGISTRATION_DATE','RETAILER_EL_NUMBER','MSISDN','FAKE'],axis=1)
    X = X.values
    Y = df['FAKE'].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42,stratify=Y)
    return X_train, X_test, y_train, y_test 

X_train, X_test, y_train, y_test = data_read('data.csv')
input_shape = (X_train.shape[1],1)
input_dim = X_train.shape[1]
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=hp.Int('units_1',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
    						kernel_size=hp.Int('kernel_1',min_value=2,max_value=3),
    									input_shape=input_shape,
    									padding='same'))
    model.add(layers.Conv1D(filters=hp.Int('units_2',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
    						kernel_size=hp.Int('kernel_2',min_value=2,max_value=3),
                           				activation='relu',
                           				padding='same'))
    model.add(layers.MaxPool1D(hp.Int('pool_1',min_value=2,max_value=3)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(filters=hp.Int('units_3',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
    						kernel_size=hp.Int('kernel_3',min_value=2,max_value=3),
                           				activation='relu',
                           				padding='same'))
    model.add(layers.MaxPool1D(hp.Int('pool_2',min_value=2,max_value=3)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int('units_4',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           				activation='relu'))
    model.add(layers.Dense(units=hp.Int('units_5',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           				activation='relu'))
    model.add(layers.Dense(units=hp.Int('units_6',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           				activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['acc'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_acc',
    max_trials=5,
    executions_per_trial=3
    )

tuner.search_space_summary()


tuner.search(X_train, y_train,
             epochs=1,
             validation_data=(X_test, y_test))

models = tuner.get_best_models(num_models=2)

print(models)

print(models[0])

print(models[1])

print(models[0].summary())

print(models[1].summary())

tuner.results_summary()
