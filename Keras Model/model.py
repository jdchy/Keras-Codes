import pandas as pd
from sklearn import preprocessing
import numpy as np
from time import time
from keras.layers import Conv1D, MaxPool1D, Flatten
from keras.layers import Dropout, Dense, TimeDistributed,LeakyReLU,LSTM,TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import backend as K
from sklearn import preprocessing
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn import svm
from tensorflow import keras
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

def data_read(file_csv):
	df = pd.read_csv(file_csv)
	df = df.replace('-',np.nan)
	df = df.fillna(0)
	X = df.drop(['REGISTRATION_DATE','RETAILER_NUMBER','MSISDN','FAKE'],axis=1)
	# scaler = preprocessing.StandardScaler()
	# X = scaler.fit_transform(X)
	X = X.values
	Y = df['FAKE']
	Y = Y.values
	# print("Before OverSampling, counts of label '1': {}".format(sum(Y==1)))
	# print("Before OverSampling, counts of label '0': {} \n".format(sum(Y==0)))
	# sm = SMOTE(random_state=2)
	# X, Y = sm.fit_sample(X, Y.ravel())
	# print("After OverSampling, counts of label '1': {}".format(sum(Y==1)))
	# print("After OverSampling, counts of label '0': {}".format(sum(Y==0)))
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42,stratify=Y)
	return X_train, X_test, y_train, y_test
	

def model(input_dim):
	model = Sequential()
	model.add(Dense(120, input_dim=input_dim, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1,activation='sigmoid'))
	model.summary()
	model.compile(optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
	return model

def model_conv(input_shape):
	model = Sequential()
	model.add(Conv1D(160, 2, padding='same', input_shape=input_shape))
	# model.add(LeakyReLU(alpha=0.01))
	model.add(Conv1D(384, 3, activation='relu',padding='same'))
	model.add(MaxPool1D(3))
	model.add(Dropout(0.5))
	model.add(Conv1D(384, 3, activation='relu',padding='same'))
	# model.add(Conv1D(64, 3, activation='relu',padding='same'))
	model.add(MaxPool1D(2))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(352,activation='relu'))
	model.add(Dense(1352,activation='relu'))
	model.add(Dense(160,activation='relu'))
	model.add(Dense(1,activation='sigmoid'))
	model.summary()
	model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.0001),metrics=['acc'])
	return model

def lstm_model(input_shape):
	model = Sequential()
	model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
	model.add(LSTM(128, return_sequences=True))
	model.add(LSTM(96, return_sequences=True))
	model.add(Dropout(0.5))
	model.add(TimeDistributed(Dense(64, activation='relu')))
	# model.add(TimeDistributed(Dense(16, activation='relu')))
	model.add(Dropout(0.3))
	model.add(TimeDistributed(Dense(64, activation='relu')))
	model.add(TimeDistributed(Dense(32, activation='relu')))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.summary()
	model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.0001),metrics=['acc'])
	return model

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


if __name__ == '__main__':
	np.random.seed(5)
	X_train, X_test, y_train, y_test = data_read('data.csv')
	input_shape = (X_train.shape[1],1)
	X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
	model = lstm_model(input_shape)
	# input_dim = X_train.shape[1]
	# model = model(input_dim)
	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
	history = model.fit(X_train,y_train, epochs=30, shuffle=True,validation_data=(X_test,y_test), callbacks=[tensorboard])
	model.save('new_model.h5')
	model.save_weights('new_model_w.h5')
	print(history.history.keys())
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model acc')
	plt.ylabel('acc')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
	# model = load_model('Trail.h5')
	# model.summary()
	# results = model.evaluate(X_test, y_test, batch_size=50)
	# print("Test Acc :", results)
	# loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
	# print(recall)
	y_pred = model.predict(X_test, batch_size=64, verbose=1)

	y_pred_bool = []
	for i in range(len(y_pred)):
		if y_pred[i] > 0.50:
			y_pred_bool.append(1)
		else:
			y_pred_bool.append(0)
	report = classification_report(y_test, y_pred_bool,output_dict=True)
	print(report)
	
'''
if __name__ == '__main__':
	np.random.seed(5)
	X_train, X_test, y_train, y_test = data_read('Low_acq_w1.csv')
	classifier_linear = svm.SVC(kernel='linear')
	classifier_linear.fit(X_train, y_train)
	prediction_linear = classifier_linear.predict(X_test)
	report = classification_report(y_test, prediction_linear, output_dict=True)
	print('positive: ', report['pos'])
	print('negative: ', report['neg'])
	print('neutral: ', report['neu'])
'''