import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def data_read(data):
	df = pd.read_csv(data)
	a = df['DISTRICT_NAME'].unique()
	dic_a = dict(zip(a,range(len(a))))
	df = df.replace({"DISTRICT_NAME" : dic_a})
	data = df.values
	X = data[:,0:11]
	Y = data[:,11]
	Y=np.reshape(Y, (-1,1))
	scaler_x = MinMaxScaler()
	scaler_y = MinMaxScaler()
	print(scaler_x.fit(X))
	xscale=scaler_x.transform(X)
	print(scaler_y.fit(Y))
	yscale=scaler_y.transform(Y)
	X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
	return X_train, X_test, y_train, y_test,scaler_x,scaler_y

def model_regressor():
	model = Sequential()
	model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mae', optimizer='adam',metrics=['acc','mae'])
	return model

def model_regressor_2():
	model = Sequential()
	model.add(Dense(32, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(16, kernel_initializer='normal', activation='relu'))
	model.add(Dense(8, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal',activation='linear'))
	model.compile(loss='mae', optimizer='adam',metrics=['acc','mae'])
	return model


if __name__ == '__main__':
	np.random.seed(6)
	X_train, X_test, y_train, y_test,scaler_x,scaler_y = data_read('train_data.csv')
	model = model_regressor_2()
	history = model.fit(X_train, y_train, epochs=200, batch_size=50,  verbose=1, validation_split=0.1)
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
	inp = model.input
	input_shape = X_train.shape
	outputs = [layer.output for layer in model.layers]
	functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]

	print(outputs)
	print(functors)
	Xnew = np.array([[1.50000000e+01, 1.88200000e+03, 4.90000000e+01, 1.12109000e+05,
       1.05965642e+07, 2.20627883e+06, 4.46300000e+05, 1.30467451e+07,
       1.52272650e+03, 1.35181435e-01, 2.51580298e+01]])
	Xnew= scaler_x.transform(Xnew)
	ynew= model.predict(Xnew)
	ynew = scaler_y.inverse_transform(ynew) 
	Xnew = scaler_x.inverse_transform(Xnew)
	print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

	