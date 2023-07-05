from time import time

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from numpy import array

import numpy as np
import keras as ks
import pandas as pd

class Predictor:
    """Predict future load given current one
    Just init this class and use predict function to predict
    """

    def __init__(self, init_load, model_path='/home/ubuntu/my_model.h5', scaler_path='/home/ubuntu/my_scaler.save', n_out=50):
        self.last_step = init_load
        self.model = ks.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path) 
        self.n_out = n_out
        print('prediction model loaded\n')

    def forecast_lstm(self, X):
        X = X.reshape(1, 1, len(X))
        forecast = self.model.predict(X, batch_size=1)
        # return an array
        return list(forecast[0, :])

    def inverse_difference(self, last_ob, forecast):
        inverted = [forecast[0] + last_ob]
        inverted.extend(forecast[i] + inverted[i-1] for i in range(1, len(forecast)))
        return inverted

    def inverse_transform(self, forecast, current_load):
        forecast = array(forecast)
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = self.scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        return self.inverse_difference(current_load, inv_scale)

    def predict(self, current_load):
        X = [[(current_load - self.last_step)]]
        self.last_step = current_load
        X=np.asarray(X)
        X.reshape(-1, 1)
        Y = self.scaler.transform(X)
        forecast = self.forecast_lstm(Y)
        return self.inverse_transform(forecast, current_load)
