"""
An deep RNN model for predictions on price sequence data
"""
import os
import re
from pdb import set_trace as bp
import fnmatch
from itertools import zip_longest
from collections import deque
import random
import numpy as np
import time
import pandas as pd

from pykalman import KalmanFilter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.regularizers import L1L2
from sklearn import preprocessing
from matplotlib import pyplot as plt


class PriceRNN:
    def __init__(
        self,
        pair="BTCUSD",
        period="1min",
        wlen=60,
        flen=3,
        years=["2015", "2016", "2017", "2018", "2019"],
        epochs=10,
        dropout=0.2,
        testpct=0.15,
        lossfunc="mse",
        batch_size=64,
        neurons=[128] * 4,
        lr=0.001,
        decay=1e-6,
        scaler=preprocessing.MinMaxScaler(feature_range=(0, 1)),
        dataprovider="gemini",
        datadir="data",
        skiprows=3,
        chunksize=10_000,
    ):
        self.dataprovider = dataprovider
        self.datadir = datadir
        self.pair = pair
        self.period = period
        self.file_filter = f"{dataprovider}_{pair}_*{period}.csv"
        self.wlen = wlen  # price data window
        self.flen = flen  # how many data points in future to predict
        self.years = years
        self.epochs = epochs
        self.dropout = dropout
        self.testpct = testpct
        self.lossfunc = lossfunc
        self.batch_size = batch_size
        self.neurons = neurons
        self.lr = lr
        self.decay = decay
        self.scaler = scaler
        self.skiprows = skiprows
        self.chunksize = chunksize
        self.usecols = ["Date", "Price", "Vol2", "High"]
        self.index_col = "Date"
        self.parse_dates = True
        self.name = f"{pair}-DPT{dropout}-BCH{batch_size}-NEUR{neurons}-LR{lr}-TPCT{testpct}-WLEN{wlen}-FLEN{flen}-{int(time.time())}"
        self.file_filter = f"{dataprovider}_{pair}_*{period}.csv"

    def extract_data(self):
        main_df = pd.DataFrame()

        df = pd.read_csv(
            f"{self.datadir}/btc_training.csv",
            index_col=self.index_col,
            parse_dates=self.parse_dates,
            usecols=self.usecols,
        )

        df = df[(df.index >= "2017-01-01")]

        # the features we care about, volume is notoriously inaccurate in cryptoland
        df = df[["Price", "High"]]
        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)

        main_df.fillna(method="ffill", inplace=True)
        main_df.dropna(inplace=True)
        return main_df

    def transform_data(self, main_df):
        # add the future price flen out as target column shifted in relation to close
        main_df["target"] = main_df["Price"].shift(-self.flen)
        main_df = main_df[~main_df.isin([np.nan, np.inf, -np.inf]).any(1)]
        self.main_df = main_df  # store pure original un-altered features

        train_df, test_df = self.split_dataset(main_df)

        # normalize
        for col in train_df.columns:
            train_df[col] = self.scaler.fit_transform(
                train_df[col].values.reshape(-1, 1)
            )

        for col in test_df.columns:
            test_df[col] = self.scaler.transform(test_df[col].values.reshape(-1, 1))

        return train_df, test_df

    # arrange, split
    def load(self, df):
        seq_data = self.load_input_sequences(df)
        # seq_data = self.balance(seq_data)

        print("SPLITTING DATA:\n", seq_data[0][0][0:1][0])
        # split data into train, test sets
        # to prevent skewing of distribution fed into network, randomize
        random.shuffle(seq_data)
        x, y = [], []
        for window_seq, target in seq_data:
            x.append(window_seq)
            y.append(target)

        return np.array(x), y

    # convert data into seq -> target pairs
    def load_input_sequences(self, df):
        print("ARRANGING DATA:\n", df.head(10))
        model_input = []

        prev_seq = deque(maxlen=self.wlen)  # acts as sliding window
        for i in df.values:
            prev_seq.append([n for n in i[:-1]])  # exclude target (i[:-1])
            if len(prev_seq) == self.wlen:
                model_input.append([np.array(prev_seq), i[-1]])

        random.shuffle(model_input)  # prevent skew in distribution
        return model_input

    def split_dataset(self, main_df):
        times = sorted(main_df.index.values)
        test_cutoff = times[-int(self.testpct * len(times))]
        print("Test cutoff: ", test_cutoff)
        # SPLIT DATA INTO (test_cutoff)% TEST, (1-test_cutoff)% TRAIN
        test_df = main_df[(main_df.index >= test_cutoff)]
        train_df = main_df[(main_df.index < test_cutoff)]
        return train_df, test_df

    def run(self):
        random.seed(230)  # determinism

        # Extract, Transform, Load
        main_df = self.extract_data()
        train_df, test_df = self.transform_data(main_df)
        x_train, y_train = self.load(train_df)
        x_test, y_test = self.load(test_df)

        # shows balance
        print(
            f"x_train: {x_train.shape}, {len(x_train)}, x_test: {x_test.shape}, {len(x_test)}"
        )
        print(f"x_train.shape[1:]: {x_train.shape[1:]}")
        print(f"y_test: {len(y_train)}, y_test: {len(y_test)}")
        model = self.model(x_train)

        opt = tf.keras.optimizers.Adam(lr=self.lr, decay=self.decay)

        model.compile(
            loss=self.lossfunc, optimizer=opt, metrics=["mean_absolute_error", "acc"]
        )

        if not os.path.exists("logs"):
            os.makedirs("logs")
        tensorboard = TensorBoard(log_dir=f"logs/{self.name}")

        if not os.path.exists("models"):
            os.makedirs("models")
        # unique filename to include epoch and validation accuracy for that epoch
        filepath = "RNN_Final-{epoch:02d}-{val_loss:.3f}"
        checkpoint = ModelCheckpoint(
            "models/{}.model".format(
                filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="max"
            )
        )

        history = model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_test, y_test),
            callbacks=[tensorboard],
        )

        self.predict(model)

        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model train vs validation loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper right")
        plt.savefig(f"plots/{self.name}.png")
        plt.clf()

        print("Eval Metrics ", model.metrics_names)
        print(model.evaluate(x_test, y_test))
        print(model.summary())

    def predict(self, model):
        df_test = pd.read_csv(
            f"{self.datadir}/btc_test.csv", index_col="Date", parse_dates=True
        )
        actual_btc_price = df_test.iloc[:, 1:2].values

        df_test = df_test[["Price", "High"]]

        df_total = pd.concat(
            (self.main_df[["Price", "High"]], df_test[["Price", "High"]]), axis=0
        )

        X_test = []
        inputs = df_total[len(df_total) - len(df_test) - self.wlen :].values
        inputs = self.scaler.transform(inputs)
        prev_days = deque(maxlen=self.wlen)
        for i in inputs:
            prev_days.append(i)
            if len(prev_days) == self.wlen:
                X_test.append(np.array(prev_days))

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))
        pred_price = model.predict(X_test)
        pred_price = self.scaler.inverse_transform(pred_price)
        pred_price = pd.DataFrame(pred_price)
        print(pred_price.info())
        print(pred_price.head())
        # Visualise the results
        plt.plot(actual_btc_price, color="red", label="Actual BTC Price")
        plt.plot(pred_price, color="blue", label="Predicted BTC Price")
        plt.title("BTC Prediction")
        plt.xlabel("Time")
        plt.ylabel("BTC")
        plt.legend()
        plt.savefig(f"plots/prediction.png")
        plt.clf()

    def model(self, x_train):
        model = Sequential()
        model.add(
            LSTM(
                self.neurons[0], input_shape=(x_train.shape[1:]), return_sequences=True
            )
        )
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(LSTM(self.neurons[1], return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(LSTM(self.neurons[2]))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(Dense(self.neurons[3]))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(self.dropout))

        model.add(Dense(1))
        return model


w = 120
for wlen, flen, btch, neurons in [(w, 13, 15, 35)]:
    wlen = int(wlen)
    flen = int(flen)
    PriceRNN(
        pair="BTCUSD",
        period="1d",
        wlen=wlen,
        flen=flen,
        dropout=0.2,
        epochs=150,
        batch_size=btch,
        neurons=[neurons] * 4,
        testpct=0.28,
        lr=0.001,
        decay=1e-6,
        datadir="/crypto_data",
    ).run()
