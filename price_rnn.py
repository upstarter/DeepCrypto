"""
An deep RNN model for price sequence data
"""
import os
import re
import fnmatch
from itertools import zip_longest
from collections import deque
import random
import numpy as np
import time
import pandas as pd

# from pykalman import KalmanFilter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.regularizers import L1L2
from sklearn import preprocessing
from matplotlib import pyplot


class PriceRNN:
    def __init__(
        self,
        pair="BTCUSD",  # UPPERCASE
        period="1min",
        window_len=60,
        forecast_len=3,
        years=["2015", "2016", "2017", "2018", "2019"],
        epochs=10,
        dropout=0.2,
        testpct=0.15,
        loss_func="sparse_categorical_crossentropy",
        batch_size=64,
        hidden_node_sizes=[128] * 4,
        learning_rate=0.001,
        decay=1e-6,
        scaler=preprocessing.MinMaxScaler(feature_range=(0, 1)),
        data_provider="gemini",
        data_dir="data",
        skiprows=3,
        chunksize=10_000,
    ):
        self.data_provider = data_provider
        self.data_dir = data_dir
        self.pair = pair
        self.period = period
        self.file_filter = f"{data_provider}_{pair}_*{period}.csv"
        self.window_len = window_len  # price data window
        self.forecast_len = forecast_len  # how many data points in future to predict
        self.years = years
        self.epochs = epochs
        self.dropout = dropout
        self.testpct = testpct
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.hidden_node_sizes = hidden_node_sizes
        self.learning_rate = learning_rate
        self.decay = decay
        self.scaler = scaler
        self.name = f"{pair}-WLEN{window_len}-FLEN{forecast_len}-{int(time.time())}"
        self.skiprows = skiprows
        self.chunksize = chunksize
        self.col_names = [
            "time",
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        self.file_filter = f"{data_provider}_{pair}_*{period}.csv"

    def classify(self, current, future):
        if float(future) > float(current):
            return 1
        else:
            return 0

    def extract_data(self):
        main_df = pd.DataFrame()
        for path, dirlist, filelist in os.walk(self.data_dir):
            for year, filename in zip(
                self.years, fnmatch.filter(filelist, self.file_filter)
            ):
                for allowed_year in self.years:
                    if not allowed_year == year:
                        continue
                print("LOADING FILE FOR YEAR: ", year)
                file = os.path.join(path, filename)
                df = pd.read_csv(
                    f"{file}",
                    names=self.col_names,
                    skiprows=self.skiprows,
                    # chunksize=self.chunksize,
                )

                # TODO move out of here, handle df as iterable in self.run
                # df = next(df)
                df.rename(
                    columns={
                        "close": f"{self.pair}_close",
                        "volume": f"{self.pair}_volume",
                    },
                    inplace=True,
                )

                df.set_index("time", inplace=True)

                # the features we care about
                df = df[[f"{self.pair}_close", f"{self.pair}_volume"]]
                if len(main_df) == 0:
                    main_df = df
                else:
                    main_df = main_df.join(df)
        # if there are gaps in data, use previously known values
        main_df.fillna(method="ffill", inplace=True)
        main_df.dropna(inplace=True)
        return main_df

    def run(self):
        random.seed(230)  # determinism

        main_df = self.extract_data()

        # add a future price column shifted in relation to close
        main_df["future"] = main_df[f"{self.pair}_close"].shift(-self.forecast_len)
        # classify and add target ground truth column
        main_df["target"] = list(
            map(self.classify, main_df[f"{self.pair}_close"], main_df["future"])
        )
        main_df = main_df.drop("future", 1)  # only needed to calculate target

        # main_df = self.denoise(main_df)

        x_train, y_train, x_test, y_test = self.split_and_preprocess_df(main_df)

        # shows balance
        print(f"train data: {len(x_train)}, validation data: {len(x_test)}")
        print(f"TRAIN do not buys: {y_train.count(0)} TRAIN buys: {y_train.count(1)}")
        print(
            f"VALIDATION Do not buys: {y_test.count(0)} VALIDATION buys: {y_test.count(1)}"
        )

        model = self.model(x_train)

        opt = tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay)

        model.compile(loss=self.loss_func, optimizer=opt, metrics=["accuracy"])

        if not os.path.exists("logs"):
            os.makedirs("logs")
        tensorboard = TensorBoard(log_dir=f"logs/{self.name}")

        if not os.path.exists("models"):
            os.makedirs("models")

        # unique filename to include epoch and validation accuracy for that epoch
        filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
        checkpoint = ModelCheckpoint(
            "models/{}.model".format(
                filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
            )
        )  # saves only the best ones

        history = model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_test, y_test),
            callbacks=[tensorboard, checkpoint],
        )

        print(history.history["loss"])
        print(history.history["acc"])
        print(history.history["val_loss"])
        print(history.history["val_acc"])

        if not os.path.exists("plots"):
            os.makedirs("plots")
        pyplot.plot(history.history["loss"])
        pyplot.plot(history.history["val_loss"])
        pyplot.title("model train vs validation loss")
        pyplot.ylabel("loss")
        pyplot.xlabel("epoch")
        pyplot.legend(["train", "validation"], loc="upper right")
        pyplot.savefig(f"plots/{self.name}.png")

        print(model.evaluate(x_test, y_test))
        print(model.summary())

    def balance(self, seq_data):
        print("BALANCING DATA:\n", seq_data[0][0][0:2])
        # balance the data
        buys, sells = [], []
        for seq, target in seq_data:
            if target == 0:
                sells.append([seq, target])
            elif target == 1:
                buys.append([seq, target])

        # randomize
        random.shuffle(buys)
        random.shuffle(sells)
        # balance out the distribution of buys and sells
        lower = min(len(buys), len(sells))
        buys = buys[:lower]
        sells = sells[:lower]
        seq_data = buys + sells
        return seq_data

    def convert_to_seq(self, df):
        print("ARRANGING DATA:\n", df.head(10))
        # convert data into seq -> target pairs for training to see how
        # self.window_len 'lookback' period effects prediction accuracy
        seq_data = []
        # sliding window cache - old values drop off
        prev_days = deque(maxlen=self.window_len)
        for i in df.values:
            prev_days.append([n for n in i[:-1]])  # exclude target (i[:-1])
            if len(prev_days) == self.window_len:
                seq_data.append([np.array(prev_days), i[-1]])
        random.shuffle(seq_data)  # prevent skew
        return seq_data

    def split_sequences(self, seq_data):
        print("SPLITTING DATA:\n", seq_data[0][0][0:2])
        # split data into train, test sets
        # to prevent buys or sells from skewing data, randomize
        random.shuffle(seq_data)
        x, y = [], []
        for window_seq, target in seq_data:
            x.append(window_seq)
            y.append(target)

        print("TRAINING DATA SAMPLE:\n", x[0][0][0:2][:])
        print("TEST DATA SAMPLE:\n", len(y))
        return np.array(x), y

    # arrange, balance, partition
    def preprocess_df(self, df):
        seq_data = self.convert_to_seq(df)
        seq_data = self.balance(seq_data)
        x, y = self.split_sequences(seq_data)
        return x, y

    def split_and_preprocess_df(self, df):
        times = sorted(df.index.values)
        print("times", len(times))
        test_cutoff = times[-int(self.testpct * len(times))]
        print("test_cutoff", test_cutoff)
        # SPLIT DATA INTO (1-test_cutoff)% TRAIN, (test_cutoff)% VALIDATE
        test_df = df[(df.index >= test_cutoff)]
        df = df[(df.index < test_cutoff)]

        ## NORMALIZE
        df = self.normalize(df)
        test_df = self.normalize(test_df, train=False)
        print("normalized train", df.head())
        print("normalized test", test_df.head())

        df.dropna(inplace=True)
        # import pdb
        #
        # pdb.set_trace()
        x_train, y_train = self.preprocess_df(df)
        x_test, y_test = self.preprocess_df(test_df)

        return x_train, y_train, x_test, y_test

    def model(self, x_train):
        model = Sequential()
        print("train shape", x_train.shape)
        print("train first example", len(x_train[0]))
        model.add(
            CuDNNLSTM(
                self.hidden_node_sizes[0],
                input_shape=(x_train.shape[1:]),
                return_sequences=True,
            )
        )
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(self.hidden_node_sizes[1], return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(self.hidden_node_sizes[2]))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        # model.add(
        #     Dense(self.hidden_node_sizes[3], activity_regularizer=regularizers.l2(0.01))
        # )
        # model.add(Dropout(self.dropout))
        # model.add(BatchNormalization())

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(self.dropout))

        model.add(Dense(2, activation="softmax"))
        return model

    def normalize(self, df, train=True):
        print("NORMALIZING DATA:\n", df.head())
        # normalize BTCUSD_close, BTCUSD_volume
        for col in df.columns:
            if col != "target":
                # print("col shape normalize", df[col].shape[1:])

                # nsamples, nx, ny = df[col].shape

                # df[col] = self.moving_average(df[col].values, 10)
                # start simple
                # df[col] = df[col].pct_change(fill_method="ffill")
                # df.dropna(inplace=True)
                print("before isin", len(df))
                # import pdb

                # pdb.set_trace()
                df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
                print("after isin", len(df))
                # df[col] = df[col].fillna(df[col].mean())

                if train:
                    print("Train Normalize")
                    # df[col] = self.scaler.fit_transform(df[col].values.reshape(-1, 1))
                    df[col] = preprocessing.scale(df[col].values)
                    # df[col] = (df[col] - df[col].mean()) / df[col].std()
                    # df[col] = (df[col] - df[col].min()) / (
                    #     df[col].max() - df[col].min()
                    # )
                else:
                    print("Test Normalize")
                    # df[col] = self.scaler.transform(df[col].values)
                    df[col] = preprocessing.scale(df[col].values)
                    # df[col] = (df[col] - df[col].mean()) / df[col].std()
                    # df[col] = (df[col] - df[col].min()) / (
                    #     df[col].max() - df[col].min()
                    # )

        return df  # seq.reshape(nsamples, nx, ny)

    def moving_average(self, seq, periods=10):
        weights = np.ones(periods) / periods
        return np.convolve(seq, weights, mode="valid")

    def denoise(self, df):
        for col in df.columns:
            if col != "target":
                # start simple, scale to interval [0,1]
                kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                prices = df[col].values
                results = kf.em(prices).smooth(prices)
                df[col] = results[0]
        df.dropna(inplace=True)
        return df


# Model to answer: If you were to buy at random based on model prediction, what
# hold period shows highest probability of profit
# TODO: stochastic random search and/or bayesian hyperparam optimization
hypers = [(120, 3)]
for wlen, flen in hypers:
    wlen = int(wlen)
    flen = int(flen)
    print("RUNNING MODEL: ")
    print("\twindow length: ", wlen)
    print("\tforecast length: ", flen)
    PriceRNN(
        pair="BTCUSD",
        period="1d",
        window_len=wlen,
        forecast_len=flen,
        dropout=0.2,
        epochs=100,
        batch_size=64,
        hidden_node_sizes=[128] * 4,
        testpct=0.40,
        learning_rate=0.001,
        # chunksize=1330,
        data_dir="/crypto_data",
    ).run()
