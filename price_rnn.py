import os
import re
import fnmatch
from itertools import zip_longest
import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

PAIR = 'BTCUSD'

# HYPERPARAMETERS
PRED_PERIOD = '1min'
WINDOW_LEN = 15 # price data window
FORECAST_LEN = 15 # how many WINDOW_LEN's distance in future to predict
EPOCHS = 5
BATCH_SIZE = 64
SKIP_ROWS = 171400 # use for 'development' mode

# FORMATTING, etc.
DATA_DIR = 'data'
DATA_PROVIDER = 'gemini'
NAME = f'{PAIR}-{WINDOW_LEN}-SEQ-{FORECAST_LEN}-PRED-{int(time.time())}'
COL_NAMES = ['time', 'date', 'symbol', 'open', 'high', 'low', 'close', 'volume']

def classify(current, future):
    span = float(future) - float(current)
    if 0.0 <= span <= float("inf"):
        return 1
    else:
        return 0

# normalize, scale, balance
def preprocess_df(df):
    df = df.drop('future', 1)

    # normalize BTCUSD_close, BTCUSD_volume
    print('NORMALIZING DATA:\n', df.sample(10))
    for col in df.columns:
        if col != 'target':
            # start simple
            # df[col] = df[col].pct_change()
            df[col] = (df[col] - df[col].mean()) / (df[col].max() - df[col].min())

    print('ARRANGING NORMALIZED DATA:\n', df.sample(10))
    # arrange data into seq -> target pairs for training, where seq
    # to see how window or 'lookback' period effects prediction accuracy
    seq_data = []
    # sliding window cache - old values drop off
    prev_days = deque(maxlen=WINDOW_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == WINDOW_LEN:
            seq_data.append([np.array(prev_days), i[-1]])

    # randomize to prevent overfitting
    random.shuffle(seq_data)
    print('BALANCING DATA:\n\n', seq_data[0][0][0:2])
    # balance the data
    buys, sells = [], []
    for seq, target in seq_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    # randomize to prevent overfitting
    random.shuffle(buys)
    random.shuffle(sells)

    # balance out the distribution of buys and sells
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    seq_data = buys + sells

    print('SPLITTING DATA:\n\n', seq_data[0][0][0:2])
    # split data into train, test sets
    # to prevent buys or sells from skewing data, randomize
    random.shuffle(seq_data)
    x, y = [], []
    for window_seq, target in seq_data:
        x.append(window_seq)
        y.append(target)

    print('TRAINING DATA SAMPLE:\n', x[0][0][0:2][:])
    print('TEST DATA SAMPLE:\n', y[0:2])
    return np.array(x), y


# don't change these, will break load loop
years = ['2015', '2016', '2017', '2018', '2019']
FILE_FILTER = f'{DATA_PROVIDER}_{PAIR}_*{PRED_PERIOD}.csv'

def load_data():
    main_df = pd.DataFrame()
    for path, dirlist, filelist in os.walk(DATA_DIR):
        for year, filename in zip(years, fnmatch.filter(filelist, FILE_FILTER)):
            if not year == '2019':
                continue
            print('LOADING FILE FOR YEAR: ', year)
            file = os.path.join(path, filename)
            df = pd.read_csv(f'{file}', skiprows=SKIP_ROWS, names=COL_NAMES)

            df.rename(columns={'close': f'{PAIR}_close', 'volume': f'{PAIR}_volume'}, inplace=True)
            df.set_index('time', inplace=True)

            # the features we care about
            df = df[[f'{PAIR}_close', f'{PAIR}_volume']]
            if len(main_df) == 0:
                main_df = df
            else:
                main_df = main_df.append(df)
    return main_df


# add to dataframe
# import pdb; pdb.set_trace()
main_df = load_data()

# add a future price column shifted in relation to close
main_df['future'] = main_df[f'{PAIR}_close'].shift(-FORECAST_LEN)
main_df['target'] = list(map(classify, main_df[f'{PAIR}_close'], main_df['future']))

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

# split data
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

# print(main_df.head())

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)
# import pdb; pdb.set_trace()

# shows balance
print(f'train data: {len(train_x)}, validation data: {len(validation_x)}')
print(f'Dont buys: {train_y.count(0)} buys: {train_y.count(1)}')
print(f'VALIDATION Dont buys: {validation_y.count(0)} VALIDATION buys: {validation_y.count(1)}')

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if not os.path.exists('logs'):
    os.makedirs('logs')
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

# unique filename to include epoch and validation accuracy for that epoch
if not os.path.exists('models'):
    os.makedirs('models')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath,
                                                      monitor='val_acc',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      mode='max')) #saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint])

print(model.evaluate(validation_x, validation_y))
