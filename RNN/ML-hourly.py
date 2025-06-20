#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
from matplotlib import rcParams
rcParams['font.family'] = 'Angsana New'
rcParams.update({'font.size': 22})
rcParams['axes.unicode_minus'] = False
#%%
# df_read = pd.read_csv('data/corrected_data.csv')
# df_read = pd.read_csv('data/meter_corrected.csv')
# df_read = pd.read_csv('data/chol-bangkla-corrected-2017-2022-hourly-corrected.csv', parse_dates=["datetime"])
df_read = pd.read_csv('data/chol-sp07-bangkhla-corrected-2017-2022-hourly.csv', parse_dates=["datetime"])

#%%
df = df_read.copy()
print(df.dtypes)

#%%
# df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['datetime'])
print(df.dtypes)

#%%
# df = df.loc[df["datetime"].dt.year==2022]
df = df.loc[(df["datetime"].dt.year==2022) & (df["datetime"].dt.hour==6) & (df["datetime"].dt.minute==0)]
df.sort_values(by=['datetime'],inplace=True)
df.reset_index(drop=True,inplace=True)

#%%
df = df.set_index(pd.DatetimeIndex(df['date'])).drop(['date'], axis=1)

#%%
# print(df["ec"].min())
# print(df["ec"].max())

#%%
# =============================================================================
# Define the variables you want to use to train the model.
# =============================================================================

data = df.filter(['ec'])
dataset = data.values

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#%%
# =============================================================================
# StandardScaler
# =============================================================================

# scaler = StandardScaler()
# scaler = scaler.fit(dataset)
# dataset_scaled = scaler.transform(dataset)

# dataset_scaled = scaler.fit_transform(dataset)

#%%
# =============================================================================
# MinMaxScaler
# =============================================================================

scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(dataset)
dataset_scaled = scaler.transform(dataset)

# dataset_scaled = scaler.fit_transform(dataset)

#%%
# =============================================================================
# Does not scale the dataset
# =============================================================================

# dataset_scaled = dataset

#%%
# n_future = 24
# n_past = 48

n_future = 14
n_past = 28

#%%
import math

train_set_len = math.ceil(len(dataset) * 0.80)
valid_set_len = math.ceil(len(dataset) * 0.10)

# # first 3yrs
# train_set_len = 26280

# # last year
# valid_set_len = 17544

#%%
# =============================================================================
# Train set
# =============================================================================

#%%
# train_set = dataset_scaled[0:train_set_len, :]
# train_set = dataset_scaled[0:train_set_len, :]


# x_train = []
# y_train = []

# for i in range(n_past, len(train_set) - n_future + 1):
#     x_train.append(train_set[i - n_past:i, 0:dataset.shape[1]])
#     y_train.append(train_set[i + n_future - 1:i + n_future, 0])

#%%
# x_train, y_train = np.array(x_train), np.array(y_train)

#%%
# =============================================================================
# Validation set
# =============================================================================

#%%
# valid_set = dataset_scaled[train_set_len - n_past:train_set_len + valid_set_len, :]

# x_valid = []
# y_valid = []

# for i in range(n_past, len(valid_set) - n_future + 1):
#     x_valid.append(valid_set[i - n_past:i, 0:dataset.shape[1]])
#     y_valid.append(valid_set[i + n_future - 1:i + n_future, 0])

#%%
# x_valid, y_valid = np.array(x_valid), np.array(y_valid)

#%%
# =============================================================================
# Test set
# =============================================================================    

#%%
# test_set = dataset_scaled[(train_set_len + valid_set_len) - n_past:, :]
# test_real = dataset[(train_set_len + valid_set_len) - n_past:, :]

# x_test = []
# y_test = []

# for i in range(n_past, len(test_set) - n_future + 1):
#     x_test.append(test_set[i - n_past:i, 0:dataset.shape[1]])
#     y_test.append(test_real[i + n_future - 1:i + n_future, 0])

#%%
# x_test, y_test = np.array(x_test), np.array(y_test)

#%%
# =============================================================================
# Train set (multiple)
# =============================================================================

#%%
train_set = dataset_scaled[0:train_set_len, :]

x_train = []
y_train = []

for i in range(n_past, len(train_set) - n_future + 1):
    x_train.append(train_set[i - n_past:i, 0:dataset.shape[1]])
    y_train.append(train_set[i:i + n_future, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

#%%
# =============================================================================
# Validation set (multiple)
# =============================================================================

#%%
valid_set = dataset_scaled[train_set_len - n_past:train_set_len + valid_set_len, :]

x_valid = []
y_valid = []

for i in range(n_past, len(valid_set) - n_future + 1):
    x_valid.append(valid_set[i - n_past:i, 0:dataset.shape[1]])
    y_valid.append(valid_set[i:i + n_future, 0])

x_valid, y_valid = np.array(x_valid), np.array(y_valid)
y_valid = np.reshape(y_valid, (y_valid.shape[0], y_valid.shape[1], 1))

#%%
# =============================================================================
# Test set (multiple)
# =============================================================================    

#%%
test_set = dataset_scaled[(train_set_len + valid_set_len) - n_past:, :]
test_real = dataset[(train_set_len + valid_set_len) - n_past:, :]

x_test = []
y_test = []

for i in range(n_past, len(test_set) - n_future + 1):
    x_test.append(test_set[i - n_past:i, 0:dataset.shape[1]])
    y_test.append(test_real[i:i + n_future, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

#%%
plt.figure(figsize=(16,8))
plt.title('Displays Train set, Validation set, and Test set.')
plt.xlabel('Date')
plt.ylabel('EC')
plt.plot(df[0:train_set_len]["ec"], label='Train')
plt.plot(df[train_set_len - n_past:train_set_len + valid_set_len]["ec"], label='Validation')
plt.plot(df[(train_set_len + valid_set_len) - n_past:]["ec"], label='Test')
plt.legend()
plt.show()

#%%
# =============================================================================
# Model
# =============================================================================

#%%
from keras import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import LSTM
# from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
# from tensorflow.keras.layers import Bidirectional
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

#%%
# =============================================================================
# RNN
# =============================================================================

# model = Sequential()
# model.add(SimpleRNN(192, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
# # model.add(SimpleRNN(192, return_sequences=True))
# model.add(SimpleRNN(192, return_sequences=False))
# model.add(Dense(n_future))

# model.add(SimpleRNN(128, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
# model.add(SimpleRNN(64, return_sequences=True))
# model.add(SimpleRNN(64, return_sequences=False))
# model.add(Dense(16))
# model.add(Dense(1))

#%%
# =============================================================================
# LSTM
# =============================================================================

# model = Sequential()
# model.add(LSTM(192, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
# # model.add(LSTM(192, return_sequences=True))
# model.add(LSTM(192, return_sequences=False))
# model.add(Dense(n_future))

# model = Sequential()
# model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dense(n_future))
# model.add(Dense(16))
# model.add(Dense(1))

#%%
# =============================================================================
# Bidirectional LSTM
# =============================================================================

model = Sequential()
model.add(Bidirectional(LSTM(196, return_sequences=True), input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Bidirectional(LSTM(196, return_sequences=False,activation="relu")))
model.add(Dense(n_future,activation="relu"))

# model.add(Dense(16))
# model.add(Dense(1))

#%%
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.summary()

#%%
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

#%%
checkpoint_filepath = 'checkpoint/bilstm-14-2022-batch-30-relu.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_freq='epoch',
    verbose=1
    )

#%%
model_fit = model.fit(
    x_train,
    y_train,
    batch_size=30,
    epochs=128,
    validation_data=(x_valid, y_valid),
    callbacks=[model_checkpoint_callback, early_stopping_callback]
    )

#%%
plt.figure(figsize=(16,8))
plt.title('Bi-LSTM: loss rate')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model_fit.history['loss'], label='Training loss')
plt.plot(model_fit.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

#%%
# from keras.models import load_model

# model = load_model('checkpoint/01-bilstm-24-3-2-1.h5')

# #%%
# pred = model.predict(x_test)

# #%%
# if dataset.shape[1] > 1:
#     pred = np.repeat(pred, dataset.shape[1], axis=-1)

# #%%
# y_pred = scaler.inverse_transform(pred)[:,0]

# #%%
# y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))

# #%%
# from sklearn.metrics import mean_squared_error

# rmse = mean_squared_error(y_test, y_pred, squared=False)
# print(f'RMSE of Bi-LSTM = {rmse}')            

# #%%
# # rmse = np.sqrt(np.mean(np.square((y_test - y_pred))))
# # print(rmse)

# #%%
# rmspe = (np.sqrt(np.mean(np.square((y_test - y_pred) / y_test)))) * 100
# print(f'RMSPE of Bi-LSTM = {rmspe}')


# #%%
# df_compare = pd.DataFrame({'Actual EC':df['ec_corrected'][train_set_len + valid_set_len:]})
# df_compare['Predicted EC'] = y_pred

# #%%
# plt.figure(figsize=(16,8))
# plt.title('Displays Actual EC vs. Predicted EC.')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.plot(df_compare['Actual EC'], label='Actual')
# plt.plot(df_compare['Predicted EC'], label='Predicted')
# plt.legend()
# plt.show()

# #%%
# plt.figure(figsize=(16,8))
# plt.title('Displays All Actual EC vs. Predicted EC.')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.plot(df['ec_new_.5'], label='All Actual EC')
# plt.plot(df_compare['Predicted EC'], label='Predicted')
# plt.legend()
# plt.show()

#%%
# =============================================================================
# Multiple output
# =============================================================================

#%%
from keras.models import load_model

model = load_model('checkpoint/bilstm-14-2022-batch-30-relu.h5')
#%%
pred = model.predict(x_test)

#%%
r_pred = None

#%%
for i in range(pred.shape[1]):
    t_pred = pred[:,i]
    t_pred = np.reshape(t_pred, (t_pred.shape[0], 1))
    t_pred = np.repeat(t_pred, dataset.shape[1], axis=-1)
    t_pred = scaler.inverse_transform(t_pred)[:,0]
    t_pred = np.reshape(t_pred, (t_pred.shape[0], 1))
    if r_pred is None:
        r_pred = t_pred
    else:
        r_pred = np.append(r_pred, t_pred, axis=1)

#%%
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

#%%
y_real = y_test[:,0]
y_real = np.reshape(y_real, (y_real.shape[0], 1))

#%%
y_pred = r_pred[:,0]
y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))

#%%
plt.figure(figsize=(16,8))
plt.title('Bi-LSTM')
plt.plot(y_real, label='y_real')
plt.plot(y_pred, label='y_pred')
plt.legend()
plt.show()

#%%
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test, r_pred, squared=False)
print("RMSE =",rmse)

#%%
from sklearn.metrics import mean_absolute_percentage_error

# calculate MAPE
mape = mean_absolute_percentage_error(y_test, r_pred)
print(f'MAPE = {mape}')

#%%
df_compare = pd.DataFrame({'Actual EC':df['ec_corrected'][train_set_len + valid_set_len:-(n_future-1)]})
df_compare['Predicted EC'] = y_pred

#%%
plt.figure(figsize=(16,8))
plt.title('Displays Actual EC vs. Predicted EC.')
plt.xlabel('Date')
plt.ylabel('Value')
plt.plot(df_compare['Actual EC'], label='Actual')
plt.plot(df_compare['Predicted EC'], label='Predicted')
plt.legend()
plt.show()

#%%
plt.figure(figsize=(16,8))
plt.title('Displays All Actual EC vs. Predicted EC.')
plt.xlabel('Date')
plt.ylabel('Value')
plt.plot(df['ec_new_.5'], label='All Actual EC')
plt.plot(df_compare['Predicted EC'], label='Predicted')
plt.legend()
plt.show()

#%%
# =============================================================================
# Calculate the Loss Function of each Timestep.
# =============================================================================

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

for i in range(r_pred.shape[1]):
    y_pred = r_pred[:,i]
    y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
    y_real = y_test[:,i]
    y_real = np.reshape(y_real, (y_real.shape[0], 1))
    rmse = mean_squared_error(y_real, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_real, y_pred)
    print(f'Calculate the loss function of Timestep t+{i+1}.')
    print(f'• RMSE of Timestep t+{i+1} = {rmse}')
    print(f'• MAPE of Timestep t+{i+1} = {mape}')
    print('')
    
#%%
# =============================================================================
# Create a line graph comparing actual and predicted values for each Timestep.
# =============================================================================

for i in range(r_pred.shape[1]):
    y_pred = r_pred[:,i]
    y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
    y_real = y_test[:,i]
    y_real = np.reshape(y_real, (y_real.shape[0], 1))
    plt.figure(figsize=(16,8))
    plt.title(f'Timestep t+{i+1}')
    plt.plot(y_real, label='y_real')
    plt.plot(y_pred, label='y_pred')
    plt.legend()
    plt.show()
    
#%%
# =============================================================================
# Create a line graph comparing actual and predicted values for each row of data.
# =============================================================================

n_row = 10
# n_row = len(r_pred)

for i in range(n_row):
    y_pred = r_pred[i,:]
    y_real = y_test[i,:]
    plt.figure(figsize=(16,8))
    plt.title(f'Row {i}')
    plt.plot(y_real, label='y_real')
    plt.plot(y_pred, label='y_pred')
    plt.legend()
    plt.show()
    
#%%
