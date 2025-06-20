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
df_sp_05_2019 = pd.read_csv('data/chol-sp05-bangkhanak-2019-hourly-corrected.csv', parse_dates=["datetime"])
df_sp_05_2022 = pd.read_csv('data/chol-sp05-bangkhanak-2022-hourly-corrected.csv', parse_dates=["datetime"])
df_sp_06 = pd.read_csv('data/chol-sp06-bangkrajed-2021-2022-hourly.csv', parse_dates=["datetime"])
df_sp_07 = pd.read_csv('data/chol-sp07-bangkhla-corrected-2017-2022-hourly.csv', parse_dates=["datetime"])
df_sp_08 = pd.read_csv('data/chol-sp08-chachoengsao-2020-2022-hourly-corrected.csv', parse_dates=["datetime"])
df_sp_09 = pd.read_csv("data/chol-sp09-cholcha-2021-2022-hourly-corrected.csv", parse_dates=["datetime"])
df_sp_10 = pd.read_csv('data/chol-sp10-bangpakong-2021-2022-hourly-corrected.csv', parse_dates=["datetime"])

#%%
df_05_2019 = df_sp_05_2019.loc[df_sp_05_2019["datetime"].dt.hour == 6]
df_05_2022 = df_sp_05_2022.loc[df_sp_05_2022["datetime"].dt.hour == 6]

df_06 = df_sp_06.loc[df_sp_06["datetime"].dt.hour == 6]
df_06_to_2021 = df_sp_06.loc[(df_sp_06["datetime"].dt.hour == 6) & (df_sp_06["datetime"].dt.year<=2021)]
df_06_2022 = df_sp_06.loc[(df_sp_06["datetime"].dt.hour == 6) & (df_sp_06["datetime"].dt.year==2022)]

df_07 = df_sp_07.loc[(df_sp_07["datetime"].dt.hour == 6) & (df_sp_07["datetime"].dt.year==2022)]

df_08 = df_sp_08.loc[df_sp_08["datetime"].dt.hour == 6]
df_08_to_2021 = df_sp_08.loc[(df_sp_08["datetime"].dt.hour == 6) & (df_sp_08["datetime"].dt.year <= 2021)]
df_08_2022 = df_sp_08.loc[(df_sp_08["datetime"].dt.hour == 6) & (df_sp_08["datetime"].dt.year == 2022)]

df_09 = df_sp_09.loc[(df_sp_09["datetime"].dt.hour == 6)]
df_09_train = df_09.loc[(df_09["datetime"].dt.year<2022)]
df_09_val = df_09.loc[(df_09["datetime"].dt.year==2022)]

df_10 = df_sp_10.loc[df_sp_10["datetime"].dt.hour == 6]

#%%
df_05_2019[["ec"]].plot(title="SP05-2019")
df_05_2022[["ec"]].plot(title="SP05-2022")

# df_06[["ec"]].plot(title="SP06")
df_06_to_2021[["ec"]].plot(title="SP06-until 2021")
df_06_2022[["ec"]].plot(title="SP06-2022")

df_07[["ec"]].plot(title="SP07")

# df_08[["ec"]].plot(title="SP08")
df_08_to_2021[["ec"]].plot(title="SP08-until 2021")
df_08_2022[["ec"]].plot(title="SP08-2022")

df_09[["ec"]].plot(title="SP09")
# df_09_train[["ec"]].plot(title="SP09_train")
# df_09_val[["ec"]].plot(title="SP09_val")

df_10[["ec"]].plot(title="SP10")

#%%
"""
train: 10 8<=2021 6 5=2019
val: 7
test: 8=2022
"""

#%%
for df in [
            df_05_2019,
           df_05_2022,
           df_06,
           df_06_to_2021,
           df_06_2022,
           df_07,
           df_08,
           df_08_to_2021,
           df_08_2022,
           df_09,
           df_10
           ]:
    df = df.set_index(pd.DatetimeIndex(df["datetime"])).drop(["datetime"], axis=1)

#%%
# # =============================================================================
# # Define the variables you want to use to train the model.
# # =============================================================================

#
dataset_05_2019 = df_05_2019.filter(["ec"]).values
dataset_05_2022 = df_05_2022.filter(["ec"]).values

dataset_06 = df_06.filter(["ec"]).values
dataset_06_to_2021 = df_06_to_2021.filter(["ec"]).values
dataset_06_2022 = df_06_2022.filter(["ec"]).values

dataset_07 = df_07.filter(["ec"]).values

dataset_08 = df_08.filter(["ec"]).values
dataset_08_to_2021 = df_08_to_2021.filter(["ec"]).values
dataset_08_2022 = df_08_2022.filter(["ec"]).values

dataset_09 = df_09.filter(["ec"]).values
dataset_09_train = df_09_train.filter(["ec"]).values
dataset_09_val = df_09_val.filter(["ec"]).values

dataset_10 = df_10.filter(["ec"]).values

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#%%
# =============================================================================
# MinMaxScaler
# =============================================================================

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(np.concatenate([
                            dataset_05_2019,
                        #    dataset_05_2022,
                           dataset_06,
                           dataset_07,
                           dataset_08,
                        #    dataset_09,
                           dataset_10
                           ]))
# scaler.fit(np.concatenate([
#                            dataset_06,
#                            dataset_07,
#                            dataset_08
#                            ]))

dataset_05_2019_scaled = scaler.transform(dataset_05_2019)
# dataset_05_2022_scaled = scaler.transform(dataset_05_2022)

dataset_06_scaled = scaler.transform(dataset_06)
dataset_06_to_2021_scaled = scaler.transform(dataset_06_to_2021)
dataset_06_2022_scaled = scaler.transform(dataset_06_2022)

dataset_07_scaled = scaler.transform(dataset_07)

dataset_08_scaled = scaler.transform(dataset_08)
dataset_08_to_2021_scaled = scaler.transform(dataset_08_to_2021)
dataset_08_2022_scaled = scaler.transform(dataset_08_2022)

# dataset_09_scaled = scaler.transform(dataset_09)
# dataset_09_train_scaled = scaler.transform(dataset_09_train)
# dataset_09_val_scaled = scaler.transform(dataset_09_val)

dataset_10_scaled = scaler.transform(dataset_10)

#%%
n_future = 14
n_past = 28

# n_future = 7
# n_past = 14

#%%
# =============================================================================
# Split the dataset into Train set, Validation set, and Test set. (for multiple output)
# =============================================================================

#%%
# =============================================================================
# Train set (for multiple output)
# =============================================================================

# test_set = datasets_test_scaled
# test_real = datasets_test

# x_test = []
# y_test = []

# for i in range(n_past, len(test_set) - n_future + 1):
#     x_test.append(test_set[i - n_past:i, 0:dataset_test.shape[1]])
#     y_test.append(test_real[i:i + n_future, 0])

# x_test, y_test = np.array(x_test), np.array(y_test)
# y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

#%%
# =============================================================================
# Validation set (for multiple output)
# =============================================================================


# valid_set = dataset_07_scaled

# x_valid = []
# y_valid = []

# for i in range(n_past, len(valid_set) - n_future + 1):
#     x_valid.append(valid_set[i - n_past:i, 0:dataset_07.shape[1]])
#     y_valid.append(valid_set[i:i + n_future, 0])

# x_valid, y_valid = np.array(x_valid), np.array(y_valid)
# y_valid = np.reshape(y_valid, (y_valid.shape[0], y_valid.shape[1], 1))

#%%
# # =============================================================================
# # Test set (for multiple output)
# # =============================================================================    

test_set = dataset_08_2022_scaled
test_real = dataset_08_2022

x_test = []
y_test = []

for i in range(n_past, len(test_set) - n_future + 1):
    x_test.append(test_set[i - n_past:i, 0:dataset_07.shape[1]])
    y_test.append(test_real[i:i + n_future, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

#%%
# =============================================================================
# Split the dataset into Train set, Validation set, and Test set. (for multiple datasets)
# =============================================================================

#%%
# =============================================================================
# Train set (for multiple datasets)
# =============================================================================

datasets_train = [dataset_10_scaled,
                #   dataset_09_scaled,
                  dataset_08_to_2021_scaled,
                  dataset_06_scaled,
                  dataset_05_2019_scaled
                #   dataset_05_2022_scaled
                  ]
# datasets_train = [dataset_08_to_2021_scaled,
#                   dataset_06_to_2021_scaled
#                   ]

x_train = []
y_train = []

for dataset in datasets_train:
    
    train_set = dataset

    for i in range(n_past, len(train_set) - n_future + 1):
        x_train.append(train_set[i - n_past:i, 0:dataset.shape[1]])
        y_train.append(train_set[i:i + n_future, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

#%%
# =============================================================================
# Validation set (for multiple datasets)
# =============================================================================

datasets = [dataset_07_scaled]

x_valid = []
y_valid = []

for dataset in datasets:
    
    valid_set = dataset

    for i in range(n_past, len(valid_set) - n_future + 1):
        x_valid.append(valid_set[i - n_past:i, 0:dataset.shape[1]])
        y_valid.append(valid_set[i:i + n_future, 0])

x_valid, y_valid = np.array(x_valid), np.array(y_valid)
y_valid = np.reshape(y_valid, (y_valid.shape[0], y_valid.shape[1], 1))

#%%
# =============================================================================
# Test set (for multiple datasets)
# =============================================================================

# datasets = [[dataset_test_scaled_2020, dataset_test_2020],
#             [dataset_test_scaled_2021, dataset_test_2021],
#             [dataset_test_scaled_2022, dataset_test_2022]]

# x_test = []
# y_test = []

# for dataset in datasets:
    
#     test_set = dataset[0]
#     test_real = dataset[1]

#     for i in range(n_past, len(test_set) - n_future + 1):
#         x_test.append(test_set[i - n_past:i, 0:dataset_test_2020.shape[1]])
#         y_test.append(test_real[i:i + n_future, 0])

# x_test, y_test = np.array(x_test), np.array(y_test)
# y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

#%%
# plt.figure(figsize=(16,8))
# plt.title('Dataset: Train set-1')
# plt.xlabel('Date')
# plt.ylabel('EC')
# plt.plot(dataset_train_1_scaled)
# plt.legend()
# plt.show()

#%%
# plt.figure(figsize=(16,8))
# plt.title('Dataset: Train set-2')
# plt.xlabel('Date')
# plt.ylabel('EC')
# plt.plot(dataset_train_2_scaled)
# plt.legend()
# plt.show()

#%%
# plt.figure(figsize=(16,8))
# plt.title('Validation set')
# plt.xlabel('Date')
# plt.ylabel('EC')
# plt.plot(dataset_val_scaled)
# plt.legend()
# plt.show()

#%%
# plt.figure(figsize=(16,8))
# plt.title('Test set')
# plt.xlabel('Date')
# plt.ylabel('EC')
# plt.plot(dataset_test_scaled)
# plt.legend()
# plt.show()

#%%

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
# model.add(SimpleRNN(192, return_sequences=False))
# model.add(Dense(n_future))

#%%
# =============================================================================
# LSTM
# =============================================================================

# model = Sequential()
# model.add(Bidirectional(LSTM(192, return_sequences=True), input_shape=(x_train.shape[1],x_train.shape[2])))
# model.add(Bidirectional(LSTM(192, return_sequences=False)))
# model.add(Dense(n_future))

#%%
# =============================================================================
# Bidirectional LSTM
# =============================================================================

model = Sequential()
model.add(Bidirectional(LSTM(140, return_sequences=True), input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Bidirectional(LSTM(140,return_sequences=False)))
# model.add(Dense(n_future))
model.add(Dense(n_future,activation='relu'))

#%%
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.summary()

#%%
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

#%%
checkpoint_filepath = 'checkpoint/bilstm-14-sp-[10-8_to2021-6-5_2019]-[7]-[8_2022]-140x2-batch-30-relu_at_output.h5'
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
plt.title('Bi-LSTM: loss rate.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model_fit.history['loss'], label='Training loss')
plt.plot(model_fit.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

#%%
from keras.models import load_model

# model = load_model('checkpoint/bilstm-14-sp-[10-9-8-6_to2021-5_2019-5_2022]-[8_2022]-[7]-140x2-batch-30-relu_at_output.h5')
model = load_model(checkpoint_filepath)

#%%
pred = model.predict(x_test)

#%%
r_pred = None

#%%
dataset_test = np.concatenate([dataset_08_2022])

#%%
for i in range(pred.shape[1]):
    t_pred = pred[:,i]
    t_pred = np.reshape(t_pred, (t_pred.shape[0], 1))
    t_pred = np.repeat(t_pred, dataset_test.shape[1], axis=-1)
    t_pred = scaler.inverse_transform(t_pred)[:,0]
    t_pred = np.reshape(t_pred, (t_pred.shape[0], 1))
    if r_pred is None:
        r_pred = t_pred
    else:
        r_pred = np.append(r_pred, t_pred, axis=1)

#%%
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

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
