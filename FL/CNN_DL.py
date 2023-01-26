# Presettings
import nest_asyncio
nest_asyncio.apply()
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from collections import  OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow_federated as tff
from tensorflow_federated import python as tff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
tf.executing_eagerly()
tf.random.set_seed(24)
np.random.seed(24)
# Hyperparameters
IFPLOT = 0
DataProcessing = 0
FL_STEPS = 0
NUM_AGENTS = 30 #For current dataset, maximum value = 30
TRAINING_LOOP = 100
LOCAL_TRAINING_STEP = 1
CONSENSUS_STEP = 1
BATCHSIZE = 48

def create_wavenet_model():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[48, 1]))
    for dilation_rate in (1, 2 , 4, 8):
        model.add(keras.layers.Conv1D(filters=24, kernel_size=2,
                                     dilation_rate=dilation_rate,
                                     padding='causal',
                                     activation='relu'))
    model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    
    return model
def create_LSTM_model():
  return tf.keras.models.Sequential([
      keras.layers.LSTM(32, input_shape=(time_step, 1), kernel_regularizer='l2'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(1),
  ])
#Predefined Function
def plot_series(time, series, format='-', start=0, end=None, label=None, color=None):
  plt.plot(time[start:end], series[start:end], format, label=label, color=color)
  plt.xlabel('Time')
  plt.ylabel('Value')
  if label:
    plt.legend(fontsize=14)
  plt.grid(True)

def create_dataset_central(data, time_step):
  x_data, y_data = [], []
  for i in range(len(data) - time_step):
    x = data[i: (i + time_step)]
    x_data.append(x)
    y = [data[i + time_step]]
    y_data.append(y)
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  return x_data, y_data

def create_dataset_fed(data, time_step):
  x_nest, y_nest = [], []
  for j in range(len(data)):
    x_data, y_data = [], []
    for i in range(len(data[j]) - time_step):
      x = data[j][i: (i + time_step)]
      x_data.append(x)
      y = [data[j][i + time_step]]
      y_data.append(y)

    x_data = np.array(x_data)[:, :, np.newaxis]
    x_nest.append(x_data)
    y_nest.append(y_data)
  x_nest = np.array(x_nest)
  return [tf.data.Dataset.from_tensor_slices((x_nest[x], np.array(y_nest[x]))) for x in range(len(x_nest))]

def preprocess_train(dataset):
  def batch_format_fn(x_d, y_d):
    return OrderedDict(
        x=x_d,
        y=tf.reshape(y_d, [-1, 1])
    )
  return dataset.repeat(num_epochs).shuffle(shuffle_buffer, seed=1).batch(
      BATCHSIZE).map(batch_format_fn).prefetch(prefetch_buffer)

def preprocess_test(dataset):
  def batch_format_fn(x_d, y_d):
    return OrderedDict(
        x=x_d,
        y=tf.reshape(y_d, [-1, 1])
    )
  return dataset.batch(BATCHSIZE).map(batch_format_fn).prefetch(prefetch_buffer)

def create_CNN_model():
  return tf.keras.models.Sequential([
      keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape = [48, 1], kernel_regularizer='l2'),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(1),
  ])

def create_DNN():
  return tf.keras.models.Sequential([
      keras.layers.Dense(48, activation='relu',
      #input_dim = 48),either use Drop layer or regularizer
      input_dim = 48, kernel_regularizer='l2'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(42, activation='relu'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(1)
  ])

def model_fn():
  keras_model = create_DNN()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec = preprocess_example.element_spec,
      loss = tf.keras.losses.MeanSquaredError(),
      metrics = [tf.keras.metrics.MeanSquaredError()]
  )

def Laplacian_Matrix(NUM_AGENTS, topology='RING'):
        if topology == 'No':
            L = np.eye(NUM_AGENTS)

        if topology == 'RING':
            L = 0.5 * np.eye(NUM_AGENTS) + 0.25 * np.eye(NUM_AGENTS, k=1) + 0.25 * np.eye(NUM_AGENTS,
                                                                                          k=-1) + 0.25 * np.eye(
                NUM_AGENTS, k=NUM_AGENTS - 1) + 0.25 * np.eye(NUM_AGENTS, k=-NUM_AGENTS + 1)

        if topology == 'FULL':
            A = np.ones([NUM_AGENTS, NUM_AGENTS]) - np.eye(NUM_AGENTS)
            L = (A + sum(A[0]) * np.eye(NUM_AGENTS)) / sum(A + sum(A[0]) * np.eye(NUM_AGENTS))

        if topology == 'MS':
            A = np.random.randint(2, size=NUM_AGENTS * NUM_AGENTS)
            A = (np.ones([NUM_AGENTS, NUM_AGENTS]) - np.eye(NUM_AGENTS)) * A.reshape([NUM_AGENTS, NUM_AGENTS])
            vec = A + np.diag(A.sum(axis=1))
            zero_id = np.where(vec.sum(axis=1) == 0)
            for k in range(len(zero_id[0])):
                vec[zero_id[0][k]][zero_id[0][k]] = 1
            L = vec / vec.sum(axis=1).reshape(-1, 1)
        return L

def Consensus(data,steps,LR = 0.02,Graph='RING',TIME_DOMAIN = 'Continuous'):
    NUM_AGENTS = len(data)
    # if Graph == 'RING':
    #     if TIME_DOMAIN == 'Continuous':
    #         L = 2 * np.eye(NUM_AGENTS) - np.eye(NUM_AGENTS, k=1) - np.eye(NUM_AGENTS, k=-1) - np.eye(NUM_AGENTS,
    #                                                                                                  k=NUM_AGENTS - 1) - np.eye(
    #             NUM_AGENTS, k=-NUM_AGENTS + 1)
    #     else:
    #         L = 0.5 * np.eye(NUM_AGENTS) + 0.25 * np.eye(NUM_AGENTS, k=1) + 0.25 * np.eye(NUM_AGENTS,
    #                                                                                       k=-1) + 0.25 * np.eye(
    #             NUM_AGENTS, k=NUM_AGENTS - 1) + 0.25 * np.eye(NUM_AGENTS, k=-NUM_AGENTS + 1)
    L = Laplacian_Matrix(NUM_AGENTS,topology=Graph)
    # lenth = 1
    # for qqq in range(len(data[0].shape)):
    #     lenth = lenth * data[0].shape[qqq]
    # consensusdata = np.array(data).reshape(len(data),lenth)
    output_temp = np.tensordot(L,data,axes=((1),(0)))
    # for i in range(steps):
    #     # data = np.matmul(np.kron(np.eye(len(data[0])),L),np.array(data).reshape(400,1))
    #     if TIME_DOMAIN == 'Continuous':
    #         consensusdata = consensusdata - LR * np.matmul(L, consensusdata)
    #     else:
    #         consensusdata = (1-LR) * np.matmul(L, consensusdata)
    #     consensusdata = np.matmul(L, consensusdata)
    # output = []
    # for i in range(len(data)):
    #     output.append(consensusdata[i].reshape(data[0].shape))
    return output_temp
def DistributedLearning(models,step=10,LR = 0.02,Graph='RING',TIME_DOMAIN = 'Discrete'):
    NUM_AGENTS = len(models)
    # if Graph=='RING':
    #     if TIME_DOMAIN=='Continuous':
    #         L = 2*np.eye(NUM_AGENTS) - np.eye(NUM_AGENTS,k=1) - np.eye(NUM_AGENTS,k=-1) - np.eye(NUM_AGENTS,k=NUM_AGENTS-1) - np.eye(NUM_AGENTS,k=-NUM_AGENTS+1)
    #     else:
    #         L = 0.5*np.eye(NUM_AGENTS) + 0.25*np.eye(NUM_AGENTS,k=1) + 0.25*np.eye(NUM_AGENTS,k=-1) + 0.25*np.eye(NUM_AGENTS,k=NUM_AGENTS-1) + 0.25*np.eye(NUM_AGENTS,k=-NUM_AGENTS+1)
    weights = [[] for i in range(NUM_AGENTS)]
    for i in range(NUM_AGENTS):
        weights[i] = models[i].get_weights()

    # for j in range(step):
    for t in range(len(weights[0])):
        temp = []
        for tt in range(NUM_AGENTS):
            temp.append(weights[tt][t])
        temp = Consensus(temp, step, LR=LR, Graph=Graph,TIME_DOMAIN = TIME_DOMAIN)
        for tt in range(NUM_AGENTS):
            weights[tt][t] = temp[tt]

        # for k in range(len(weights[0])):
        #     weights_temp = [weights[i][k] for i in range(NUM_AGENTS)]
        #     if TIME_DOMAIN == 'Continuous':
        #         weights_temp = weights_temp - LR * L * weights_temp
        #     else:
        #         weights_temp = (1-LR) * L * weights_temp




    for i in range(NUM_AGENTS):
        models[i].set_weights(weights[i])
    return models
# Data Processing

random.seed(24)
if DataProcessing == 1:
    data = []
    with open ('House_30.txt', 'r') as reader:
      for line in reader:
        stripped_line = line.strip().split()
        data.append(stripped_line)
    random.seed(24)
    tem = [x[0] for x in data]
    houses = list(set(tem))
    date = []
    consumption = []
    for i in houses:
      date.append([float(x[1]) for x in data if x[0]==i])
      consumption.append([float(x[2]) for x in data if x[0]==i])

    if IFPLOT==1:
        for i in range(0, 30):
          plot_series(range(0, len(consumption[0])), consumption[i])
          plt.show()
          print('client {:2d}'.format(i))

          plt.clf()

    percentile = np.percentile(consumption, [0, 25, 50, 75, 100])
    IQR = percentile[3] - percentile[1]
    # increase the whisker to 3 instead of 1.5
    UPL = percentile[3] + IQR*3
    DNL = percentile[1] - IQR*3
    out_up = [0 for i in range(30)]
    out_dn = [0 for i in range(30)]
    for i in range(0, 30):
      for j in range(0, len(consumption[i])):
        if consumption[i][j] > UPL:
          consumption[i][j] = UPL
          out_up[i] = out_up[i] + 1
        elif consumption[i][j] < DNL:
          consumption[i][j] = DNL
          out_dn[i] = out_dn[i] + 1
else:
    consumption=np.load('/home/yi/Projects/FL-DL/FL/consumption.npy')


# Preparing Training dataset
length = len(consumption[0])
split = int(0.8*length)
train_raw = [x[0: split] for x in consumption]
test_raw = [x[split: ] for x in consumption]
time_step = 48
train_data_fed = create_dataset_fed(train_raw, time_step)
test_data_fed = create_dataset_fed(test_raw, time_step)
example_dataset = train_data_fed[0]
example_element = next(iter(example_dataset))
num_epochs = 10
shuffle_buffer = 100
prefetch_buffer = 10
preprocess_example = preprocess_train(example_dataset)

train_set_fed = [preprocess_train(train_data_fed[i]) for i in range(len(train_raw))]
test_set_fed = [preprocess_test(test_data_fed[i]) for i in range(len(test_raw))]
train_set_central = train_set_fed[0]
test_set_central = test_set_fed[0]

# # Creating Models
# if FL_STEPS==1:
#     iterative_process = tff.learning.build_federated_averaging_process(
#         model_fn,
#         client_optimizer_fn=lambda: keras.optimizers.Adam(0.001),
#         server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1)
#     )
#
#     logdir = "/tmp/logs/scalars/training"
#     summary_writer = tf.summary.create_file_writer(logdir)
#     state = iterative_process.initialize()
#     num_rounds = 10
#     fed_metrics = [[] for i in range(0, num_rounds)]
#
#     start_time = time.time()
#     for i in range(0, num_rounds):
#       state, metrics = iterative_process.next(state, train_set_fed)
#       fed_metrics[i] = metrics
#       print('round {:2d}, metrics={}'.format(i+1, metrics))
#
#     end_time = time.time()
#     train_time = end_time - start_time
#     print(train_time)
#
#     evaluation = tff.learning.build_federated_evaluation(model_fn)
#     metrics = evaluation(state.model, test_set_fed)
#     print(metrics)
#     model_fed = create_DNN()
#     state.model.assign_weights_to(model_fed)
#     sample = tf.nest.map_structure(lambda x: x.numpy(), next(iter(test_set_fed[0])))
#     prediciton_fed = model_fed.predict(sample['x'])
#     print(f"MSE_fed: {mean_squared_error(prediciton_fed, sample['y'])}")
#     print(f"R2_fed: {r2_score(sample['y'], prediciton_fed)}")
#     if IFPLOT==1:
#       time_plot = range(0, 48)
#       plt.figure(figsize=(10, 6))
#       plot_series(time_plot, prediciton_fed, color='red')
#       plot_series(time_plot, sample['y'])



# Distributed Learning Part
xx = [[] for i in range(NUM_AGENTS)]
yy = [[] for i in range(NUM_AGENTS)]
xx_test = [[] for i in range(NUM_AGENTS)]
yy_test = [[] for i in range(NUM_AGENTS)]
for i in range(0, NUM_AGENTS):
  xx[i], yy[i] = create_dataset_central(train_raw[i], time_step)
  xx_test[i], yy_test[i] = create_dataset_central(test_raw[i], time_step)





# # models = [[] for i in range(NUM_AGENTS)]
# # for i in range(NUM_AGENTS):
# #     models[i] = keras.models.load_model('Results/wavenet'+str(i))
# # for i in range(NUM_AGENTS):
# #         # models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
# #         l[i] = models[i].predict(xx_test[i])
# #         print(l[i])

# models = [create_wavenet_model() for i in range(NUM_AGENTS)]
# garph = 'RING'
# l = [[] for i in range(NUM_AGENTS)]
# MSE_cen = [[] for i in range(NUM_AGENTS)]
# R2_cen = [[] for i in range(NUM_AGENTS)]
# adam = keras.optimizers.Adam(learning_rate=0.001)
# for i in range(NUM_AGENTS):
#     # models[i] = create_DNN()
#     models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
#     # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# hist_l = [[] for i in range(NUM_AGENTS)]
# hist_mse = [[] for i in range(NUM_AGENTS)]
# hist_mean = []
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# for steps in range(TRAINING_LOOP):
#     for i in range(NUM_AGENTS):
#         models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
#         l[i] = models[i].predict(xx_test[i])
#         MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
#         hist_l[i].append(l[i])
#         hist_mse[i].append(MSE_cen[i])

#     models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
#     print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
#     hist_mean.append(np.mean(MSE_cen))

# np.savez('Results/wavenet/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
# for i in range(NUM_AGENTS):
#     models[i].save('Results/wavenet/model'+garph+str(i),include_optimizer = True)

# # models = [create_DNN() for i in range(NUM_AGENTS)]
# # models = [create_LSTM_model() for i in range(NUM_AGENTS)]
# # models = [create_CNN_model() for i in range(NUM_AGENTS)]
# models = [create_wavenet_model() for i in range(NUM_AGENTS)]
# garph = 'FULL'
# l = [[] for i in range(NUM_AGENTS)]
# MSE_cen = [[] for i in range(NUM_AGENTS)]
# R2_cen = [[] for i in range(NUM_AGENTS)]
# adam = keras.optimizers.Adam(learning_rate=0.001)
# for i in range(NUM_AGENTS):
#     # models[i] = create_DNN()
#     models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
#     # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# hist_l = [[] for i in range(NUM_AGENTS)]
# hist_mse = [[] for i in range(NUM_AGENTS)]
# hist_mean = []
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# for steps in range(TRAINING_LOOP):
#     for i in range(NUM_AGENTS):
#         models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
#         l[i] = models[i].predict(xx_test[i])
#         MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
#         hist_l[i].append(l[i])
#         hist_mse[i].append(MSE_cen[i])

#     models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
#     print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
#     hist_mean.append(np.mean(MSE_cen))

# np.savez('Results/wavenet/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
# for i in range(NUM_AGENTS):
#     models[i].save('Results/wavenet/model'+garph+str(i),include_optimizer = True)


# models = [create_wavenet_model() for i in range(NUM_AGENTS)]
# garph = 'MS'
# l = [[] for i in range(NUM_AGENTS)]
# MSE_cen = [[] for i in range(NUM_AGENTS)]
# R2_cen = [[] for i in range(NUM_AGENTS)]
# adam = keras.optimizers.Adam(learning_rate=0.001)
# for i in range(NUM_AGENTS):
#     # models[i] = create_DNN()
#     models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
#     # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# hist_l = [[] for i in range(NUM_AGENTS)]
# hist_mse = [[] for i in range(NUM_AGENTS)]
# hist_mean = []
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# for steps in range(TRAINING_LOOP):
#     for i in range(NUM_AGENTS):
#         models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
#         l[i] = models[i].predict(xx_test[i])
#         MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
#         hist_l[i].append(l[i])
#         hist_mse[i].append(MSE_cen[i])

#     models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
#     print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
#     hist_mean.append(np.mean(MSE_cen))

# np.savez('Results/wavenet/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
# for i in range(NUM_AGENTS):
#     models[i].save('Results/wavenet/model'+garph+str(i),include_optimizer = True)


# models = [create_LSTM_model() for i in range(NUM_AGENTS)]
# garph = 'RING'
# l = [[] for i in range(NUM_AGENTS)]
# MSE_cen = [[] for i in range(NUM_AGENTS)]
# R2_cen = [[] for i in range(NUM_AGENTS)]
# adam = keras.optimizers.Adam(learning_rate=0.001)
# for i in range(NUM_AGENTS):
#     # models[i] = create_DNN()
#     models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
#     # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# hist_l = [[] for i in range(NUM_AGENTS)]
# hist_mse = [[] for i in range(NUM_AGENTS)]
# hist_mean = []
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# for steps in range(TRAINING_LOOP):
#     for i in range(NUM_AGENTS):
#         models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
#         l[i] = models[i].predict(xx_test[i])
#         MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
#         hist_l[i].append(l[i])
#         hist_mse[i].append(MSE_cen[i])

#     models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
#     print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
#     hist_mean.append(np.mean(MSE_cen))

# np.savez('Results/LSTM/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
# for i in range(NUM_AGENTS):
#     models[i].save('Results/LSTM/model'+garph+str(i),include_optimizer = True)


# models = [create_LSTM_model() for i in range(NUM_AGENTS)]
# garph = 'FULL'
# l = [[] for i in range(NUM_AGENTS)]
# MSE_cen = [[] for i in range(NUM_AGENTS)]
# R2_cen = [[] for i in range(NUM_AGENTS)]
# adam = keras.optimizers.Adam(learning_rate=0.001)
# for i in range(NUM_AGENTS):
#     # models[i] = create_DNN()
#     models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
#     # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# hist_l = [[] for i in range(NUM_AGENTS)]
# hist_mse = [[] for i in range(NUM_AGENTS)]
# hist_mean = []
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# for steps in range(TRAINING_LOOP):
#     for i in range(NUM_AGENTS):
#         models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
#         l[i] = models[i].predict(xx_test[i])
#         MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
#         hist_l[i].append(l[i])
#         hist_mse[i].append(MSE_cen[i])

#     models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
#     print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
#     hist_mean.append(np.mean(MSE_cen))

# np.savez('Results/LSTM/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
# for i in range(NUM_AGENTS):
#     models[i].save('Results/LSTM/model'+garph+str(i),include_optimizer = True)


# models = [create_LSTM_model() for i in range(NUM_AGENTS)]
# garph = 'MS'
# l = [[] for i in range(NUM_AGENTS)]
# MSE_cen = [[] for i in range(NUM_AGENTS)]
# R2_cen = [[] for i in range(NUM_AGENTS)]
# adam = keras.optimizers.Adam(learning_rate=0.001)
# for i in range(NUM_AGENTS):
#     # models[i] = create_DNN()
#     models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
#     # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# hist_l = [[] for i in range(NUM_AGENTS)]
# hist_mse = [[] for i in range(NUM_AGENTS)]
# hist_mean = []
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# for steps in range(TRAINING_LOOP):
#     for i in range(NUM_AGENTS):
#         models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
#         l[i] = models[i].predict(xx_test[i])
#         MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
#         hist_l[i].append(l[i])
#         hist_mse[i].append(MSE_cen[i])

#     models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
#     print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
#     hist_mean.append(np.mean(MSE_cen))

# np.savez('Results/LSTM/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
# for i in range(NUM_AGENTS):
#     models[i].save('Results/LSTM/model'+garph+str(i),include_optimizer = True)


models = [create_CNN_model() for i in range(NUM_AGENTS)]
garph = 'RING'
l = [[] for i in range(NUM_AGENTS)]
MSE_cen = [[] for i in range(NUM_AGENTS)]
R2_cen = [[] for i in range(NUM_AGENTS)]
adam = keras.optimizers.Adam(learning_rate=0.001)
for i in range(NUM_AGENTS):
    # models[i] = create_DNN()
    models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

hist_l = [[] for i in range(NUM_AGENTS)]
hist_mse = [[] for i in range(NUM_AGENTS)]
hist_mean = []
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
for steps in range(TRAINING_LOOP):
    for i in range(NUM_AGENTS):
        models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
        l[i] = models[i].predict(xx_test[i])
        MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
        hist_l[i].append(l[i])
        hist_mse[i].append(MSE_cen[i])

    models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
    print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
    hist_mean.append(np.mean(MSE_cen))

np.savez('Results/CNN/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
for i in range(NUM_AGENTS):
    models[i].save('Results/CNN/model'+garph+str(i),include_optimizer = True)

models = [create_CNN_model() for i in range(NUM_AGENTS)]
garph = 'FULL'
l = [[] for i in range(NUM_AGENTS)]
MSE_cen = [[] for i in range(NUM_AGENTS)]
R2_cen = [[] for i in range(NUM_AGENTS)]
adam = keras.optimizers.Adam(learning_rate=0.001)
for i in range(NUM_AGENTS):
    # models[i] = create_DNN()
    models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

hist_l = [[] for i in range(NUM_AGENTS)]
hist_mse = [[] for i in range(NUM_AGENTS)]
hist_mean = []
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
for steps in range(TRAINING_LOOP):
    for i in range(NUM_AGENTS):
        models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
        l[i] = models[i].predict(xx_test[i])
        MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
        hist_l[i].append(l[i])
        hist_mse[i].append(MSE_cen[i])

    models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
    print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
    hist_mean.append(np.mean(MSE_cen))

np.savez('Results/CNN/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
for i in range(NUM_AGENTS):
    models[i].save('Results/CNN/model'+garph+str(i),include_optimizer = True)

models = [create_CNN_model() for i in range(NUM_AGENTS)]
garph = 'MS'
l = [[] for i in range(NUM_AGENTS)]
MSE_cen = [[] for i in range(NUM_AGENTS)]
R2_cen = [[] for i in range(NUM_AGENTS)]
adam = keras.optimizers.Adam(learning_rate=0.001)
for i in range(NUM_AGENTS):
    # models[i] = create_DNN()
    models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

hist_l = [[] for i in range(NUM_AGENTS)]
hist_mse = [[] for i in range(NUM_AGENTS)]
hist_mean = []
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
for steps in range(TRAINING_LOOP):
    for i in range(NUM_AGENTS):
        models[i].fit(x=xx[i],y=yy[i],batch_size=BATCHSIZE,epochs=LOCAL_TRAINING_STEP,callbacks=[callback],shuffle=True,verbose=0,)
        l[i] = models[i].predict(xx_test[i])
        MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
        hist_l[i].append(l[i])
        hist_mse[i].append(MSE_cen[i])

    models = DistributedLearning(models,step=CONSENSUS_STEP,Graph=garph)
    print('Steps:',steps,'Average_MSE=', np.mean(MSE_cen))
    hist_mean.append(np.mean(MSE_cen))

np.savez('Results/CNN/results_of_model'+str(NUM_AGENTS)+garph+'.npz',hist_l=hist_l,hist_mse=hist_mse,hist_mean=hist_mean)
for i in range(NUM_AGENTS):
    models[i].save('Results/CNN/model'+garph+str(i),include_optimizer = True)


