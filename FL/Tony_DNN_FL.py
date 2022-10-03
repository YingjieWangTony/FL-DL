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
      batch_size).map(batch_format_fn).prefetch(prefetch_buffer)

def preprocess_test(dataset):
  def batch_format_fn(x_d, y_d):
    return OrderedDict(
        x=x_d,
        y=tf.reshape(y_d, [-1, 1])
    )
  return dataset.batch(batch_size).map(batch_format_fn).prefetch(prefetch_buffer)

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

def Consensus(models,step=10,Graph='RING'):
    NUM_AGENTS = len(models)
    if Graph=='RING':
        L =


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
    consumption=np.load('consumption.npy')


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
batch_size = 48
shuffle_buffer = 100
prefetch_buffer = 10
preprocess_example = preprocess_train(example_dataset)

train_set_fed = [preprocess_train(train_data_fed[i]) for i in range(len(train_raw))]
test_set_fed = [preprocess_test(test_data_fed[i]) for i in range(len(test_raw))]
train_set_central = train_set_fed[0]
test_set_central = test_set_fed[0]

# Creating Models
if FL_STEPS==1:
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: keras.optimizers.Adam(0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1)
    )

    logdir = "/tmp/logs/scalars/training"
    summary_writer = tf.summary.create_file_writer(logdir)
    state = iterative_process.initialize()
    num_rounds = 10
    fed_metrics = [[] for i in range(0, num_rounds)]

    start_time = time.time()
    for i in range(0, num_rounds):
      state, metrics = iterative_process.next(state, train_set_fed)
      fed_metrics[i] = metrics
      print('round {:2d}, metrics={}'.format(i+1, metrics))

    end_time = time.time()
    train_time = end_time - start_time
    print(train_time)

    evaluation = tff.learning.build_federated_evaluation(model_fn)
    metrics = evaluation(state.model, test_set_fed)
    print(metrics)
    model_fed = create_DNN()
    state.model.assign_weights_to(model_fed)
    sample = tf.nest.map_structure(lambda x: x.numpy(), next(iter(test_set_fed[0])))
    prediciton_fed = model_fed.predict(sample['x'])
    print(f"MSE_fed: {mean_squared_error(prediciton_fed, sample['y'])}")
    print(f"R2_fed: {r2_score(sample['y'], prediciton_fed)}")
    if IFPLOT==1:
      time_plot = range(0, 48)
      plt.figure(figsize=(10, 6))
      plot_series(time_plot, prediciton_fed, color='red')
      plot_series(time_plot, sample['y'])



# Distributed Learning Part
NUM_AGENTS = 4
TRAINING_LOOP = 10
xx = [[] for i in range(30)]
yy = [[] for i in range(30)]
xx_test = [[] for i in range(30)]
yy_test = [[] for i in range(30)]
for i in range(0, 30):
  xx[i], yy[i] = create_dataset_central(train_raw[i], time_step)
  xx_test[i], yy_test[i] = create_dataset_central(test_raw[i], time_step)

models = [create_DNN() for i in range(NUM_AGENTS)]
adam = keras.optimizers.Adam(learning_rate=0.001)
for i in range(NUM_AGENTS):
    models[i] = create_DNN()
    models[i].compile(loss='mse', optimizer=adam, metrics=['mse'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

l = [[] for i in range(NUM_AGENTS)]
MSE_cen = [[] for i in range(NUM_AGENTS)]
R2_cen = [[] for i in range(NUM_AGENTS)]

for steps in range(TRAINING_LOOP):
    for i in range(NUM_AGENTS):
        models[i].fit(x=xx[i],y=yy[i],batch_size=batch_size,epochs=1,callbacks=[callback],shuffle=True,verbose=0,)
        l[i] = models[i].predict(xx_test[i])
        MSE_cen[i] = mean_squared_error(yy_test[i], l[i])

    models = Consensus(models,step=10)

    # for i in range(NUM_AGENTS):
    #     l[i] = models[i].predict(xx_test[i])
    #     MSE_cen[i] = mean_squared_error(yy_test[i], l[i])
    print('Average_MSE=', np.mean(MSE_cen))




