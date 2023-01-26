# Only need to change accordingly in the "model_fn()"

def create_LSTM_model():
  return tf.keras.models.Sequential([
      keras.layers.LSTM(32, input_shape=(time_step, 1), kernel_regularizer='l2'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(1),
  ])
  
def create_CNN_model():
  return tf.keras.models.Sequential([
      keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape = [48, 1], kernel_regularizer='l2'),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(1),
  ])
  
  # This one is WaveNet, a speciall NN based on CNN
def create_CNN_model():
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
