import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
import yfinance as yf

def getData():
    stocks = yf.Tickers(['FB','AMZN','NFLX','GOOGL','AAPL'])
    time_period = 'max'#can change this
    df = stocks.history(period = time_period)
    df = df['Close']
    df = df.reset_index()
    df = df[df.notna().all(axis = 1)]
    df.to_csv("prices.csv", index=False)
    # df = pd.read_csv("prices.csv")
    df["Date"] = pd.to_datetime(df["Date"]) # convert string to datetime format
    stocks = ['FB','AMZN','NFLX','GOOGL','AAPL']
    scaler = StandardScaler()
    df[stocks] = scaler.fit_transform(df[stocks])
    train_size = int(df.shape[0]*0.80)
    val_size = int(df.shape[0]*0.90)
    a = df.iloc[:train_size, :]
    b = df.iloc[train_size:val_size, :]
    c = df.iloc[val_size:, :]
    a=a.drop("Date", axis =1)
    b=b.drop("Date", axis =1)
    c=c.drop("Date", axis =1)
    # dfRecommendations = pd.read_csv("recommendations.csv")# Read from cache to reduce time taken
    stocks = ['FB','AMZN','NFLX','GOOGL','AAPL']
    masterlist = []
    for each in stocks:
        x = yf.Ticker(each)
        y = x.recommendations
        y['stock_symbol'] = each
        y = y.reset_index()
        masterlist.append(y)
    dfRecommendations = pd.concat(masterlist,axis= 0)
    dfRecommendations["Date"] = pd.to_datetime(dfRecommendations["Date"]) # convert string to datetime format
    return df, a, b, c, dfRecommendations

df, a, b, c, dfRecommendations = getData()    

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=a, val_df=b, test_df=c, # a, b, c are train_df, val_df, test_df respectively
               label_columns=None):
    # Store the raw data.
    self.train_df = a
    self.val_df = b
    self.test_df = c

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels
def plot(self, model=None, plot_col='FB', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(25, 15))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(3, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]', size = 18)
      plt.xticks(size = 16)
      plt.yticks(size = 16)
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
              label='Inputs', marker='.', zorder=-10)
      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue
      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
      if n == 0:
        plt.legend(prop={'size': 20})
    plt.xlabel('Time (1 day per timestep)', size = 18)

WindowGenerator.split_window = split_window
WindowGenerator.plot = plot

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=512,)
  ds = ds.map(self.split_window)
  return ds

multi_val_performance = {}
multi_performance = {}
WindowGenerator.make_dataset = make_dataset

def get_model():
  return tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

IN_STEPS = 180 # approximately 6 months
OUT_STEPS = 30 # approximately 1 month
MAX_EPOCHS = 500
num_features = 5

for stock in ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL']:
    multi_window = WindowGenerator(input_width=180,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS,
                                   label_columns=[stock])  
    data = multi_window
    model = get_model()
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
    print('Input shape:', data.example[0].shape)
    print('Output shape:', model(data.example[0]).shape)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=2,
                                                        mode='min')

    model.build((None, 1, num_features))
    history = model.fit(data.train, epochs=MAX_EPOCHS,
                        validation_data=data.val,
                        callbacks=[early_stopping],
                        verbose = 1)
    model.save("C:\\Users\\shyma\\OneDrive - Nanyang Technological University\\School\\BC3409 - AI in A&F\\Group Project\\Chatbot\\linear{}".format(stock))
    print("Completed: {}".format(stock))
    