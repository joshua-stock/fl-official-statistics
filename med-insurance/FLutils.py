import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import r2_score
# r2 only available in tf-nightly
#https://www.tensorflow.org/api_docs/python/tf/keras/metrics/R2Score

import matplotlib.pyplot as plt
from itertools import product
from math import floor
import time

import tensorflow as tf
import tensorflow_federated as tff
#import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.callbacks import CSVLogger


def load_df(paths: list[str]) -> pd.DataFrame:

  """Reads a dataset from a list of possible paths to a csv-file.

  Args:
    paths (list[str]): Possible locations of the csv-file.

  Returns:
    pd.DataFrame: The loaded dataset.
  """

  df = pd.DataFrame()

  if isinstance(paths, str): paths = [paths]
    
  for path in paths:
    try:
      df = pd.read_csv(path, index_col = 0)
      print("loaded data from {}".format(path))
      if len(df) != 0: break
    except Exception as ex:
      print("{} in ".format(type(ex).__name__), path)

  return df

def prep_fed_train(X_train: pd.DataFrame, y_train: pd.DataFrame):
  """Converts training data to Tensor Object.
  
  See https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#preprocessing_the_input_data

  Args:
      X_train (pd.DataFrame): Training features.
      y_train (pd.DataFrame): Training target.

  Returns:
      A `Tensor` based on `X_test`, `y_test`.
  """

  return tf.data.Dataset.from_tensor_slices((
    tf.convert_to_tensor(X_train), 
    tf.convert_to_tensor(y_train)
    ))

def prep_fed_test(X_test: pd.DataFrame, y_test: pd.DataFrame):
  """Converts test data to Tensor Object.

  See https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#preprocessing_the_input_data

  Args:
      X_test (pd.DataFrame): Test features.
      y_test (pd.DataFrame): Test target.

  Returns:
      A `Tensor` based on `X_test`, `y_test`.
  """
  return tf.data.Dataset.from_tensor_slices((
    tf.convert_to_tensor(np.expand_dims(X_test, axis=0)), 
    tf.convert_to_tensor(np.expand_dims(y_test, axis=0))
    )) 

def create_keras_model(
    nfeatures: int = 9,
    units: list[int] = [40, 40, 20], 
    activations: list[str] = ['relu'] * 3, 
    compile: bool = True,
    loss = 'mean_squared_error',
    optimizer = tf.optimizers.legacy.Adam(learning_rate = .05),
    metrics = ["mae", 'mean_squared_error', r2_score], 
    run_eagerly = True
    ):
  
  """Construct a fully connected neural network and compile it.
  
  Parameters
  ------------
  nfeatures: int, optional
    Number of input features. Default is 9.
  units: list of int, optional
    List of number of units of the hidden dense layers. The length of ``units`` defines the number of hidden layers. Default are 3 layers with 40, 40 an 20 units, respectively.
  activations: list of str, optional
    List of activation functions used in the hidden layers.
  loss: str, optional
    Used loss function for compiling.
  optimizer: keras.optimizers, optional
    Used optimizer for compiling.
  metrics: list of str or sklearn.metrics
    List of metrics for compiling.
  run_eagerly: bool
    Parameter for compiling

  Return
  ------------
    model: keras.engine.sequential.Sequential
      Keras sequential fully connected neural network. Already compiled.
  """
  
  # construct model
  model = Sequential()
  model.add(InputLayer(input_shape = [nfeatures]))
  for ind in range(len(units)):
    model.add(Dense(
      units = units[ind], 
      activation = activations[ind]
      ))
  model.add(Dense(1))
  
  # compile model
  if compile == True:
    model.compile(
      loss = loss,
      optimizer = optimizer,
      metrics = metrics,
      run_eagerly = run_eagerly
    )

  return model

def model_fn(keras_creator,
  loss: tf.keras.losses = tf.keras.losses.MeanAbsoluteError()
  #,metrics = [tf.keras.metrics.MeanAbsoluteError()]
  ) -> tff.learning.models:
  """  Wrap a Keras model as Tensorflow Federated model. 

  Cf. https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#creating_a_model_with_keras


  Args:
      keras_creator: Function returning a keras.engine.sequential.Sequential.
      loss (tf.keras.losses, optional): loss used for optimization. Defaults to tf.keras.losses.MeanAbsoluteError().
  
  Returns:
    
  """


  def _model():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    
    #keras_model = create_keras_model(
    #    nfeatures = nfeatures, compile = False#, **kwargs
    #    )
    
    keras_model = keras_creator()

    return tff.learning.models.from_keras_model(
      keras_model,
      input_spec = (
        tf.TensorSpec((None, keras_model.input.shape[1]
        ), dtype = tf.float64),
        tf.TensorSpec((None,),           dtype = tf.float64)
      ), loss = loss, 
      metrics =  [
        tf.keras.metrics.MeanAbsoluteError()
        , tf.keras.metrics.MeanSquaredError()
        #, tfa.metrics.RSquare()
        #, tf.keras.metrics.R2Score() # only available in tf-nightly
        ]
    )

  return _model

def train_model(model, X_train, y_train,
    epochs           = 100,
    batch_size       = 128,
    shuffle          = True,
    validation_split = 0.2,
    verbose          = 0,
    output_msr       = 'loss',
    seed             = 42,
    **kwargs
    ):
  
  """Compile and train a Keras neural network.
  
  For additional arguments see https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit.

  Parameters
  ------------
  model: keras.engine.sequential.Sequential
  X_train: dataFrame
  y_train: dataFrame
  shuffle: bool
  epochs: int
  validation_split: float
  verbose: int
    verbose of model.fit(...)
  output_msr: str
    measure for custom output.
  batch_size: int
    batch_size of model.fit(...)   

  Return
  ------------
    hist: keras.callbacks.History
      History of model.fit(...)
  """

  # fit with custom verbose
  starttime = time.time()
  
  if seed != None: tf.keras.utils.set_random_seed(seed)

  hist = model.fit(
    X_train, 
    y_train,
    batch_size = batch_size, 
    shuffle    = shuffle,
    validation_split = validation_split,
    epochs     = epochs,
    verbose    = verbose, 
    **kwargs
  )
  
  if verbose != 0:
    print(
      "%s  = %.4f," % (output_msr, hist.history[output_msr][-1]),
      "time = %.1f sec" % ((time.time() - starttime))
    )
  
  return hist

def train_fed(model, train_data: list,
    eval_data: list = None,
    client_optimizer = lambda: tf.optimizers.Adam(learning_rate = .05),
    server_optimizer = lambda: tf.optimizers.Adam(learning_rate = .05),
    NUM_ROUNDS: int = 50,
    NUM_EPOCHS: int = 50,
    BATCH_SIZE: int = 128,
    SHUFFLE_BUFFER: int = 20,
    PREFETCH_BUFFER: int = 5,
    SEED: int = 42,
    verbose: bool = True
    ) -> dict:
    """Train a keras model with distributed data by federated learning.

    cf. https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#evaluation


    Args:
        model: keras_model as archetype for the federated trained model. 
        train_data (list): List of training datasets (as tensors)
        eval_data (list, optional): List of evaluation datasets (as tensors). Defaults to None.
        client_optimizer (optional): Optimizing algorithm for the learning of each client. Defaults to lambda:tf.optimizers.Adam(learning_rate = .05).
        server_optimizer (optional): Optimizing algorithm for the server. Defaults to lambda:tf.optimizers.Adam(learning_rate = .05).
        NUM_ROUNDS (int, optional): Number of federated learning rounds. In each round the server sends each client a model to improve it. Defaults to 50.
        NUM_EPOCHS (int, optional): Number of epochs per client per round. Defaults to 50.
        BATCH_SIZE (int, optional): Number of samples per learning update of each client. Defaults to 128.
        SHUFFLE_BUFFER (int, optional): The dataset is shuffled. `SHUFFLE_BUFFER` is the number of samples which are randomized. Defaults to 20.
        PREFETCH_BUFFER (int, optional): See https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#prefetch. Defaults to 5.
        SEED (int, optional): SEED for all randomized calculations. Defaults to 42.
        verbose (bool, optional): Status messages? Defaults to True.

    Returns:
        dict: Dict containing the resulting learning_process, history, state. 
    """


    

    # prep the data
    train_data = [
        data.
            repeat(NUM_EPOCHS).
            shuffle(SHUFFLE_BUFFER, seed = SEED).
            batch(BATCH_SIZE).
            prefetch(PREFETCH_BUFFER)

      for data in train_data]
    
    # initialize the process (and evalation if any)

    process = tff.learning.algorithms.build_weighted_fed_avg(
        model,
        client_optimizer_fn = client_optimizer,
        server_optimizer_fn = server_optimizer)
    

    state = process.initialize()
    hist  = []

    if eval_data != None:
        eval_process  = tff.learning.algorithms.build_fed_eval(model)
        eval_state    = eval_process.initialize()
        model_weights = process.get_model_weights(state)
        eval_state    = eval_process.set_model_weights(eval_state, model_weights)
        _, perf_eval  = eval_process.next(eval_state, eval_data)

    # training
    for round in range(NUM_ROUNDS):
        
        # calculate the evaluation performance (before updating the model!)
        # Note:
        # - 'process' generally reflect the performance of the model at the beginning of the round, 
        # - if not calculated first, the evaluation metrics will always be one step ahead
        if eval_data != None:
            
            model_weights = process.get_model_weights(state)
            eval_state    = eval_process.set_model_weights(eval_state, model_weights)
            _, perf_eval  = eval_process.next(eval_state, eval_data)
            perf_eval = dict(perf_eval['client_work']['eval']['current_round_metrics'].items())
            
        
        # update FL process
        if SEED != None: tf.keras.utils.set_random_seed(SEED)
        state, perf = process.next(state, train_data)
        perf = dict(perf['client_work']['train'].items())

        # generate outputs
        if verbose: 
            print('====== ROUND {:2d} / {} ======'.format(round + 1, NUM_ROUNDS))
            print('TRAIN: {}'.format(perf))
            if eval_data != None: 
                print('EVAL:  {}'.format(perf_eval))

        
        # combine perf and perf_eval
        if eval_data != None:
            # rename keys in perf_eval
            perf_eval = {'val_' + key : val for key, val in  perf_eval.items()}
            # merge perf and perf_eval (> Python 3.9)
            perf = perf | perf_eval
        
        # extend history
        hist.append(perf)

    return {'process': process, 'history': hist, 'state': state}

def plot_hist(history, 
  msr: str = "r2_score", 
  title: str = None,
  ylim : list[float] = [0.5, 0.9], 
  savepath: str = None):
  """Plots the performance of keras histories.

  Args:
      history: Keras History.
      msr (str, optional): Name of a performance measure. Defaults to "r2_score".
      title (str, optional): Title of the plot. Defaults to None.
      ylim (list[float], optional): y limits. Defaults to [0.5, 0.9].
      savepath (str, optional): Save the plot to the path. Defaults to None.
  """
  
  
    
  if not isinstance(history, list): history = [history]

  
  y = np.array([hist[msr] for hist in history]).transpose()
  yval = np.array([hist['val_' + msr] for hist in history]).transpose()
  plt.plot(y, color = 'blue', alpha = .2)
  plt.plot(yval, color = 'orange', alpha = .2)
  plt.plot(np.quantile(y,.5, axis = 1), label = 'Training', color = 'blue')
  plt.plot(np.quantile(yval,.5, axis = 1), label = 'Evaluation', color = 'orange')
  plt.ylim(ylim)
  plt.xlabel("epochs")
  plt.ylabel(msr)
  if title != None: plt.suptitle(title)
  plt.title('Training performance')
  plt.legend()
  
  if savepath != None: plt.savefig(savepath)
  plt.show()

def test_model(model, X_test, y_test, 
               verbose = False):
  """
  Parameters
  ------------
  model: keras.engine.sequential.Sequential
    Fitted model.
  X_test, y_test: dataFrame
    Test data.
  verbose: bool
    Output control.

  Output
  ------------
  perf: list of float
    The test performances.
  """

  start = time.time() 
  perf  = model.evaluate(X_test, y_test, verbose = 0)[1:]

  if verbose: print('time - test: %.2f' % (time.time() - start / 60))
  
  return perf