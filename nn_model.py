import mido
import collections
from playsound import playsound
import numpy as np
import datetime
import os
import tensorflow as tf


def train(train_data, train_targs, val_data, val_targs):

  # STEP 1: Alert
  print "STARTING TRAINING; shape" + str(train_data.shape)

  # STEP 2: Create the model
  l0_input = tf.keras.layers.Input(shape=train_data[0].shape)
  l1_dense = tf.keras.layers.Dense(64, use_bias=True, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=1))(l0_input)
  l2_dense = tf.keras.layers.Dense(128, use_bias=True, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=2))(l1_dense)
  l3_dense = tf.keras.layers.Dense(1, use_bias=False,activation='sigmoid',
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=3))(l2_dense)
  model = tf.keras.models.Model(inputs=l0_input, outputs=l3_dense, name="rhythm_model")

  # compile the model: define optimizer, loss, and metrics
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                loss='binary_crossentropy', metrics=['binary_accuracy'])

  # STEP 3: Actually do the training

  # tensorboard callback
  logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
  tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)
  checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir, 'rhythm_weights.h5'),
                                                            monitor='binary_accuracy',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            save_freq=1)

  # do training for the specified number of epochs and with the given batch size
  model.fit(train_data, train_targs, epochs=3, batch_size=4,
            validation_data=(val_data, val_targs),
            callbacks=[tbCallBack, checkpointCallBack])

  return model

def evaluate(model, test_data, test_targs):
  # evaluate
  predicted_targs = model.predict(test_data, batch_size=4)
  predicted_targs = [1 if a > 0.2 else 0 for a in predicted_targs]

  n_correct = 0
  n_total = 0
  tp = 0
  fn = 0
  fp = 0
  for p in zip(test_targs, predicted_targs): 
    if p[0] == p[1]:
      n_correct += 1
    if p[0] == 1 and p[1] == 1:
      tp += 1
    if p[0] == 1 and p[1] == 0:
      fn += 1
    if p[0] == 0 and p[1] == 1:
      fp += 1
    n_total += 1

  return (predicted_targs, tp, fn, fp, n_total)


def load(file):
  return tf.keras.models.load_model(file, compile=True)