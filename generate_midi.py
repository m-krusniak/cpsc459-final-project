#!/usr/bin/env python

import argparse
from process_midi import *
from save_midi import *
from nn_model import *
import tensorflow as tf
import evaluate_inference_machine_filter as eimf
import inference_machine

def generate(train_dir, predict_dir, output_dir, drum):

  # Train up a feed-forward model
  (train_data, train_targs, n_train) = load_all_songs(train_dir, memory_length=16, drum=drum)
  ff_model = train(*separate_train_val(train_data, train_targs))

  # Train up a perceptron-backed inference machine
  inference_machine.main(train_dir, n_train, 4)

  # Test all songs in test directory
  for filename in os.listdir(predict_dir):
    (test_data, test_targs) = load_song(predict_dir+filename, memory_length=16, drum=drum)
    test_data = np.reshape(test_data, (len(test_data), 63))
    test_targs = np.reshape(test_targs, (len(test_targs), 1))

    (im_pred, im_tp, im_fn, im_fp, im_n_total) = eimf.main(predict_dir+filename, 'filter_fn.h5')
    (ff_pred, ff_tp, ff_fn, ff_fp, ff_n_total) = evaluate(ff_model, test_data, test_targs)

    save_song(output_dir + "/im_pred__" + filename, im_pred)
    save_song(output_dir + "/ff_pred__" + filename, im_pred)


if __name__ == '__main__':
  # parse command line args
  parser = argparse.ArgumentParser(description="Generate predicted MIDI output from inputs")
  parser.add_argument('train_dir', help="path of folder containing MIDI data on which to train", type=str)
  parser.add_argument('predict_dir', help="path of folder containing MIDI file from which to predict", type=str)
  parser.add_argument('output_dir', help="path of folder into which to place predictions", type=str)
  args = parser.parse_args()

  # run main()
  print "Analyzing..."
  generate(args.train_dir, args.predict_dir, args.output_dir, 38)