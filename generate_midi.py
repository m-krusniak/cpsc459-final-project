#!/usr/bin/env python

import argparse
from process_midi import *
from save_midi import *
from nn_model import *
import tensorflow as tf
import evaluate_inference_machine_filter as eimf
import inference_machine

def generate_im_mlp(model_file, predict_dir, output_dir, drum):
  model = InferenceMachine(drum)
  model.load(model_file)
  for filename in os.listdir(predict_dir):
    (im_pred, im_tp, im_fn, im_fp, im_n_total) = model.evaluate(predict_dir+filename)
    save_song(output_dir + "/im_mlp_pred__" + filename, im_pred)


def generate_ff(model_file, predict_dir, output_dir, drum):

  # Train up a feed-forward model
  model = FeedForward(drum)
  model.load(model_file)

  # Test all songs in test directory
  for filename in os.listdir(predict_dir):
    (test_data, test_targs) = load_song(predict_dir+filename, memory_length=16, drum=drum)
    test_data = np.reshape(test_data, (len(test_data), 63))
    test_targs = np.reshape(test_targs, (len(test_targs), 1))

    (ff_pred, ff_tp, ff_fn, ff_fp, ff_n_total) = model.evaluate(test_data, test_targs)

    save_song(output_dir + "/ff_pred__" + filename, ff_pred)

def generate_im_rf(model_file, predict_dir, output_dir, drum):
  model = InferenceMachine(drum)
  model.load(model_file)
  for filename in os.listdir(predict_dir):
    (im_pred, im_tp, im_fn, im_fp, im_n_total) = model.evaluate(predict_dir+filename)
    save_song(output_dir + "/im_rf_pred__" + filename, im_pred)


if __name__ == '__main__':
  # parse command line args
  parser = argparse.ArgumentParser(description="Generate predicted MIDI output from inputs")
  parser.add_argument('model_type', help="type of model: one of im_mlp, im_rf, or ff", type=str)
  parser.add_argument('model_file', help="h5 file in which model is stored")
  parser.add_argument('input_dir', help="path of folder containing MIDI file from which to predict", type=str)
  parser.add_argument('output_dir', help="path of folder into which to place predictions", type=str)
  args = parser.parse_args()

  if args.model_type == "im_mlp": generate_im_mlp(args.model_file, args.input_dir, args.output_dir, 38)
  if args.model_type == "im_rf": generate_im_rf(args.model_file, args.input_dir, args.output_dir, 38)
  if args.model_type == "ff": generate_ff(args.model_file, args.input_dir, args.output_dir, 38)