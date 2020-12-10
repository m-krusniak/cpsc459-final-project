#!/usr/bin/env python

import argparse
from process_midi import *
from save_midi import *
from nn_model import *
import tensorflow as tf
import evaluate_inference_machine_filter as eimf
from InferenceMachine import InferenceMachine
from FeedForward import FeedForward

def make_ff(train_dir, model_file, drum):

  # Train up a feed-forward model
  (train_data, train_targs, n_train) = load_all_songs(train_dir, memory_length=16, drum=drum)
  model = FeedForward(drum)
  model.train(*separate_train_val(train_data, train_targs))
  model.export(model_file)


def make_im(train_dir, model_file, drum, machine_type):

  # Train up the inference machine
  model = InferenceMachine(machine_type, drum=drum)
  n = len(os.listdir(train_dir))
  model.train(train_dir, n, 4)
  model.export(model_file)


if __name__ == '__main__':
  # parse command line args
  parser = argparse.ArgumentParser(description="Train a rhythm model")
  parser.add_argument('model_type', help="type of model: one of im_mlp, im_rf, or ff", type=str)
  parser.add_argument('train_dir', help="path of folder containing training MIDI data", type=str)
  parser.add_argument('model_file', help="location at which to save the trained model", type=str)
  args = parser.parse_args()

  if args.model_type == "ff": make_ff(args.train_dir, args.model_file, 36)
  if args.model_type == "im_mlp": make_im(args.train_dir, args.model_file, 36, 'MLP')
  if args.model_type == "im_rf": make_im(args.train_dir, args.model_file, 36, 'RF')