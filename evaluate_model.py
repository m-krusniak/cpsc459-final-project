#!/usr/bin/env python

import argparse
from process_midi import *
from save_midi import *
from nn_model import *
import tensorflow as tf
from InferenceMachine import InferenceMachine
from FeedForward import FeedForward

def print_results(tp, fn, fp, total):

  # Print results
  print "\n\nRESULTS:"
  print "Total tests: " + str(total)
  print "False positive: " + str(fp)
  print "False negative: " + str(fn)
  print "True positive: " + str(tp)

  if tp + fp == 0: 
    print "  ERROR: All outputs were zero.\n\n"
    return

  if tp + fn == 0: 
    print "  ERROR: No true positives or false negatives.\n\n"
    return

  precision = tp / float(tp + fp)
  recall = tp / float(tp + fn)
  f1 = (2 * precision * recall) / float(precision + recall)
  pos = tp+fn
  true = (total-fn-fp)

  print "%d correct out of %d total (%2.3f accuracy)" % (true, total, 100*(true)/float(total))
  print "  Precision: %.3f    Recall: %.3f    F1 score: %.3f" % (precision, recall, f1)
  print "  Skew: %.3f%% of targets are 0" % ((1 - float(pos)/total) * 100)


def evaluate_im(model_file, predict_file, model_type, drum):
  model = InferenceMachine(model_type, drum)
  model.load(model_file)
  (pred, tp, fn, fp, n_total) = model.evaluate(predict_file)
  print_results(tp, fn, fp, n_total)

def evaluate_ff(model_file, predict_file, drum):

  # Train up a feed-forward model
  model = FeedForward(drum)
  model.load(model_file)

  (test_data, test_targs) = load_song(predict_file, memory_length=16, drum=drum)

  test_data = np.reshape(test_data, (len(test_data), 63))
  test_targs = np.reshape(test_targs, (len(test_targs), 1))

  (pred, tp, fn, fp, n_total) = model.evaluate(test_data, test_targs)
  print_results(tp, fn, fp, n_total)


if __name__ == '__main__':
  # parse command line args
  parser = argparse.ArgumentParser(description="Evaluate a model's predictions on MIDI files")
  parser.add_argument('model_type', help="type of model: one of im_mlp, im_rf, or ff", type=str)
  parser.add_argument('model_file', help="h5 file in which model is stored")
  parser.add_argument('input_file', help="path of a file to predict", type=str)
  parser.add_argument('--drum', help="MIDI drum number", type=int, default=36)
  args = parser.parse_args()

  if args.model_type == "im_mlp": evaluate_im(args.model_file, args.input_file, "MLP", args.drum)
  if args.model_type == "im_rf": evaluate_im(args.model_file, args.input_file, "RF", args.drum)
  if args.model_type == "ff": evaluate_ff(args.model_file, args.input_file, args.drum)