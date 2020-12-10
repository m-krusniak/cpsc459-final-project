#!/usr/bin/env python

import argparse
from process_midi import *
from save_midi import *
from nn_model import *
import tensorflow as tf
from InferenceMachine import InferenceMachine
from FeedForward import FeedForward


def print_results(genre_name, model_name, tp, fn, fp, total, output_dir):
  log = open(output_dir + "/genre_output.txt", "a")

  # Print results
  log.write("\n\nRESULTS FOR GENRE '%s' AND MODEL '%s'" % (genre_name, model_name) + "\n")
  log.write("Total tests: " + str(total) + "\n")
  log.write("False positive: " + str(fp) + "\n")
  log.write("False negative: " + str(fn) + "\n")
  log.write("True positive: " + str(tp) + "\n")

  if tp + fp == 0: 
    log.write("  ERROR: All outputs were zero.\n\n")
    return

  if tp == 0:
    log.write("  ERROR: No true positives.\n\n")
    return

  precision = tp / float(tp + fp)
  recall = tp / float(tp + fn)
  f1 = (2 * precision * recall) / float(precision + recall)
  pos = tp+fn
  true = (total-fn-fp)

  log.write("%d correct out of %d total (%2.3f accuracy)\n" % (true, total, 100*(true)/float(total)))
  log.write("  Precision: %.3f    Recall: %.3f    F1 score: %.3f\n" % (precision, recall, f1))
  log.write("  Skew: %.3f%% of targets are 0\n" % ((1 - float(pos)/total) * 100))

def analyze_ff(genres_dir, output_dir, drum):

  genres = os.listdir(genres_dir + "/train/") 

  tp_all = fp_all = fn_all = n_total_all = 0

  for g in genres:

    train_dir = genres_dir + '/train/%s/'%g
    test_dir = genres_dir + '/test/%s/'%g
    model_file = output_dir + "/ff_model_%s.h5"%g

    # Train up a feed-forward model
    (train_data, train_targs, n_train) = load_all_songs(train_dir, memory_length=16, drum=drum)
    model = FeedForward(drum)
    model.train(*separate_train_val(train_data, train_targs))
    model.export(model_file)

    # Test all songs in test directory
    for filename in os.listdir(test_dir):
      (test_data, test_targs) = load_song(test_dir+filename, memory_length=16, drum=drum)
      test_data = np.reshape(test_data, (len(test_data), 63))
      test_targs = np.reshape(test_targs, (len(test_targs), 1))

      # An invalid test song (e.g., wrong time signature) will return empty data
      if len(test_targs) == 0 or len(test_data) == 0: continue

      (pred, tp, fn, fp, n_total) = model.evaluate(test_data, test_targs)

      fn_all += fn
      tp_all += tp
      fp_all += fp
      n_total_all += n_total

    print_results(g, "feed-forward", tp_all, fn_all, fp_all, n_total_all, output_dir)


def analyze_im(genres_dir, output_dir, drum, machine_type):

  genres = os.listdir(genres_dir + "/train/") 

  tp_all = fp_all = fn_all = n_total_all = 0

  for g in genres:

    train_dir = genres_dir + '/train/%s/'%g
    test_dir = genres_dir + '/test/%s/'%g
    model_file = output_dir + "/im-%s_model_%s.h5"%(machine_type, g)

    # Train up the inference machine
    model = InferenceMachine(machine_type, drum=drum)
    n = len(os.listdir(train_dir))
    model.train(train_dir, n, 4)
    model.export(model_file)

    # Test all songs in test directory
    for filename in os.listdir(test_dir):

      (pred, tp, fn, fp, n_total) = model.evaluate(test_dir+filename)

      fn_all += fn
      tp_all += tp
      fp_all += fp
      n_total_all += n_total

    print_results(g, "inference-machine-"+machine_type , tp_all, fn_all, fp_all, n_total_all, output_dir)


if __name__ == '__main__':
  # parse command line args
  parser = argparse.ArgumentParser(description="Run available models on a variety of genres")
  parser.add_argument('model_type', help="type of model: one of im_mlp, im_rf, or ff", type=str)
  parser.add_argument('genres_dir', help="path of folder containing train/ and test/, each containing a folder for each genre", type=str)
  parser.add_argument('output_dir', help="path of folder in which to place models and output log", type=str)
  parser.add_argument('--drum', help="MIDI drum number", type=int, default=36)
  args = parser.parse_args()

  if args.model_type == "ff": analyze_ff(args.genres_dir, args.output_dir, args.drum)
  if args.model_type == "im_mlp": analyze_im(args.genres_dir, args.output_dir, args.drum, 'MLP')
  if args.model_type == "im_rf": analyze_im(args.genres_dir, args.output_dir, args.drum, 'RF')