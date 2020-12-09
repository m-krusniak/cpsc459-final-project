#!/usr/bin/env python

import argparse
from process_midi import *
from save_midi import *
from nn_model import *
import tensorflow as tf
import evaluate_inference_machine_filter as eimf
import inference_machine


def print_results(genre_name, model_name, tp, fn, fp, total):
  log = open("genre_output.txt", "a")

  # Print results
  log.write("\n\nRESULTS FOR GENRE '%s' AND MODEL '%s'" % (genre_name, model_name) + "\n")
  log.write("Total tests: " + str(total) + "\n")
  log.write("False positive: " + str(fp) + "\n")
  log.write("False negative: " + str(fn) + "\n")
  log.write("True positive: " + str(tp) + "\n")

  if tp + fp == 0: 
    log.write("  ERROR: All outputs were zero.\n\n")
    return

  if tp + fn == 0: 
    log.write("  ERROR: No true positives or false negatives.\n\n")
    return
  

  precision = tp / float(tp + fp)
  recall = tp / float(tp + fn)
  f1 = (2 * precision * recall) / float(precision + recall)

  pos = tp+fn
  true = (total-fn-fp)

  log.write("%d correct out of %d total (%2.3f accuracy)\n" % (true, total, 100*(true)/float(total)))
  log.write("  Precision: %.3f    Recall: %.3f    F1 score: %.3f\n" % (precision, recall, f1))
  log.write("  Skew: %.3f%% of targets are 0\n" % ((1 - float(pos)/total) * 100))

def analyze(genres_dir, drum):

  genres = os.listdir(genres_dir + "/train/") 

  im_tp_all = im_fp_all = im_fn_all = im_n_total_all = 0
  ff_tp_all = ff_fp_all = ff_fn_all = ff_n_total_all = 0

  for g in genres:

    train_dir = genres_dir + '/train/%s/'%g
    test_dir = genres_dir + '/test/%s/'%g

    # Train up a feed-forward model
    (train_data, train_targs, n_train) = load_all_songs(train_dir, memory_length=16, drum=drum)
    ff_model = train(*separate_train_val(train_data, train_targs))

    # Train up a perceptron-backed inference machine
    inference_machine.main(train_dir, n_train, 4)

    # Test all songs in test directory
    for filename in os.listdir(test_dir):
      (test_data, test_targs) = load_song(test_dir+filename, memory_length=16, drum=drum)
      test_data = np.reshape(test_data, (len(test_data), 63))
      test_targs = np.reshape(test_targs, (len(test_targs), 1))

      (im_pred, im_tp, im_fn, im_fp, im_n_total) = eimf.main(test_dir+filename, 'filter_fn.h5')
      (ff_pred, ff_tp, ff_fn, ff_fp, ff_n_total) = evaluate(ff_model, test_data, test_targs)

      ff_fn_all += ff_fn; ff_tp_all += ff_tp; ff_fp_all += ff_fp; ff_n_total_all += ff_n_total
      im_fn_all += im_fn; im_tp_all += im_tp; im_fp_all += im_fp; im_n_total_all += im_n_total

    print_results(g, "perceptron-im", im_tp_all, im_fn_all, im_fp_all, im_n_total_all)
    print_results(g, "feed-forward" , ff_tp_all, ff_fn_all, ff_fp_all, ff_n_total_all)

if __name__ == '__main__':
  # parse command line args
  parser = argparse.ArgumentParser(description="Run available models on a variety of genres")
  parser.add_argument('genres_dir', help="path of folder containing train/ and test/, each containing a folder for each genre", type=str)
  # parser.add_optional_argument('drum', help="MIDI drum for the analysis (e.g., 38 is bass, 42 is snare)", type=int, default=38)
  args = parser.parse_args()

  # run main()
  print "Analyzing..."
  analyze(args.genres_dir, 38)