#!/usr/bin/env python

import argparse
import numpy as np
import os
import sys
import tensorflow as tf

from process_midi import *


def main(songpath, modelpath):

    # load song
    hints, obs = load_song(songpath)

    obs = np.array(obs)
    # load model, if path specified via command-line
    # otherwise, need to train (todo)
    model = tf.keras.models.load_model(modelpath, compile=True)

    # freebie: we set a random initial belief
    belief = np.ones((31,))

    correct_count = 0.0
    fp = 0.0
    fn = 0.0
    # for each time step in the trajectory
    # the target is the next observation
    predicted_targs = []

    for action in obs:

        prev_belief = belief

        # then predict the next window, given the previous belief and the next target
        filter_input = np.append(prev_belief, action, axis=0)
        filter_input = np.expand_dims(filter_input, axis=0)
        inference = model.predict(filter_input)
        belief = inference[0]

        roc = 0.249
        predicted_targs += [1 if belief[0] > roc else 0]
        print "True: %d | Predicted: %d (%.3f)" % (action, 1 if belief[0] > roc else 0, belief[0])

    n_correct = 0
    n_total = 0
    tp = 0
    fn = 0
    fp = 0
    for p in zip(obs, predicted_targs): 
      if p[0] == p[1]:
        n_correct += 1
      if p[0] == 1 and p[1] == 1:
        tp += 1
      if p[0] == 1 and p[1] == 0:
        fn += 1
      if p[0] == 0 and p[1] == 1:
        fp += 1
      n_total += 1


    print "\n\nRESULTS:"
    print "False total: " + str(len(obs) - correct_count)
    print "False positive: " + str(fp)
    print "False negative: " + str(fn)
    print "True positive: " + str(tp)

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = (2 * precision * recall) / float(precision + recall)

    print "%d correct out of %d total (%2.3f accuracy)" % (n_correct, n_total, 100*n_correct/float(n_total))
    print "  Precision: %.3f    Recall: %.3f    F1 score: %.3f" % (precision, recall, f1)
    print "  Skew: %.3f%% of targets are 0" % ((1 - sum(obs) / n_total) * 100)
    return predicted_targs

    

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="evaluate an inference machine")
    parser.add_argument('songpath', help="path to filter function model", type=str)
    parser.add_argument('modelpath', help="path to song to test", type=str)

    args = parser.parse_args()

    main(args.songpath, args.modelpath)
    sys.exit(0)
