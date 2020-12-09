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
    true_targs = [0] # offset by one

    for t in range(0,len(obs)-1):
        action = obs[t]
        next_action = obs[t+1]
        prev_belief = belief

        # then predict the next window, given the previous belief and the next target
        filter_input = np.append(prev_belief, action, axis=0)
        filter_input = np.expand_dims(filter_input, axis=0)
        inference = model.predict(filter_input)
        belief = inference[0]

        roc = 0.0003
        predicted_targs += [1 if belief[0] < roc else 0]
        true_targs += [next_action]
        print ("True: %d | Predicted: %d (%.10f) " % (next_action, 1 if belief[0] > roc else 0, belief[0])) + str(belief)

    n_total = 0
    tp = 0
    fn = 0
    fp = 0
    for p in zip(true_targs, predicted_targs):
      if p[0] == 1 and p[1] == 1: tp += 1
      if p[0] == 1 and p[1] == 0: fn += 1
      if p[0] == 0 and p[1] == 1: fp += 1
      n_total += 1

    return (predicted_targs, tp, fn, fp, n_total)



if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="evaluate an inference machine")
    parser.add_argument('songpath', help="path to filter function model", type=str)
    parser.add_argument('modelpath', help="path to song to test", type=str)

    args = parser.parse_args()

    main(args.songpath, args.modelpath)
    sys.exit(0)
