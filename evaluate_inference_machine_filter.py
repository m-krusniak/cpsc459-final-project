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
    for action in obs:
        print action

        prev_belief = belief

        # then predict the next window, given the previous belief and the next target
        filter_input = np.append(prev_belief, action, axis=0)
        filter_input = np.expand_dims(filter_input, axis=0)
        inference = model.predict(filter_input)
        belief = inference[0]

        roc = 0.0027
        belief2 = belief
        belief2[belief > roc] = 1.0
        belief2[belief <= roc] = 0.0

        print belief
        if belief2[0] != action:
            if belief2[0] > action:
                fp += 1.0
            else:
                fn += 1.0
        else:
            correct_count += 1.0

    # after the loop is finished, add up the misses and the hits
    # print these out!
    print correct_count / len(obs)
    print len(obs) - correct_count
    print fp
    print fn

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="evaluate an inference machine")
    parser.add_argument('songpath', help="path to filter function model", type=str)
    parser.add_argument('modelpath', help="path to song to test", type=str)

    args = parser.parse_args()

    main(args.songpath, args.modelpath)
    sys.exit(0)
