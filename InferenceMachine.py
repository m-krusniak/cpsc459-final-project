#!/usr/bin/env python

import os
import sys
import pickle

from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

from process_midi import *
from inference_machine_utils import *



class InferenceMachine:

    def __init__(self, learner, drum=36):
        self.learner = learner
        self.drum = drum
        self.roc = 0.5


    def train(self, data_dir, i, N):
        # get trajectories
        trajectories = get_trajectories(data_dir, i, self.drum)

        # build either MLP or random forest
        if self.learner == 'MLP':
            model = build_filter_mlp((32,))
            # DAgger training
            self.model = train_mlp_dagger(model, trajectories, N)

        elif self.learner == 'RF':
            self.model = train_forest_dagger(trajectories, N)

        else:
            print "ERROR: Unknown learner type"


    def evaluate(self, songpath):
        # load song
        hints, obs = load_song(songpath)
        obs = np.array(obs)

        # freebie: we set a random initial belief
        belief = np.ones((31,))

        # for each time step in the trajectory
        # the target is the next observation
        predicted_targs = []

        for t in range(0,len(obs)-1):
            action = obs[t]
            next_action = obs[t+1]
            prev_belief = belief

            # then predict the next window, given the previous belief and the next target
            filter_input = np.append(prev_belief, [action], axis=0)
            filter_input = np.expand_dims(filter_input, axis=0)
            inference = self.model.predict(filter_input)
            belief = inference[0]

            predicted_targs += [1 if belief[0] > self.roc else 0]

        n_correct = 0
        n_total = 0
        tp = 0
        fn = 0
        fp = 0
        for p in zip(obs, predicted_targs):
            if p[0] == 1 and p[1] == 1: tp += 1
            if p[0] == 1 and p[1] == 0: fn += 1
            if p[0] == 0 and p[1] == 1: fp += 1
            n_total += 1

        return (predicted_targs, tp, fn, fp, n_total)

    def export(self, filename):
        if self.learner == "RF":
            file = open(filename, 'w+')
            pickle.dump(self.model, file)
        if self.learner == "MLP":
            self.model.save(filename)

    def load(self, filename):
        if self.learner == "RF":
            file = open(filename)
            pickle.load(file)
        if self.learner == "MLP":
            self.model = tf.keras.models.load_model(filename, compile=True)
