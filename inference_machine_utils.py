#!/usr/bin/env python

import argparse
import os
import sys
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

from process_midi import *


def binary_crossentropy(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true[1], y_pred[1])

def get_trajectories(midi_dir, n=1, drum=36):
    # check that the directory specified by midi_dir exists
    assert os.path.isdir(midi_dir)

    trajectories = []
    # get the first N songs as trajectories
    for file in os.listdir(midi_dir):
        if (len(trajectories) >= n): break
        data, targ = load_song(midi_dir + str(file), drum)

        if len(data) != 0 and len(targ) != 0:
            hints = np.array(data)
            obs = np.array(targ)
            # obs = np.append(np.array(0),obs)
            # obs = obs[range(0,len(targ))]
            obs = np.expand_dims(obs, axis=1)

            tau = (hints, obs)
            trajectories.append(tau)


    return trajectories


def build_filter_mlp(input_shape):

    input = tf.keras.layers.Input(shape=input_shape, name="inputs")
    hidden1 = tf.keras.layers.Dense(256, activation="relu", use_bias=True)(input)
    hidden2 = tf.keras.layers.Dense(128, activation="relu", use_bias=True)(hidden1)
    hidden3 = tf.keras.layers.Dense(64, activation="relu", use_bias=True)(hidden2)
    hidden4 = tf.keras.layers.Dense(32, activation="relu", use_bias=True)(hidden3)

    output = tf.keras.layers.Dense(31, activation="sigmoid", use_bias=True)(hidden4)
    model = tf.keras.models.Model(inputs=input, outputs=output, name="filter")
    return model


def train_mlp_dagger(model, trajectories, dagger_n=50):

    # Initialize dataset: D <- D_0 <- Null
    dataset_features = np.empty((0,32), dtype=np.float64)
    dataset_targets = np.empty((0,31), dtype=np.float64)

    # Initialize model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='mae',
                  metrics=['mse'])

    # Initialize predicted belief
    belief = np.ones((31,))

    # DAgger training
    for n in range(0,dagger_n):
        print "DAgger STEP: %d" % (n)
        dataset_n_features = np.empty((0,32), dtype=np.float64)
        dataset_n_targets = np.empty((0,31), dtype=np.float64)

        # roll out belief propagation on each trajectory
        for tau in trajectories:
            for t in range(0,len(tau[0])-31):
                prev_belief = belief

                # combine previous belief with current observation
                # predict belief
                filter_input = np.append(tau[1][t], prev_belief, axis=0)
                filter_input = np.expand_dims(filter_input, axis=0)
                inference = model.predict(filter_input)
                belief = inference[0] # we're only predicting one sample

                # create dataset D_n
                # as features, add the predicted belief and the observation
                z = np.append(tau[1][t], belief)
                z = np.expand_dims(z, axis=0)
                dataset_n_features = np.concatenate((dataset_n_features, z), axis=0)
                # as targets, add the next hint
                phi = np.expand_dims(tau[0][t+31], axis=0)
                dataset_n_targets = np.concatenate((dataset_n_targets, phi), axis=0)

        # DAgger step: aggregate D = D U D_n
        dataset_features = np.concatenate((dataset_features, dataset_n_features),axis=0)
        dataset_targets = np.concatenate((dataset_targets, dataset_n_targets),axis=0)

        # Train new hypothesis on D
        model.fit(dataset_features, dataset_targets, batch_size=2004, epochs=1)
    # return best hypothesis on validation trajectories
    return model


def train_forest_dagger(trajectories, dagger_n=50):

    # Initialize dataset: D <- D_0 <- Null
    dataset_features = np.empty((0,32), dtype=np.float64)
    dataset_targets = np.empty((0,31), dtype=np.float64)

    # Specify random forest model
    forest = RandomForestRegressor(n_estimators=32, max_depth=3)

    # Initialize predicted belief
    belief = np.ones((31,))
    belief_init = np.append(belief, np.array(0.0))
    belief_init = np.expand_dims(belief_init, axis=0)

    target_init = np.ones((1,31))
    forest.fit(belief_init, target_init)

    # DAgger training
    for n in range(0,dagger_n):
        print "DAgger STEP: %d" % (n)
        dataset_n_features = np.empty((0,32), dtype=np.float64)
        dataset_n_targets = np.empty((0,31), dtype=np.float64)

        # roll out belief propagation on each trajectory
        for tau in trajectories:
            for t in range(0,len(tau[0])-31):
                prev_belief = belief

                # combine previous belief with current observation
                # predict belief
                filter_input = np.append(tau[1][t], prev_belief, axis=0)
                filter_input = np.expand_dims(filter_input, axis=0)
                inference = forest.predict(filter_input)
                belief = inference[0] # we're only predicting one sample

                # create dataset D_n
                # as features, add the predicted belief and the observation
                z = np.append(tau[1][t], belief)
                z = np.expand_dims(z, axis=0)
                dataset_n_features = np.concatenate((dataset_n_features, z), axis=0)
                # as targets, add the next hint
                phi = np.expand_dims(tau[0][t+31], axis=0)
                dataset_n_targets = np.concatenate((dataset_n_targets, phi), axis=0)


        # DAgger step: aggregate D = D U D_n
        dataset_features = np.concatenate((dataset_features, dataset_n_features),axis=0)
        dataset_targets = np.concatenate((dataset_targets, dataset_n_targets),axis=0)

        # Train new hypothesis on D
        forest.fit(dataset_features, dataset_targets)
    # return best hypothesis on validation trajectories
    return forest


def main(midi_dir, traj_i, dagger_n):
    # get trajectories
    trajectories = get_trajectories(midi_dir, traj_i)

    # build model
    # filter_fn = build_filter_mlp((32,))

    # DAgger training
    trained_forest = train_forest_dagger(trajectories, dagger_n)
    # trained_filter_fn = train_mlp_dagger(filter_fn, trajectories, dagger_n)

    # save model to file
    # trained_filter_fn.save('filter_fn.h5')

if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description="Train an inference machine filter function")
    parser.add_argument('data', help="path to MIDI data", type=str)
    parser.add_argument('i', help="number of trajectories", type=int, default=5)
    parser.add_argument('N', help="number of DAgger iterations", type=int, default=5)
    args = parser.parse_args()

    # run main()
    main(args.data, args.i, args.N)
    sys.exit(0)
