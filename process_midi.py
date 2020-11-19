import mido
import collections
from playsound import playsound
import numpy as np
import datetime
import os
import tensorflow as tf

def load_data():

  drum = 36 # 36 is bass
  memory_length = 8 # beats
  ticks_per_beat = 16 # ticks
  t = 0
  r = 0
  data = []
  targ = []
  ndata = 0
  nsongs = 0

  for file in os.listdir('data/rock1/'):
    nsongs += 1
    mid = mido.MidiFile('data/rock1/' + str(file))
    for msg in mid:
      if msg.type == 'set_tempo':
        bps = mido.tempo2bpm(msg.tempo)/60.0 # beats per second 
        model_input = np.zeros(int(memory_length * ticks_per_beat)+1)
        print "SET TEMPO %.2f BPS; now using %d inputs (%d beats, %d ticks per beat)" % (bps, len(model_input), memory_length, ticks_per_beat)

      t += msg.time
      if msg.type == 'note_on' and msg.note == drum:
        d = int((msg.time+r) * bps * ticks_per_beat)
        r = 0
        for i in range(0, d):
          data += [model_input[:-2]]
          targ += [[model_input[-1]]]
          ndata += 1
          model_input[0] = 0
          model_input = np.roll(model_input, -1)
        model_input[len(model_input)-1] = 1
        print str(t) + "(" + str(float(d)/ticks_per_beat) + " beats): " + str(model_input)
      else: 
        r += msg.time
    if ndata > 50000: break 

  print "LOADED %d SONGS (%d examples)" % (nsongs, ndata)
  return data, targ

def separate_data(data, targ):

  inputs = np.array(data)
  targs = np.array(targ)
  ndata = len(inputs)

  ntrain = int(ndata*.7)
  nval = int(ndata * .2)
  ntest = int(ndata * .1)

  train_data = inputs[0:ntrain]
  train_targs = targs[0:ntrain]
  val_data = inputs[ntrain:ntrain+nval]
  val_targs = targs[ntrain:ntrain+nval]
  test_data = inputs[ntrain+nval:ntrain+nval+ntest]
  test_targs = targs[ntrain+nval:ntrain+nval+ntest]
  return (train_data, train_targs, val_data, val_targs, test_data, test_targs)