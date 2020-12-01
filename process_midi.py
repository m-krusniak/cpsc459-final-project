import mido
import collections
from playsound import playsound
import numpy as np
import datetime
import os
import tensorflow as tf

def examine_song(filename):
  for msg in mido.MidiFile(filename):
    if not msg.is_meta: print ("[%d] " % msg.channel) + str(msg)

def load_song(filename, drum=36, memory_length=8, ticks_per_beat=4):
  t = 0
  r = 0
  data = []
  targ = []
  ndata = 0
  nsongs = 0
  bps = 7.0/3.0 # around two beats per second is a reasonable default, musically
  model_input = np.zeros(int(memory_length * ticks_per_beat)+1)

  mid = mido.MidiFile(filename)
  for msg in mid:
    if msg.type == 'set_tempo':
      bps = mido.tempo2bpm(msg.tempo)/60.0 # beats per second
      model_input = np.zeros(int(memory_length * ticks_per_beat)+1)

    t += msg.time
    if msg.type == 'note_on' and msg.note == drum and msg.velocity > 1.0:
      d = int((msg.time+r) * bps * ticks_per_beat + 0.5) # nice trick: int(x+0.5) rounds x to the nearest whole
      if d < 1: r += msg.time
      else:
        r = 0
        for i in range(0, d):
          data += [model_input[:-2]]
          targ += [[model_input[-1]]]
          ndata += 1
          model_input[0] = 0
          model_input = np.roll(model_input, -1)
        model_input[len(model_input)-1] = 1
    else: 
      r += msg.time

  print "LOADED SONG %s (%d examples)" % (filename, ndata)
  return data, targ

def load_all_songs(dirname, drum=36, memory_length=8, ticks_per_beat=4):

  t = 0
  r = 0
  data = []
  targ = []
  ndata = 0
  nsongs = 0

  for file in os.listdir(dirname):
    print "Loading song %s" % str(file)
    nsongs += 1
    mid = mido.MidiFile(dirname + str(file))
    for msg in mid:
      if msg.type == 'set_tempo':
        bps = mido.tempo2bpm(msg.tempo)/60.0 # beats per second
        model_input = np.zeros(int(memory_length * ticks_per_beat)+1)

      if msg.type == 'time_signature':
        # gotta constrict ourselves to 4/4 for now
        if msg.numerator != 4 or msg.denominator != 4: 
          print "  Aborted: time signature is %d/%d" % (msg.numerator, msg.denominator)
          break

      t += msg.time
      if msg.type == 'note_on' and msg.note == drum and msg.velocity > 1.0:
        d = int((msg.time+r) * bps * ticks_per_beat + 0.5) # nice trick: int(x+0.5) rounds x to the nearest whole
        if d < 1: r += msg.time
        else:
          r = 0
          for i in range(0, d):
            data += [model_input[:-2]]
            targ += [[model_input[-1]]]
            ndata += 1
            model_input[0] = 0
            model_input = np.roll(model_input, -1)
            print file
          model_input[len(model_input)-1] = 1
      else:
        r += msg.time
    if ndata > 50000: break

  print "LOADED %d SONGS (%d examples)" % (nsongs, ndata)
  return data, targ

def separate_data(data, targ):

  inputs = np.array(data)
  targs = np.array(targ)
  ndata = len(inputs)

  # Here's an issue: we probably shouldn't take exactly every 2 in 10 elements because there could be a pattern across 10
  # Really we should randomize, but I don't know off the top of my head how to simultaneously randomize two lists in python -
  # probably zip or something, but a simpler workaround is just to choose a number that has no musical significance - e.g,
  # a small prime.
  train_data = np.array([e for i, e in enumerate(inputs) if i % 19 < 12])
  train_targs = np.array([e for i, e in enumerate(targs) if i % 19 < 12])
  val_data = np.array([e for i, e in enumerate(inputs) if i % 19 >= 12 and i % 19 < 17])
  val_targs = np.array([e for i, e in enumerate(targs) if i % 19 >= 12 and i % 19 < 17])
  test_data = np.array([e for i, e in enumerate(inputs) if i % 19 >= 17])
  test_targs = np.array([e for i, e in enumerate(targs) if i % 19 >= 17])

  print train_data.shape

  # If you want to do it sequentially here's how it'd work.
  #  (but you don't; you want each set to contain parts of each song)

  # train_data = inputs[0:ntrain]
  # train_targs = targs[0:ntrain]
  # val_data = inputs[ntrain:ntrain+nval]
  # val_targs = targs[ntrain:ntrain+nval]
  # test_data = inputs[ntrain+nval:ntrain+nval+ntest]
  # test_targs = targs[ntrain+nval:ntrain+nval+ntest]
  return (train_data, train_targs, val_data, val_targs, test_data, test_targs)
