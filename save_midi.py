import mido
import collections
from playsound import playsound
import numpy as np
import datetime
import os
import tensorflow as tf

def save_song(filename, sequence, drum=35, bps=7.0/3.0, ticks_per_beat=4):
  
  file = mido.MidiFile()
  file.ticks_per_beat=ticks_per_beat 
  track = mido.MidiTrack()
  file.tracks.append(track)

  # There are some meta messages that occur at the beginning of midi files that we can't miss.
  # First, time signature. We pretty much have to assume 4/4 for now.
  track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

  # Second, key signature. This actually doesn't matter for unpitched percussion.
  track.append(mido.MetaMessage('key_signature', key='C', time=0))

  # Third, tempo. Midi tempo is given in microseconds per beat, which is actually pretty convenient for us.
  track.append(mido.MetaMessage('set_tempo', tempo=int(1.0/float(bps)*1000000.0)))

  # Fourth, midi port. Port 0 is fine.
  track.append(mido.MetaMessage('midi_port', port=0, time=0))

  # Some channel related messages:
  track.append(mido.Message('control_change', channel=9, control=121, value=0, time=0))
  track.append(mido.Message('program_change', channel=9, program=48, time=0))
  track.append(mido.Message('control_change', channel=9, control=7, value=100, time=0))
  track.append(mido.Message('control_change', channel=9, control=10, value=64, time=0))
  track.append(mido.Message('control_change', channel=9, control=91, value=0, time=0))
  track.append(mido.Message('control_change', channel=9, control=93, value=0, time=0))

  # Then, normal messages.
  # Critical note: mido gives us delta time in seconds on input, but it's actually represented in ticks for midi.
  #   (I actually did not know that midi had a concept of ticks outside of our own until now. They were independently created.)

  t = 0
  for i in sequence:
    if i == 1:
      # note that unfortunately we do lose information on loudness / dynamics
      # and all notes occur precisely on a tick, which is not always so good (e.g., triplets)
      track.append(mido.Message('note_on', note=drum, velocity=70, time=t, channel=9))
      t = 0 # timing is relative to the last message
    t += 1

  # And, crucially, an end of track
  track.append(mido.MetaMessage('end_of_track', time=t+1))


  file.save(filename)