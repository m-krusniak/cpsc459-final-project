
from process_midi import *
from save_midi import *
from nn_model import *
import tensorflow as tf
import evaluate_inference_machine_filter as eimf


# (train_data, train_targs, val_data, val_targs, test_data, test_targs) = separate_data(*load_all_songs('data/rock/', memory_length=16, drum=38))

# model = train(train_data, train_targs, val_data, val_targs)
# evaluate(model, test_data, test_targs)

# model = load('/home/miles/Academics/CPSC_459/final-project/logs/log_11-30-2020-20-50/best_face_weights.h5')
# print test_data[0:20]
# print test_targs[0:20]


# examine_song('/home/miles/Academics/CPSC_459/final-project/data/rock/10_rock-folk_90_beat_4-4_4.midi')

# test1 = load_song('/home/miles/Academics/CPSC_459/final-project/data/genre/train/rock/10_rock-folk_90_beat_4-4_4.midi', memory_length=16, drum=38)
# test1_data = np.reshape(test1[0], (len(test1[0]), 63))
# test1_targs = np.reshape(test1[1], (len(test1[1]), 1))



# for d in test1_data:
#   print d[-1]

# print "-=====================================-"

# for t in test1_targs:
#   print t[0]


# # # pred = eimf.main('/home/miles/Academics/CPSC_459/final-project/data/rock/10_rock-folk_90_beat_4-4_4.midi', '/home/miles/Academics/CPSC_459/final-project/filter_fn.h5')
# pred = evaluate(model, test1_data, test1_targs)

# save_song('test7_pred_hat.mid', np.reshape(pred, (len(pred), 1)), drum=42)
# save_song('test7_orig_hat.mid', np.reshape(test1_targs, (len(test1_targs), 1)), drum=42)

