
from process_midi import *
from save_midi import *
from nn_model import *
import tensorflow as tf


(train_data, train_targs, val_data, val_targs, test_data, test_targs) = separate_data(*load_all_songs('data/rock2/', memory_length=16))

model = train(train_data, train_targs, val_data, val_targs)
evaluate(model, test_data, test_targs)

# model = load('/home/miles/Academics/CPSC_459/final-project/logs/log_11-30-2020-19-28/best_face_weights.h5')
# print test_data[0:20]
# print test_targs[0:20]


# test1 = load_song('/home/miles/Academics/CPSC_459/final-project/examine/Test_4__Bass_One_And_Three.mid', memory_length=16)
# examine_song('/home/miles/Academics/CPSC_459/final-project/examine/Test_3__Bass_And_Three.mid')

# save_song("test1.mid", np.reshape(test1[1], (len(test1[1]))))


# examine_song('/home/miles/Academics/CPSC_459/final-project/examine/Test_2__Bass_One_Three.mid')
# print "-------------------"
# examine_song('test1.mid')

# test1_data = np.reshape(test1[0], (len(test1[0]), 63))
# test1_targs = np.reshape(test1[1], (len(test1[1]), 1))

# # print test1_data
# # print np.reshape(test1_targs, (len(test1_targs), ))

# # # for i in np.reshape(test_targs, (len(test_targs), )):
# #   # print i


# evaluate(model, test_data, test_targs)
# save_song('test4_pred.mid', test_pred)

