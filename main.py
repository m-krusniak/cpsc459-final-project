
from process_midi import *
from nn_model import *
import tensorflow as tf


(train_data, train_targs, val_data, val_targs, test_data, test_targs) = separate_data(*load_all_songs('data/rock1/'))

# model = train(train_data, train_targs, val_data, val_targs)

model = load('/home/miles/Academics/CPSC_459/final-project/logs/log_11-28-2020-21-22/best_face_weights.h5')
# print test_data[0:20]
# print test_targs[0:20]

# test1 = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * 8)[1:]
# test1 = np.reshape(test1, (1, 31))

test1 = load_song('/home/miles/Academics/CPSC_459/final-project/examine/Test_2__Bass_One_Three.mid')

test1_data = np.reshape(test1[0], (len(test1[0]), 31))
test1_targs = np.reshape(test1[1], (len(test1[1]), 1))

print test1_data
print np.reshape(test1_targs, (len(test1_targs), ))

# for i in np.reshape(test_targs, (len(test_targs), )):
  # print i


print evaluate(model, test1_data, test1_targs)

