
from process_midi import *
from nn_model import *
import tensorflow as tf


# (train_data, train_targs, val_data, val_targs, test_data, test_targs) = separate_data(*load_data())

# model = train(data, targ)

model = load('/home/miles/Academics/CPSC_459/final-project/logs/log_11-18-2020-18-21/best_face_weights.h5')
# print test_data[0:20]
# print test_targs[0:20]

test1 = ([1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0] * 8)[1:]
test1 = np.reshape(test1, (1, 127))

print model.predict([test1], batch_size=4)

