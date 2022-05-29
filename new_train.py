import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_test = x_test.astype('float32') / 255
x_train = x_train.astype('float32') / 255

loaded_model = tf.keras.models.load_model('digits.h5')

# чтобы изменить кол-во эпох обучения нужно заменить значение epochs
loaded_model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
loaded_model.save('digits.h5')
















