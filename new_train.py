import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

loaded_model = tf.keras.models.load_model('digits.h5')

# чтобы изменить кол-во эпох обучения нужно заменить значение epochs
loaded_model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

# v -- функция для тестирования модели на тестовом наборе (раскомментировать)
# print(loaded_model.evaluate(x_test, y_test))















