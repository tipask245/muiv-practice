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

# def test_predict(img):
#     img = img.reshape(1, 28, 28, 1)
#     res = loaded_model.predict([img])[0]
#     return np.argmax(res), max(res)
#
# for i in range(15):
#     image_test = x_test[i]
#     plt.figure()
#     plt.imshow(image_test, cmap='gray')
#     plt.show()
#     print(test_predict(image_test))

# v -- функция для тестирования модели на тестовом наборе (раскомментировать)
# print(loaded_model.evaluate(x_test, y_test))















