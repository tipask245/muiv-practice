import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = load_data()
print(type(x_train))

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_test = x_test.astype('float32') / 255
x_train = x_train.astype('float32') / 255

#вывод датасета
plt.figure()
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])

plt.show()


#Структура сверточной модели
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    # tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    # tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# чтобы изменить кол-во эпох обучения нужно заменить значение epochs
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
model.save('digits.h5')

