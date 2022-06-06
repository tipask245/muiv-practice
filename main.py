import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf

#Загрузка датасета
(x_train, y_train), (x_test, y_test) = load_data()

#Нормализация данных
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_test = x_test.astype('float32') / 255
x_train = x_train.astype('float32') / 255

#вывод датасета
plt.figure()
for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
plt.show()


#Структура сверточной модели
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

#Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Обучение модели, чтобы изменить кол-во эпох нужно заменить значение epochs
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
model.save('digits.h5')

