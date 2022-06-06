from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Загрузка модели
loaded_model = tf.keras.models.load_model('digits.h5')

#Функция предсказаний с визуализацией данных
def predict_d(img):
    img = img.resize((28, 28))
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    img = np.invert(np.array(img))
    img = img.reshape(1, 28, 28, 1)
    img = img/255
    res = loaded_model.predict([img])[0]
    plt.title(f'Цифра={np.argmax(res)}   Вероятность={max(res)}')
    plt.show()
    return np.argmax(res), max(res)

# Цикл для вызова функции предсказания для каждого изображения
for i in range(10):
    image = load_img(f'digit/{i}.png', color_mode="grayscale")
    print(predict_d(image))